use engine_codec::{Error, ErrorCode, Expected};
use engine_math::collections::{FixedState, HashMap};

use crate::format::{
    BvhNode, ClusterHeader, MANIFEST_MAGIC, MANIFEST_VERSION, MeshletManifest, PageSpan,
};

const HEADER_BYTES: usize = 64;
const GROUP_STRIDE_BYTES: usize = 24;
const CLUSTER_FIXED_STRIDE_BYTES: usize = 44;
const SPHERE_BYTES: usize = 16;
const BVH_NODE_BYTES: usize = 64;
const PAGE_SPAN_FULL_BYTES: usize = 12;
const PAGE_SPANS_COMPACT_U16: u8 = 0;
const PAGE_SPANS_FULL_U64_U32: u8 = 1;

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GroupKey {
    group_id: u32,
    parent_lod_sphere: [u32; 4],
    parent_error: u32,
}

impl GroupKey {
    fn from_cluster(c: &ClusterHeader) -> Self {
        Self {
            group_id: c.page_and_fallback[3],
            parent_lod_sphere: [
                c.parent_lod_sphere[0].to_bits(),
                c.parent_lod_sphere[1].to_bits(),
                c.parent_lod_sphere[2].to_bits(),
                c.parent_lod_sphere[3].to_bits(),
            ],
            parent_error: c.errors_and_cone[1],
        }
    }
}

struct ManifestHeader {
    source_tri_count: u32,
    lod_levels: u32,
    root_page_count: u32,
    page_count: u32,
    cluster_count: usize,
    bvh_count: usize,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    group_count: usize,
    lod_diff_count: usize,
}

fn read_u16(bytes: &[u8], offset: &mut usize, path: &str) -> Result<u16, Error> {
    let end = offset.saturating_add(2);
    let slice = bytes
        .get(*offset..end)
        .ok_or_else(|| invalid(path).expected(Expected::Present))?;
    *offset = end;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u32(bytes: &[u8], offset: &mut usize, path: &str) -> Result<u32, Error> {
    let end = offset.saturating_add(4);
    let slice = bytes
        .get(*offset..end)
        .ok_or_else(|| invalid(path).expected(Expected::Present))?;
    *offset = end;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_u64(bytes: &[u8], offset: &mut usize, path: &str) -> Result<u64, Error> {
    let end = offset.saturating_add(8);
    let slice = bytes
        .get(*offset..end)
        .ok_or_else(|| invalid(path).expected(Expected::Present))?;
    *offset = end;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn read_f32(bytes: &[u8], offset: &mut usize, path: &str) -> Result<f32, Error> {
    Ok(f32::from_bits(read_u32(bytes, offset, path)?))
}

fn write_transposed_u32(out: &mut Vec<u8>, values: &[u32]) {
    let n = values.len();
    let base = out.len();
    out.resize(base + n * 4, 0);
    let dst = &mut out[base..];
    for (i, &v) in values.iter().enumerate() {
        let b = v.to_le_bytes();
        dst[i] = b[0];
        dst[n + i] = b[1];
        dst[2 * n + i] = b[2];
        dst[3 * n + i] = b[3];
    }
}

fn read_transposed_u32(
    bytes: &[u8],
    offset: &mut usize,
    count: usize,
    path: &str,
) -> Result<Vec<u32>, Error> {
    let byte_len = count
        .checked_mul(4)
        .ok_or_else(|| invalid(path).got(count))?;
    let end = offset
        .checked_add(byte_len)
        .ok_or_else(|| invalid(path).expected(Expected::Present))?;
    let slice = bytes
        .get(*offset..end)
        .ok_or_else(|| invalid(path).expected(Expected::Present))?;
    *offset = end;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(u32::from_le_bytes([
            slice[i],
            slice[count + i],
            slice[2 * count + i],
            slice[3 * count + i],
        ]));
    }
    Ok(out)
}

fn cluster_lod_eq_bound(c: &ClusterHeader) -> bool {
    c.lod_sphere[0].to_bits() == c.bound_sphere[0].to_bits()
        && c.lod_sphere[1].to_bits() == c.bound_sphere[1].to_bits()
        && c.lod_sphere[2].to_bits() == c.bound_sphere[2].to_bits()
        && c.lod_sphere[3].to_bits() == c.bound_sphere[3].to_bits()
}

fn collect_groups(clusters: &[ClusterHeader]) -> (Vec<GroupKey>, Vec<u32>, usize) {
    let mut map: HashMap<GroupKey, u32> =
        HashMap::with_capacity_and_hasher(clusters.len(), FixedState::default());
    let mut groups = Vec::new();
    let mut indices = Vec::with_capacity(clusters.len());
    let mut lod_diff_count = 0_usize;
    for c in clusters {
        if !cluster_lod_eq_bound(c) {
            lod_diff_count += 1;
        }
        let key = GroupKey::from_cluster(c);
        let idx = if let Some(&existing) = map.get(&key) {
            existing
        } else {
            let next = groups.len() as u32;
            groups.push(key);
            map.insert(key, next);
            next
        };
        indices.push(idx);
    }
    (groups, indices, lod_diff_count)
}

fn page_spans_are_compact(spans: &[PageSpan]) -> bool {
    let mut expected_offset = 0_u64;
    for span in spans {
        if span.byte_offset != expected_offset || span.byte_len > u32::from(u16::MAX) {
            return false;
        }
        let Some(next) = expected_offset.checked_add(u64::from(span.byte_len)) else {
            return false;
        };
        expected_offset = next;
    }
    true
}

fn encoded_body_len(
    group_count: usize,
    cluster_count: usize,
    lod_diff_count: usize,
    bvh_count: usize,
) -> Option<usize> {
    let g_bytes = group_count.checked_mul(GROUP_STRIDE_BYTES)?;
    let c_bytes = cluster_count
        .checked_mul(CLUSTER_FIXED_STRIDE_BYTES)?
        .checked_add(cluster_count.div_ceil(8))?
        .checked_add(lod_diff_count.checked_mul(SPHERE_BYTES)?)?;
    let b_bytes = bvh_count.checked_mul(BVH_NODE_BYTES)?;
    g_bytes.checked_add(c_bytes)?.checked_add(b_bytes)
}

fn read_manifest_header(bytes: &[u8], offset: &mut usize) -> Result<ManifestHeader, Error> {
    if bytes.len() < HEADER_BYTES {
        return Err(invalid("$.manifest")
            .at_least(HEADER_BYTES as u64)
            .got(bytes.len()));
    }
    if bytes[0..4] != MANIFEST_MAGIC {
        return Err(invalid("$.manifest.magic")
            .expected(Expected::Matching)
            .got(String::from_utf8_lossy(&bytes[0..4]).into_owned()));
    }
    *offset = 4;
    let version = read_u32(bytes, offset, "$.manifest.version")?;
    if version != MANIFEST_VERSION {
        return Err(invalid("$.manifest.version")
            .expected(Expected::Supported)
            .got(version));
    }
    let source_tri_count = read_u32(bytes, offset, "$.manifest.source_tri_count")?;
    let lod_levels = read_u32(bytes, offset, "$.manifest.lod_levels")?;
    let root_page_count = read_u32(bytes, offset, "$.manifest.root_page_count")?;
    let page_count = read_u32(bytes, offset, "$.manifest.page_count")?;
    let cluster_count = read_u32(bytes, offset, "$.manifest.cluster_count")? as usize;
    let bvh_count = read_u32(bytes, offset, "$.manifest.bvh_count")? as usize;
    let bounds_min = [
        read_f32(bytes, offset, "$.manifest.bounds_min[0]")?,
        read_f32(bytes, offset, "$.manifest.bounds_min[1]")?,
        read_f32(bytes, offset, "$.manifest.bounds_min[2]")?,
    ];
    let bounds_max = [
        read_f32(bytes, offset, "$.manifest.bounds_max[0]")?,
        read_f32(bytes, offset, "$.manifest.bounds_max[1]")?,
        read_f32(bytes, offset, "$.manifest.bounds_max[2]")?,
    ];
    let group_count = read_u32(bytes, offset, "$.manifest.group_count")? as usize;
    let lod_diff_count = read_u32(bytes, offset, "$.manifest.lod_diff_count")? as usize;
    Ok(ManifestHeader {
        source_tri_count,
        lod_levels,
        root_page_count,
        page_count,
        cluster_count,
        bvh_count,
        bounds_min,
        bounds_max,
        group_count,
        lod_diff_count,
    })
}

fn write_header(
    out: &mut Vec<u8>,
    manifest: &MeshletManifest,
    group_count: u32,
    lod_diff_count: u32,
) {
    out.extend_from_slice(&MANIFEST_MAGIC);
    out.extend_from_slice(&MANIFEST_VERSION.to_le_bytes());
    out.extend_from_slice(&manifest.source_tri_count.to_le_bytes());
    out.extend_from_slice(&manifest.lod_levels.to_le_bytes());
    out.extend_from_slice(&manifest.root_page_count.to_le_bytes());
    out.extend_from_slice(&manifest.page_count.to_le_bytes());
    out.extend_from_slice(&(manifest.clusters.len() as u32).to_le_bytes());
    out.extend_from_slice(&(manifest.bvh.len() as u32).to_le_bytes());
    for value in manifest.bounds_min {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    for value in manifest.bounds_max {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    out.extend_from_slice(&group_count.to_le_bytes());
    out.extend_from_slice(&lod_diff_count.to_le_bytes());
}

fn write_groups(out: &mut Vec<u8>, groups: &[GroupKey]) {
    let n = groups.len();
    let mut ids = Vec::with_capacity(n);
    let mut ps = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut pe = Vec::with_capacity(n);
    for g in groups {
        ids.push(g.group_id);
        for (col, &val) in ps.iter_mut().zip(&g.parent_lod_sphere) {
            col.push(val);
        }
        pe.push(g.parent_error);
    }
    write_transposed_u32(out, &ids);
    for col in &ps {
        write_transposed_u32(out, col);
    }
    write_transposed_u32(out, &pe);
}

fn write_clusters(
    out: &mut Vec<u8>,
    clusters: &[ClusterHeader],
    group_indices: &[u32],
    lod_diff_count: usize,
) {
    let n = clusters.len();
    let mut bs = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut lod_eq = vec![0_u8; n.div_ceil(8)];
    let mut ld = [
        Vec::with_capacity(lod_diff_count),
        Vec::with_capacity(lod_diff_count),
        Vec::with_capacity(lod_diff_count),
        Vec::with_capacity(lod_diff_count),
    ];
    let mut self_err = Vec::with_capacity(n);
    let mut cone = Vec::with_capacity(n);
    let mut meta = Vec::with_capacity(n);
    let mut page_id = Vec::with_capacity(n);
    let mut offset_flags = Vec::with_capacity(n);
    let mut fallback = Vec::with_capacity(n);
    for (i, c) in clusters.iter().enumerate() {
        for (col, &val) in bs.iter_mut().zip(&c.bound_sphere) {
            col.push(val.to_bits());
        }
        if cluster_lod_eq_bound(c) {
            lod_eq[i >> 3] |= 1_u8 << (i & 7);
        } else {
            for (col, &val) in ld.iter_mut().zip(&c.lod_sphere) {
                col.push(val.to_bits());
            }
        }
        self_err.push(c.errors_and_cone[0]);
        cone.push(c.errors_and_cone[2]);
        meta.push(c.errors_and_cone[3]);
        page_id.push(c.page_and_fallback[0]);
        offset_flags.push(c.page_and_fallback[1]);
        fallback.push(c.page_and_fallback[2]);
    }
    for col in &bs {
        write_transposed_u32(out, col);
    }
    out.extend_from_slice(&lod_eq);
    for col in &ld {
        write_transposed_u32(out, col);
    }
    write_transposed_u32(out, &self_err);
    write_transposed_u32(out, &cone);
    write_transposed_u32(out, &meta);
    write_transposed_u32(out, &page_id);
    write_transposed_u32(out, &offset_flags);
    write_transposed_u32(out, &fallback);
    write_transposed_u32(out, group_indices);
}

fn write_bvh(out: &mut Vec<u8>, bvh: &[BvhNode]) {
    let n = bvh.len();
    let mut sphere = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut child = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut max_err = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    let mut meta = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for node in bvh {
        for k in 0..4 {
            sphere[k].push(node.sphere[k].to_bits());
            child[k].push(node.child_or_cluster[k]);
            max_err[k].push(node.max_parent_error[k].to_bits());
            meta[k].push(node.meta[k]);
        }
    }
    for col in &sphere {
        write_transposed_u32(out, col);
    }
    for col in &child {
        write_transposed_u32(out, col);
    }
    for col in &max_err {
        write_transposed_u32(out, col);
    }
    for col in &meta {
        write_transposed_u32(out, col);
    }
}

fn write_page_spans(out: &mut Vec<u8>, spans: &[PageSpan], compact: bool) {
    if compact {
        out.push(PAGE_SPANS_COMPACT_U16);
        for span in spans {
            out.extend_from_slice(&(span.byte_len as u16).to_le_bytes());
        }
    } else {
        out.push(PAGE_SPANS_FULL_U64_U32);
        for span in spans {
            out.extend_from_slice(&span.byte_offset.to_le_bytes());
            out.extend_from_slice(&span.byte_len.to_le_bytes());
        }
    }
}

fn read_transposed_u32x4(
    bytes: &[u8],
    offset: &mut usize,
    count: usize,
    path: &str,
) -> Result<[Vec<u32>; 4], Error> {
    Ok([
        read_transposed_u32(bytes, offset, count, path)?,
        read_transposed_u32(bytes, offset, count, path)?,
        read_transposed_u32(bytes, offset, count, path)?,
        read_transposed_u32(bytes, offset, count, path)?,
    ])
}

fn read_groups(
    bytes: &[u8],
    offset: &mut usize,
    group_count: usize,
) -> Result<Vec<GroupKey>, Error> {
    let ids = read_transposed_u32(bytes, offset, group_count, "$.manifest.groups.id")?;
    let ps = read_transposed_u32x4(bytes, offset, group_count, "$.manifest.groups.sphere")?;
    let pe = read_transposed_u32(bytes, offset, group_count, "$.manifest.groups.error")?;
    let mut groups = Vec::with_capacity(group_count);
    for i in 0..group_count {
        groups.push(GroupKey {
            group_id: ids[i],
            parent_lod_sphere: [ps[0][i], ps[1][i], ps[2][i], ps[3][i]],
            parent_error: pe[i],
        });
    }
    Ok(groups)
}

fn read_clusters(
    bytes: &[u8],
    offset: &mut usize,
    cluster_count: usize,
    lod_diff_count: usize,
    groups: &[GroupKey],
) -> Result<Vec<ClusterHeader>, Error> {
    let bs = read_transposed_u32x4(bytes, offset, cluster_count, "$.manifest.clusters.bound")?;
    let bitset_len = cluster_count.div_ceil(8);
    let bitset_end = offset
        .checked_add(bitset_len)
        .ok_or_else(|| invalid("$.manifest.clusters.lod_eq").expected(Expected::Present))?;
    let bitset = bytes
        .get(*offset..bitset_end)
        .ok_or_else(|| invalid("$.manifest.clusters.lod_eq").expected(Expected::Present))?;
    *offset = bitset_end;
    let ld = read_transposed_u32x4(bytes, offset, lod_diff_count, "$.manifest.clusters.lod")?;
    let self_err = read_transposed_u32(bytes, offset, cluster_count, "$.manifest.clusters.err")?;
    let cone = read_transposed_u32(bytes, offset, cluster_count, "$.manifest.clusters.cone")?;
    let meta = read_transposed_u32(bytes, offset, cluster_count, "$.manifest.clusters.meta")?;
    let page_id = read_transposed_u32(bytes, offset, cluster_count, "$.manifest.clusters.page")?;
    let offset_flags =
        read_transposed_u32(bytes, offset, cluster_count, "$.manifest.clusters.flags")?;
    let fallback =
        read_transposed_u32(bytes, offset, cluster_count, "$.manifest.clusters.fallback")?;
    let group_idx = read_transposed_u32(bytes, offset, cluster_count, "$.manifest.clusters.group")?;
    let mut clusters = Vec::with_capacity(cluster_count);
    let mut diff_cursor = 0_usize;
    for i in 0..cluster_count {
        let gi = group_idx[i] as usize;
        let grp = groups.get(gi).ok_or_else(|| {
            invalid(format!("$.manifest.clusters[{i}].group_idx"))
                .at_most(groups.len().saturating_sub(1) as u64)
                .got(gi)
        })?;
        let bound_sphere = [
            f32::from_bits(bs[0][i]),
            f32::from_bits(bs[1][i]),
            f32::from_bits(bs[2][i]),
            f32::from_bits(bs[3][i]),
        ];
        let eq = (bitset[i >> 3] & (1_u8 << (i & 7))) != 0;
        let lod_sphere = if eq {
            bound_sphere
        } else {
            if diff_cursor >= lod_diff_count {
                return Err(invalid("$.manifest.clusters.lod_diff_count").got(diff_cursor));
            }
            let s = [
                f32::from_bits(ld[0][diff_cursor]),
                f32::from_bits(ld[1][diff_cursor]),
                f32::from_bits(ld[2][diff_cursor]),
                f32::from_bits(ld[3][diff_cursor]),
            ];
            diff_cursor += 1;
            s
        };
        clusters.push(ClusterHeader {
            bound_sphere,
            lod_sphere,
            parent_lod_sphere: [
                f32::from_bits(grp.parent_lod_sphere[0]),
                f32::from_bits(grp.parent_lod_sphere[1]),
                f32::from_bits(grp.parent_lod_sphere[2]),
                f32::from_bits(grp.parent_lod_sphere[3]),
            ],
            errors_and_cone: [self_err[i], grp.parent_error, cone[i], meta[i]],
            page_and_fallback: [page_id[i], offset_flags[i], fallback[i], grp.group_id],
        });
    }
    if diff_cursor != lod_diff_count {
        return Err(invalid("$.manifest.clusters.lod_diff_count")
            .expected(Expected::Matching)
            .got(diff_cursor));
    }
    Ok(clusters)
}

fn read_bvh(bytes: &[u8], offset: &mut usize, bvh_count: usize) -> Result<Vec<BvhNode>, Error> {
    let s = read_transposed_u32x4(bytes, offset, bvh_count, "$.manifest.bvh.sphere")?;
    let c = read_transposed_u32x4(bytes, offset, bvh_count, "$.manifest.bvh.child")?;
    let e = read_transposed_u32x4(bytes, offset, bvh_count, "$.manifest.bvh.error")?;
    let m = read_transposed_u32x4(bytes, offset, bvh_count, "$.manifest.bvh.meta")?;
    let mut bvh = Vec::with_capacity(bvh_count);
    for i in 0..bvh_count {
        bvh.push(BvhNode {
            sphere: [
                f32::from_bits(s[0][i]),
                f32::from_bits(s[1][i]),
                f32::from_bits(s[2][i]),
                f32::from_bits(s[3][i]),
            ],
            child_or_cluster: [c[0][i], c[1][i], c[2][i], c[3][i]],
            max_parent_error: [
                f32::from_bits(e[0][i]),
                f32::from_bits(e[1][i]),
                f32::from_bits(e[2][i]),
                f32::from_bits(e[3][i]),
            ],
            meta: [m[0][i], m[1][i], m[2][i], m[3][i]],
        });
    }
    Ok(bvh)
}

fn read_page_spans(
    bytes: &[u8],
    offset: &mut usize,
    span_count: usize,
) -> Result<Vec<PageSpan>, Error> {
    let flag = *bytes
        .get(*offset)
        .ok_or_else(|| invalid("$.manifest.page_spans.format").expected(Expected::Present))?;
    *offset += 1;
    let mut page_spans = Vec::with_capacity(span_count);
    match flag {
        PAGE_SPANS_COMPACT_U16 => {
            let mut cur_offset = 0_u64;
            for _ in 0..span_count {
                let len = u32::from(read_u16(bytes, offset, "$.manifest.page_spans.len")?);
                page_spans.push(PageSpan {
                    byte_offset: cur_offset,
                    byte_len: len,
                });
                cur_offset += u64::from(len);
            }
        }
        PAGE_SPANS_FULL_U64_U32 => {
            for _ in 0..span_count {
                let byte_offset = read_u64(bytes, offset, "$.manifest.page_spans.offset")?;
                let byte_len = read_u32(bytes, offset, "$.manifest.page_spans.len")?;
                page_spans.push(PageSpan {
                    byte_offset,
                    byte_len,
                });
            }
        }
        other => {
            return Err(invalid("$.manifest.page_spans.format")
                .expected(Expected::Supported)
                .got(other));
        }
    }
    Ok(page_spans)
}

impl MeshletManifest {
    pub fn byte_len(&self) -> usize {
        let (groups, _, lod_diff_count) = collect_groups(&self.clusters);
        let compact_spans = page_spans_are_compact(&self.page_spans);
        let span_entry_bytes = if compact_spans {
            2
        } else {
            PAGE_SPAN_FULL_BYTES
        };
        HEADER_BYTES
            + groups.len() * GROUP_STRIDE_BYTES
            + self.clusters.len() * CLUSTER_FIXED_STRIDE_BYTES
            + self.clusters.len().div_ceil(8)
            + lod_diff_count * SPHERE_BYTES
            + self.bvh.len() * BVH_NODE_BYTES
            + 1
            + self.page_spans.len() * span_entry_bytes
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let (groups, group_indices, lod_diff_count) = collect_groups(&self.clusters);
        let compact_spans = page_spans_are_compact(&self.page_spans);
        let span_entry_bytes = if compact_spans {
            2
        } else {
            PAGE_SPAN_FULL_BYTES
        };
        let total_len = HEADER_BYTES
            + groups.len() * GROUP_STRIDE_BYTES
            + self.clusters.len() * CLUSTER_FIXED_STRIDE_BYTES
            + self.clusters.len().div_ceil(8)
            + lod_diff_count * SPHERE_BYTES
            + self.bvh.len() * BVH_NODE_BYTES
            + 1
            + self.page_spans.len() * span_entry_bytes;
        let mut out = Vec::with_capacity(total_len);
        write_header(&mut out, self, groups.len() as u32, lod_diff_count as u32);
        write_groups(&mut out, &groups);
        write_clusters(&mut out, &self.clusters, &group_indices, lod_diff_count);
        write_bvh(&mut out, &self.bvh);
        write_page_spans(&mut out, &self.page_spans, compact_spans);
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let mut offset = 0_usize;
        let hdr = read_manifest_header(bytes, &mut offset)?;
        if hdr.lod_diff_count > hdr.cluster_count {
            return Err(invalid("$.manifest.lod_diff_count")
                .at_most(hdr.cluster_count as u64)
                .got(hdr.lod_diff_count));
        }
        let body_len = encoded_body_len(
            hdr.group_count,
            hdr.cluster_count,
            hdr.lod_diff_count,
            hdr.bvh_count,
        )
        .ok_or_else(|| invalid("$.manifest.size").got(bytes.len()))?;
        let spans_flag_offset = HEADER_BYTES
            .checked_add(body_len)
            .ok_or_else(|| invalid("$.manifest.size").got(bytes.len()))?;
        let &spans_flag = bytes.get(spans_flag_offset).ok_or_else(|| {
            invalid("$.manifest.size")
                .expected(Expected::Matching)
                .got(bytes.len())
        })?;
        let span_entry_bytes = match spans_flag {
            PAGE_SPANS_COMPACT_U16 => 2_usize,
            PAGE_SPANS_FULL_U64_U32 => PAGE_SPAN_FULL_BYTES,
            other => {
                return Err(invalid("$.manifest.page_spans.format")
                    .expected(Expected::Supported)
                    .got(other));
            }
        };
        let span_count = hdr.page_count as usize;
        let expected_len = spans_flag_offset
            .checked_add(1)
            .and_then(|n| {
                span_count
                    .checked_mul(span_entry_bytes)
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or_else(|| invalid("$.manifest.size").got(bytes.len()))?;
        if bytes.len() != expected_len {
            return Err(invalid("$.manifest.size")
                .expected(Expected::Matching)
                .got(bytes.len()));
        }
        let groups = read_groups(bytes, &mut offset, hdr.group_count)?;
        let clusters = read_clusters(
            bytes,
            &mut offset,
            hdr.cluster_count,
            hdr.lod_diff_count,
            &groups,
        )?;
        let bvh = read_bvh(bytes, &mut offset, hdr.bvh_count)?;
        let page_spans = read_page_spans(bytes, &mut offset, span_count)?;
        if offset != bytes.len() {
            return Err(invalid("$.manifest.size")
                .expected(Expected::Matching)
                .got(bytes.len()));
        }
        Ok(Self {
            source_tri_count: hdr.source_tri_count,
            lod_levels: hdr.lod_levels,
            root_page_count: hdr.root_page_count,
            page_count: hdr.page_count,
            bounds_min: hdr.bounds_min,
            bounds_max: hdr.bounds_max,
            clusters,
            bvh,
            page_spans,
        })
    }
}
