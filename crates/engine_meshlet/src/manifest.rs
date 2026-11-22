use std::mem::size_of;

use engine_codec::{Error, ErrorCode, Expected};

use crate::format::{BvhNode, ClusterHeader, MANIFEST_MAGIC, MANIFEST_VERSION, MeshletManifest};

const HEADER_BYTES: usize = 64;

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

fn push_f32x4(out: &mut Vec<u8>, values: [f32; 4]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_u32x4(out: &mut Vec<u8>, values: [u32; 4]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn read_u32(bytes: &[u8], offset: &mut usize, path: &str) -> Result<u32, Error> {
    let end = offset.saturating_add(4);
    let slice = bytes
        .get(*offset..end)
        .ok_or_else(|| invalid(path).expected(Expected::Present))?;
    *offset = end;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_f32(bytes: &[u8], offset: &mut usize, path: &str) -> Result<f32, Error> {
    Ok(f32::from_bits(read_u32(bytes, offset, path)?))
}

fn read_f32x4(bytes: &[u8], offset: &mut usize, path: &str) -> Result<[f32; 4], Error> {
    Ok([
        read_f32(bytes, offset, path)?,
        read_f32(bytes, offset, path)?,
        read_f32(bytes, offset, path)?,
        read_f32(bytes, offset, path)?,
    ])
}

fn read_u32x4(bytes: &[u8], offset: &mut usize, path: &str) -> Result<[u32; 4], Error> {
    Ok([
        read_u32(bytes, offset, path)?,
        read_u32(bytes, offset, path)?,
        read_u32(bytes, offset, path)?,
        read_u32(bytes, offset, path)?,
    ])
}

impl MeshletManifest {
    pub fn byte_len(&self) -> usize {
        HEADER_BYTES
            + self.clusters.len() * size_of::<ClusterHeader>()
            + self.bvh.len() * size_of::<BvhNode>()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.byte_len());
        out.extend_from_slice(&MANIFEST_MAGIC);
        out.extend_from_slice(&MANIFEST_VERSION.to_le_bytes());
        out.extend_from_slice(&self.source_tri_count.to_le_bytes());
        out.extend_from_slice(&self.lod_levels.to_le_bytes());
        out.extend_from_slice(&self.root_page_count.to_le_bytes());
        out.extend_from_slice(&self.page_count.to_le_bytes());
        out.extend_from_slice(&(self.clusters.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.bvh.len() as u32).to_le_bytes());
        for value in self.bounds_min {
            out.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.bounds_max {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out.extend_from_slice(&[0_u8; 8]);
        for cluster in &self.clusters {
            push_f32x4(&mut out, cluster.bound_sphere);
            push_f32x4(&mut out, cluster.lod_sphere);
            push_f32x4(&mut out, cluster.aabb_min);
            push_f32x4(&mut out, cluster.aabb_extent);
            push_f32x4(&mut out, cluster.parent_lod_sphere);
            push_u32x4(&mut out, cluster.cone_and_meta);
            push_u32x4(&mut out, cluster.streaming_meta);
        }
        for node in &self.bvh {
            push_f32x4(&mut out, node.sphere);
            push_u32x4(&mut out, node.child_or_cluster);
            push_f32x4(&mut out, node.max_parent_error);
            push_u32x4(&mut out, node.meta);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
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
        let mut offset = 4_usize;
        let version = read_u32(bytes, &mut offset, "$.manifest.version")?;
        if version != MANIFEST_VERSION {
            return Err(invalid("$.manifest.version")
                .expected(Expected::Supported)
                .got(version));
        }
        let source_tri_count = read_u32(bytes, &mut offset, "$.manifest.source_tri_count")?;
        let lod_levels = read_u32(bytes, &mut offset, "$.manifest.lod_levels")?;
        let root_page_count = read_u32(bytes, &mut offset, "$.manifest.root_page_count")?;
        let page_count = read_u32(bytes, &mut offset, "$.manifest.page_count")?;
        let cluster_count = read_u32(bytes, &mut offset, "$.manifest.cluster_count")? as usize;
        let bvh_count = read_u32(bytes, &mut offset, "$.manifest.bvh_count")? as usize;
        let bounds_min = [
            read_f32(bytes, &mut offset, "$.manifest.bounds_min[0]")?,
            read_f32(bytes, &mut offset, "$.manifest.bounds_min[1]")?,
            read_f32(bytes, &mut offset, "$.manifest.bounds_min[2]")?,
        ];
        let bounds_max = [
            read_f32(bytes, &mut offset, "$.manifest.bounds_max[0]")?,
            read_f32(bytes, &mut offset, "$.manifest.bounds_max[1]")?,
            read_f32(bytes, &mut offset, "$.manifest.bounds_max[2]")?,
        ];
        offset += 8;
        let expected_len = HEADER_BYTES
            .checked_add(
                cluster_count
                    .checked_mul(size_of::<ClusterHeader>())
                    .ok_or_else(|| invalid("$.manifest.cluster_count").got(cluster_count))?,
            )
            .and_then(|n| {
                bvh_count
                    .checked_mul(size_of::<BvhNode>())
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or_else(|| invalid("$.manifest.size").got(bytes.len()))?;
        if bytes.len() != expected_len {
            return Err(invalid("$.manifest.size")
                .expected(Expected::Matching)
                .got(bytes.len()));
        }
        let mut clusters = Vec::with_capacity(cluster_count);
        for _ in 0..cluster_count {
            clusters.push(ClusterHeader {
                bound_sphere: read_f32x4(bytes, &mut offset, "$.manifest.clusters.bound_sphere")?,
                lod_sphere: read_f32x4(bytes, &mut offset, "$.manifest.clusters.lod_sphere")?,
                aabb_min: read_f32x4(bytes, &mut offset, "$.manifest.clusters.aabb_min")?,
                aabb_extent: read_f32x4(bytes, &mut offset, "$.manifest.clusters.aabb_extent")?,
                parent_lod_sphere: read_f32x4(
                    bytes,
                    &mut offset,
                    "$.manifest.clusters.parent_lod_sphere",
                )?,
                cone_and_meta: read_u32x4(bytes, &mut offset, "$.manifest.clusters.cone_and_meta")?,
                streaming_meta: read_u32x4(
                    bytes,
                    &mut offset,
                    "$.manifest.clusters.streaming_meta",
                )?,
            });
        }
        let mut bvh = Vec::with_capacity(bvh_count);
        for _ in 0..bvh_count {
            bvh.push(BvhNode {
                sphere: read_f32x4(bytes, &mut offset, "$.manifest.bvh.sphere")?,
                child_or_cluster: read_u32x4(
                    bytes,
                    &mut offset,
                    "$.manifest.bvh.child_or_cluster",
                )?,
                max_parent_error: read_f32x4(
                    bytes,
                    &mut offset,
                    "$.manifest.bvh.max_parent_error",
                )?,
                meta: read_u32x4(bytes, &mut offset, "$.manifest.bvh.meta")?,
            });
        }
        Ok(Self {
            source_tri_count,
            lod_levels,
            root_page_count,
            page_count,
            bounds_min,
            bounds_max,
            clusters,
            bvh,
        })
    }
}
