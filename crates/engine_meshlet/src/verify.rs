use engine_codec::{Error, ErrorCode, Expected};
use engine_math::collections::{FixedState, HashMap, HashSet};
use engine_math::max;
use glam::Vec3;

use crate::encode::decode_cluster_words;
use crate::format::{
    BVH_BRANCHING, BvhNode, ClusterHeader, MAX_CLUSTER_TRIS, MAX_CLUSTER_VERTS, MeshletManifest,
    NO_BVH_CHILD, PAGE_WORDS, ROOT_PARENT_ERROR, VerifyReport,
};

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

fn sphere_contains_sphere(outer: [f32; 4], inner: [f32; 4]) -> bool {
    let co = Vec3::new(outer[0], outer[1], outer[2]);
    let ci = Vec3::new(inner[0], inner[1], inner[2]);
    let dist = (ci - co).length();
    let tol = max(outer[3], inner[3]) * 1.0e-3 + 1.0e-4;
    dist + inner[3] <= outer[3] + tol
}

fn sphere_is_valid(s: [f32; 4]) -> bool {
    s.iter().all(|v| v.is_finite()) && s[3] >= 0.0
}

fn projected_error(cam: Vec3, sphere: [f32; 4], error: f32) -> f32 {
    let center = Vec3::new(sphere[0], sphere[1], sphere[2]);
    let dist = max((cam - center).length() - sphere[3], 1.0e-4);
    error / dist
}

fn verify_cluster_header(
    path: &str,
    header: &ClusterHeader,
    manifest: &MeshletManifest,
) -> Result<(), Error> {
    if !sphere_is_valid(header.bound_sphere)
        || !sphere_is_valid(header.lod_sphere)
        || !sphere_is_valid(header.parent_lod_sphere)
    {
        return Err(invalid(path).expected(Expected::Finite));
    }
    let self_err = header.self_error();
    let parent_err = header.parent_error();
    if !self_err.is_finite() || !parent_err.is_finite() || self_err < 0.0 || parent_err <= self_err {
        return Err(invalid(format!("{path}.monotonicity"))
            .expected(Expected::Above)
            .got([self_err, parent_err]));
    }
    if header.lod_level() == 0 && self_err != 0.0 {
        return Err(invalid(format!("{path}.leaf_self_error")).got(self_err));
    }
    if !sphere_contains_sphere(header.lod_sphere, header.bound_sphere) {
        return Err(invalid(format!("{path}.lod_sphere_containment")));
    }
    if !sphere_contains_sphere(header.parent_lod_sphere, header.lod_sphere) {
        return Err(invalid(format!("{path}.parent_lod_sphere_containment")));
    }
    let vc = header.vert_count() as usize;
    let tc = header.tri_count() as usize;
    if vc == 0 || vc > MAX_CLUSTER_VERTS || tc == 0 || tc > MAX_CLUSTER_TRIS {
        return Err(invalid(format!("{path}.counts")).got([vc, tc]));
    }
    let pid = header.page_id();
    if pid >= manifest.page_count {
        return Err(invalid(format!("{path}.page_id"))
            .at_most(u64::from(manifest.page_count.saturating_sub(1)))
            .got(pid));
    }
    let fb_start = header.fallback_start() as usize;
    let fb_count = header.fallback_count() as usize;
    if header.is_root() {
        if parent_err != ROOT_PARENT_ERROR || pid >= manifest.root_page_count || fb_count != 0 {
            return Err(invalid(format!("{path}.root_invariant")).got([parent_err, pid as f32]));
        }
    } else if pid < manifest.root_page_count {
        if !header.is_in_root_page() || fb_count != 0 {
            return Err(invalid(format!("{path}.root_page_fallback")).got(fb_count));
        }
    } else {
        if fb_count == 0 || fb_start + fb_count > manifest.clusters.len() {
            return Err(invalid(format!("{path}.fallback_range")).got([fb_start, fb_count]));
        }
        for p in fb_start..fb_start + fb_count {
            if manifest.clusters[p].page_id() >= pid {
                return Err(invalid(format!("{path}.fallback_page_order"))
                    .got([manifest.clusters[p].page_id(), pid]));
            }
        }
    }
    Ok(())
}

fn verify_cluster(
    index: usize,
    header: &ClusterHeader,
    manifest: &MeshletManifest,
    page_words: &[u32],
    cullable_cones: &mut u32,
    max_quant_err: &mut f32,
) -> Result<(), Error> {
    let path = format!("$.clusters[{index}]");
    verify_cluster_header(&path, header, manifest)?;
    let page_start = (header.page_id() as usize) * PAGE_WORDS;
    let page = &page_words[page_start..page_start + PAGE_WORDS];
    let decoded = decode_cluster_words(header, page)?;
    let b_center = Vec3::new(
        header.bound_sphere[0],
        header.bound_sphere[1],
        header.bound_sphere[2],
    );
    let b_radius = header.bound_sphere[3];
    let tol = b_radius * 1.0e-3 + 1.0e-4;
    *max_quant_err = max(*max_quant_err, decoded.step * 0.5);
    for (vi, v) in decoded.vertices.iter().enumerate() {
        if !v.position.is_finite() || !v.normal.is_finite() {
            return Err(invalid(format!("{path}.vertices[{vi}]")).expected(Expected::Finite));
        }
        if (v.position - b_center).length() > b_radius + tol {
            return Err(invalid(format!("{path}.vertices[{vi}].bound_sphere")));
        }
        if (v.normal.length() - 1.0).abs() > 2.0e-2 {
            return Err(invalid(format!("{path}.vertices[{vi}].normal")).got(v.normal.length()));
        }
    }
    let ([ax, ay, az], cutoff) = header.normal_cone();
    if cutoff < 127 {
        *cullable_cones += 1;
        let q_axis = Vec3::new(f32::from(ax), f32::from(ay), f32::from(az));
        let q_len = q_axis.length();
        if q_len <= 1.0e-4 {
            return Err(invalid(format!("{path}.cone_axis")));
        }
        let unit_axis = q_axis / q_len;
        let sin_limit = f32::from(cutoff) / 127.0 + 2.0e-2;
        for (ti, &[i0, i1, i2]) in decoded.triangles.iter().enumerate() {
            if i0 == i1 || i1 == i2 || i2 == i0 {
                return Err(invalid(format!("{path}.triangles[{ti}].degenerate")));
            }
            let p0 = decoded.vertices[usize::from(i0)].position;
            let p1 = decoded.vertices[usize::from(i1)].position;
            let p2 = decoded.vertices[usize::from(i2)].position;
            let cross = (p1 - p0).cross(p2 - p0);
            let len_sq = cross.length_squared();
            if len_sq > 1.0e-18 {
                let n = cross / len_sq.sqrt();
                let d = n.dot(unit_axis);
                let sin_v = max(0.0, 1.0 - d * d).sqrt();
                if d < 0.0 || sin_v > sin_limit {
                    return Err(
                        invalid(format!("{path}.cone_violation")).got([d, sin_v, sin_limit])
                    );
                }
            }
        }
    } else {
        for (ti, &[i0, i1, i2]) in decoded.triangles.iter().enumerate() {
            if i0 == i1 || i1 == i2 || i2 == i0 {
                return Err(invalid(format!("{path}.triangles[{ti}].degenerate")));
            }
        }
    }
    Ok(())
}

fn verify_bvh_subtree(
    ni: usize,
    manifest: &MeshletManifest,
    visited_nodes: &mut [bool],
    cluster_hits: &mut [u32],
) -> Result<f32, Error> {
    let n_nodes = manifest.bvh.len();
    let n_clusters = manifest.clusters.len();
    if ni >= n_nodes {
        return Err(invalid("$.bvh.index").got(ni));
    }
    if visited_nodes[ni] {
        return Err(invalid(format!("$.bvh[{ni}]")).expected(Expected::Acyclic));
    }
    visited_nodes[ni] = true;
    let node: &BvhNode = &manifest.bvh[ni];
    if !sphere_is_valid(node.sphere) {
        return Err(invalid(format!("$.bvh[{ni}].sphere")).expected(Expected::Finite));
    }
    if node.is_leaf() {
        let start = node.leaf_cluster_start() as usize;
        let count = node.leaf_cluster_count() as usize;
        if count == 0 || start + count > n_clusters {
            return Err(invalid(format!("$.bvh[{ni}].leaf_range")).got([start, count]));
        }
        let leaf_max_pe = node.max_parent_error[0];
        let mut actual_max = 0.0_f32;
        for (ci, header) in manifest.clusters[start..start + count].iter().enumerate() {
            cluster_hits[start + ci] += 1;
            if !sphere_contains_sphere(node.sphere, header.parent_lod_sphere) {
                return Err(invalid(format!("$.bvh[{ni}].leaf_sphere")));
            }
            actual_max = max(actual_max, header.parent_error());
            if header.parent_error() > leaf_max_pe + 1.0e-6 {
                return Err(invalid(format!("$.bvh[{ni}].leaf_max_error")));
            }
        }
        return Ok(actual_max);
    }
    let child_count = node.child_count() as usize;
    if child_count == 0 || child_count > BVH_BRANCHING {
        return Err(invalid(format!("$.bvh[{ni}].child_count")).got(child_count));
    }
    let mut subtree_max = 0.0_f32;
    for slot in 0..child_count {
        let child_idx = node.child_or_cluster[slot];
        if child_idx == NO_BVH_CHILD || (child_idx as usize) >= n_nodes {
            return Err(invalid(format!("$.bvh[{ni}].child[{slot}]")).got(child_idx));
        }
        let child = &manifest.bvh[child_idx as usize];
        if !sphere_contains_sphere(node.sphere, child.sphere) {
            return Err(invalid(format!("$.bvh[{ni}].child_sphere[{slot}]")));
        }
        let child_max = verify_bvh_subtree(child_idx as usize, manifest, visited_nodes, cluster_hits)?;
        if child_max > node.max_parent_error[slot] + 1.0e-6 {
            return Err(invalid(format!("$.bvh[{ni}].max_parent_error[{slot}]"))
                .got([child_max, node.max_parent_error[slot]]));
        }
        subtree_max = max(subtree_max, child_max);
    }
    Ok(subtree_max)
}

fn verify_bvh(manifest: &MeshletManifest) -> Result<(), Error> {
    if manifest.bvh.is_empty() {
        return Err(invalid("$.bvh").expected(Expected::NonEmpty));
    }
    let mut visited_nodes = vec![false; manifest.bvh.len()];
    let mut cluster_hits = vec![0_u32; manifest.clusters.len()];
    verify_bvh_subtree(0, manifest, &mut visited_nodes, &mut cluster_hits)?;
    for (ci, &hits) in cluster_hits.iter().enumerate() {
        if hits != 1 {
            return Err(invalid(format!("$.bvh.coverage[{ci}]")).got(hits));
        }
    }
    Ok(())
}

fn verify_partial_residency_cuts(manifest: &MeshletManifest) -> Result<u32, Error> {
    let total_pages = manifest.page_count;
    let root_pages = manifest.root_page_count;
    let budgets = [
        root_pages,
        root_pages + (total_pages - root_pages) / 4,
        root_pages + (total_pages - root_pages) / 2,
        total_pages,
    ];
    let mut checks = 0_u32;
    for resident_pages in budgets {
        let mut emitted = 0_u32;
        let mut seen: HashSet<usize> = HashSet::with_hasher(FixedState::default());
        let mut stack: Vec<usize> = Vec::new();
        for (ci, header) in manifest.clusters.iter().enumerate() {
            if header.lod_level() != 0 {
                continue;
            }
            if header.page_id() < resident_pages {
                emitted += 1;
                continue;
            }
            let fb_s = header.fallback_start() as usize;
            let fb_c = header.fallback_count() as usize;
            if fb_c == 0 {
                return Err(invalid(format!("$.residency.no_fallback[{ci}]")));
            }
            stack.clear();
            for p in fb_s..fb_s + fb_c {
                stack.push(p);
            }
            let mut steps = 0_usize;
            while let Some(cur) = stack.pop() {
                steps += 1;
                if steps > manifest.clusters.len().max(256) {
                    return Err(invalid("$.residency.cycle").got(cur));
                }
                if !seen.insert(cur) {
                    continue;
                }
                let ch = &manifest.clusters[cur];
                if ch.page_id() < resident_pages {
                    emitted += 1;
                } else {
                    let ps = ch.fallback_start() as usize;
                    let pc = ch.fallback_count() as usize;
                    if pc == 0 || ps + pc > manifest.clusters.len() {
                        return Err(invalid("$.residency.dead_end").got(cur));
                    }
                    for p in ps..ps + pc {
                        stack.push(p);
                    }
                }
            }
        }
        if emitted == 0 {
            return Err(invalid("$.residency.empty").got(resident_pages));
        }
        checks += 1;
    }
    let mut per_leaf_seen: HashSet<usize> = HashSet::with_hasher(FixedState::default());
    let mut per_leaf_stack: Vec<usize> = Vec::new();
    for (ci, header) in manifest.clusters.iter().enumerate() {
        if header.lod_level() != 0 || header.page_id() < root_pages {
            continue;
        }
        per_leaf_seen.clear();
        per_leaf_stack.clear();
        let fb_s = header.fallback_start() as usize;
        let fb_c = header.fallback_count() as usize;
        for p in fb_s..fb_s + fb_c {
            per_leaf_stack.push(p);
        }
        let mut steps = 0_usize;
        while let Some(cur) = per_leaf_stack.pop() {
            steps += 1;
            if steps > 1024 {
                return Err(invalid(format!("$.residency.walk_limit[{ci}]")).got(steps));
            }
            if !per_leaf_seen.insert(cur) {
                continue;
            }
            let ch = &manifest.clusters[cur];
            if ch.page_id() >= root_pages {
                let ps = ch.fallback_start() as usize;
                let pc = ch.fallback_count() as usize;
                for p in ps..ps + pc {
                    per_leaf_stack.push(p);
                }
                if per_leaf_stack.len() > 128 {
                    return Err(
                        invalid(format!("$.residency.stack_limit[{ci}]")).got(per_leaf_stack.len())
                    );
                }
            }
        }
    }
    Ok(checks + 1)
}

fn verify_group_page_atomicity(manifest: &MeshletManifest) -> Result<(), Error> {
    let mut groups: HashMap<u32, (usize, u32)> =
        HashMap::with_capacity_and_hasher(manifest.clusters.len(), FixedState::default());
    for header in &manifest.clusters {
        let gid = header.group_id();
        let entry = groups.entry(gid).or_insert((0, header.page_id()));
        entry.0 += header.word_count() as usize;
    }
    for (ci, header) in manifest.clusters.iter().enumerate() {
        let gid = header.group_id();
        if let Some(&(total_words, first_page)) = groups.get(&gid)
            && total_words <= PAGE_WORDS
            && header.page_id() != first_page
        {
            return Err(invalid(format!("$.clusters[{ci}].group_page_split"))
                .got([first_page, header.page_id()]));
        }
    }
    Ok(())
}

fn verify_group_consistency_and_cuts(
    manifest: &MeshletManifest,
    leaf_count: u32,
    root_count: u32,
) -> Result<u32, Error> {
    let mut group_ref: HashMap<u32, (f32, [f32; 4])> = HashMap::with_hasher(FixedState::default());
    for (ci, header) in manifest.clusters.iter().enumerate() {
        let gid = header.group_id();
        if let Some(&(pe, ps)) = group_ref.get(&gid) {
            if header.parent_error().to_bits() != pe.to_bits()
                || header.parent_lod_sphere[0].to_bits() != ps[0].to_bits()
                || header.parent_lod_sphere[1].to_bits() != ps[1].to_bits()
                || header.parent_lod_sphere[2].to_bits() != ps[2].to_bits()
                || header.parent_lod_sphere[3].to_bits() != ps[3].to_bits()
            {
                return Err(invalid(format!("$.clusters[{ci}].group_consistency")).got(gid));
            }
        } else {
            group_ref.insert(gid, (header.parent_error(), header.parent_lod_sphere));
        }
    }
    verify_group_page_atomicity(manifest)?;
    let min_b = Vec3::from_array(manifest.bounds_min);
    let max_b = Vec3::from_array(manifest.bounds_max);
    let center = (min_b + max_b) * 0.5;
    let diag = max((max_b - min_b).length(), 1.0);
    let cameras = [
        center + Vec3::new(0.0, diag * 0.2, diag * 0.8),
        center + Vec3::new(diag * 2.5, diag * 1.2, diag * 2.5),
        center + Vec3::new(-diag * 6.0, diag * 3.0, -diag * 4.0),
        center,
    ];
    let thresholds = [0.0_f32, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e20];
    let mut checks = 0_u32;
    for &cam in &cameras {
        for &thresh in &thresholds {
            let mut selected = 0_u32;
            for header in &manifest.clusters {
                let p_err = projected_error(cam, header.parent_lod_sphere, header.parent_error());
                let s_err = projected_error(cam, header.lod_sphere, header.self_error());
                if p_err > thresh && s_err <= thresh {
                    selected += 1;
                }
            }
            if selected == 0 {
                return Err(invalid("$.dag.cut_empty").got([thresh, 0.0]));
            }
            if thresh == 0.0 && selected != leaf_count {
                return Err(invalid("$.dag.finest_cut").got([selected, leaf_count]));
            }
            if thresh >= 1.0e19 && selected != root_count {
                return Err(invalid("$.dag.coarsest_cut").got([selected, root_count]));
            }
            checks += 1;
        }
    }
    checks += verify_partial_residency_cuts(manifest)?;
    Ok(checks)
}

pub fn verify(manifest: &MeshletManifest, page_words: &[u32]) -> Result<VerifyReport, Error> {
    if manifest.clusters.is_empty() {
        return Err(invalid("$.clusters").expected(Expected::NonEmpty));
    }
    if manifest.page_count == 0
        || manifest.root_page_count == 0
        || manifest.root_page_count > manifest.page_count
        || manifest.page_spans.len() != manifest.page_count as usize
    {
        return Err(invalid("$.pages").got([manifest.root_page_count, manifest.page_count]));
    }
    let expected_words = (manifest.page_count as usize) * PAGE_WORDS;
    if page_words.len() != expected_words {
        return Err(invalid("$.page_words")
            .expected(Expected::Matching)
            .got(page_words.len()));
    }
    for axis in 0..3 {
        if !manifest.bounds_min[axis].is_finite()
            || !manifest.bounds_max[axis].is_finite()
            || manifest.bounds_min[axis] > manifest.bounds_max[axis]
        {
            return Err(invalid("$.bounds").expected(Expected::Finite));
        }
    }
    let mut leaf_clusters = 0_u32;
    let mut root_clusters = 0_u32;
    let mut total_tris = 0_u32;
    let mut total_verts = 0_u32;
    let mut geom_words = 0_u64;
    let mut cullable_cones = 0_u32;
    let mut max_quant_err = 0.0_f32;
    for (ci, header) in manifest.clusters.iter().enumerate() {
        verify_cluster(
            ci,
            header,
            manifest,
            page_words,
            &mut cullable_cones,
            &mut max_quant_err,
        )?;
        if header.lod_level() == 0 {
            leaf_clusters += 1;
        }
        if header.is_root() {
            root_clusters += 1;
        }
        total_tris += header.tri_count();
        total_verts += header.vert_count();
        geom_words += u64::from(header.word_count());
    }
    if leaf_clusters == 0 || root_clusters == 0 {
        return Err(invalid("$.dag.endpoints").got([leaf_clusters, root_clusters]));
    }
    verify_bvh(manifest)?;
    let cut_checks_passed =
        verify_group_consistency_and_cuts(manifest, leaf_clusters, root_clusters)?;
    let bytes_per_cluster_tri = ((geom_words * 4) as f64) / f64::from(total_tris.max(1));
    Ok(VerifyReport {
        cluster_count: manifest.clusters.len() as u32,
        leaf_cluster_count: leaf_clusters,
        root_cluster_count: root_clusters,
        lod_levels: manifest.lod_levels,
        bvh_node_count: manifest.bvh.len() as u32,
        page_count: manifest.page_count,
        root_page_count: manifest.root_page_count,
        total_cluster_tris: total_tris,
        total_cluster_verts: total_verts,
        cullable_cone_clusters: cullable_cones,
        cut_checks_passed,
        bytes_per_cluster_tri,
        max_quant_error: max_quant_err,
    })
}
