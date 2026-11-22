use std::collections::BTreeMap;

use engine_codec::{Error, ErrorCode, Expected};
use engine_math::collections::{FixedState, HashMap, HashSet};
use engine_math::{clamp, max};
use glam::{Vec2, Vec3};

use crate::bounds::{aabb_of_points, bounding_sphere_points, compute_normal_cone, merge_spheres};
use crate::bvh::build_bvh_and_order;
use crate::encode::{decode_cluster_words, encode_cluster_words, encode_compressed_pages};
use crate::format::{
    BuildStats, BuiltMeshlet, ClusterHeader, ClusterVertex, FLAG_IN_ROOT_PAGE, FLAG_PINNED_COARSE,
    FLAG_ROOT_CLUSTER, MAX_CLUSTER_TRIS, MeshletManifest, NO_FALLBACK_CLUSTER, PAGE_WORDS,
    ROOT_PARENT_ERROR, SourceMesh,
};
use crate::group::{GroupInput, group_clusters};
use crate::partition::{MeshletData, partition_meshlets, weld_positions};
use crate::simplify::{cluster_weld_group, simplify_group};

fn invalid(path: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidMesh, path)
}

#[derive(Clone, Copy)]
struct QuantBounds {
    min: Vec3,
    extent: Vec3,
}

struct DagNode {
    meshlet: MeshletData,
    header: ClusterHeader,
    payload: Vec<u32>,
}

struct PreparedSource {
    vertices: Vec<ClusterVertex>,
    pos_ids: Vec<u32>,
    by_material: BTreeMap<u8, Vec<[u32; 3]>>,
    material_seams: HashSet<u32>,
    source_tri_count: u32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
}

fn compute_smooth_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut accum = vec![Vec3::ZERO; positions.len()];
    for &[a, b, c] in indices.as_chunks::<3>().0 {
        let (a, b, c) = (a as usize, b as usize, c as usize);
        let cross = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        if cross.is_finite() {
            accum[a] += cross;
            accum[b] += cross;
            accum[c] += cross;
        }
    }
    accum
        .into_iter()
        .map(|n| {
            if n.length_squared() > 1.0e-18 {
                n.normalize()
            } else {
                Vec3::Y
            }
        })
        .collect()
}

fn validate_lengths(source: &SourceMesh) -> Result<(usize, usize), Error> {
    let v_count = source.positions.len();
    if v_count == 0 {
        return Err(invalid("$.positions").expected(Expected::NonEmpty));
    }
    for (vi, &p) in source.positions.iter().enumerate() {
        if !p.is_finite() {
            return Err(invalid(format!("$.positions[{vi}]")).expected(Expected::Finite));
        }
    }
    if source.indices.is_empty() || !source.indices.len().is_multiple_of(3) {
        return Err(invalid("$.indices")
            .expected(Expected::Multiple)
            .got(source.indices.len()));
    }
    let tri_count = source.indices.len() / 3;
    if !source.normals.is_empty() && source.normals.len() != v_count {
        return Err(invalid("$.normals")
            .expected(Expected::Matching)
            .got(source.normals.len()));
    }
    if !source.uvs.is_empty() && source.uvs.len() != v_count {
        return Err(invalid("$.uvs")
            .expected(Expected::Matching)
            .got(source.uvs.len()));
    }
    if !source.colors.is_empty() && source.colors.len() != v_count {
        return Err(invalid("$.colors")
            .expected(Expected::Matching)
            .got(source.colors.len()));
    }
    if !source.ao.is_empty() && source.ao.len() != v_count {
        return Err(invalid("$.ao")
            .expected(Expected::Matching)
            .got(source.ao.len()));
    }
    if !source.material_ids.is_empty() && source.material_ids.len() != tri_count {
        return Err(invalid("$.material_ids")
            .expected(Expected::Matching)
            .got(source.material_ids.len()));
    }
    for (ii, &idx) in source.indices.iter().enumerate() {
        if (idx as usize) >= v_count {
            return Err(invalid(format!("$.indices[{ii}]"))
                .at_most(v_count.saturating_sub(1) as u64)
                .got(idx));
        }
    }
    Ok((v_count, tri_count))
}

fn validate_and_prepare(source: &SourceMesh) -> Result<PreparedSource, Error> {
    let (v_count, _) = validate_lengths(source)?;
    let normals = if source.normals.is_empty() {
        compute_smooth_normals(&source.positions, &source.indices)
    } else {
        source
            .normals
            .iter()
            .map(|&n| {
                if n.is_finite() && n.length_squared() > 1.0e-18 {
                    n.normalize()
                } else {
                    Vec3::Y
                }
            })
            .collect()
    };
    let (pos_ids, _) = weld_positions(&source.positions);
    let mut vertices = Vec::with_capacity(v_count);
    for (i, &normal) in normals.iter().enumerate() {
        let uv = source.uvs.get(i).copied().unwrap_or(Vec2::ZERO);
        let raw_color = source.colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0]);
        let ao = source.ao.get(i).copied().unwrap_or(1.0);
        let color = [
            if raw_color[0].is_finite() {
                clamp(raw_color[0], 0.0, 1.0)
            } else {
                1.0
            },
            if raw_color[1].is_finite() {
                clamp(raw_color[1], 0.0, 1.0)
            } else {
                1.0
            },
            if raw_color[2].is_finite() {
                clamp(raw_color[2], 0.0, 1.0)
            } else {
                1.0
            },
        ];
        vertices.push(ClusterVertex {
            position: source.positions[i],
            normal,
            uv: if uv.is_finite() { uv } else { Vec2::ZERO },
            color,
            ao: if ao.is_finite() {
                clamp(ao, 0.0, 1.0)
            } else {
                1.0
            },
        });
    }
    let mut by_material: BTreeMap<u8, Vec<[u32; 3]>> = BTreeMap::new();
    let mut pos_mat: HashMap<u32, u8> =
        HashMap::with_capacity_and_hasher(v_count, FixedState::default());
    let mut material_seams: HashSet<u32> = HashSet::with_hasher(FixedState::default());
    let mut valid_tris = 0_u32;
    for (ti, &[a, b, c]) in source.indices.as_chunks::<3>().0.iter().enumerate() {
        let (pa, pb, pc) = (
            pos_ids[a as usize],
            pos_ids[b as usize],
            pos_ids[c as usize],
        );
        if pa == pb || pb == pc || pc == pa {
            continue;
        }
        let cross = (vertices[b as usize].position - vertices[a as usize].position)
            .cross(vertices[c as usize].position - vertices[a as usize].position);
        if !cross.is_finite() || cross.length_squared() <= 1.0e-18 {
            continue;
        }
        let mat = source.material_ids.get(ti).copied().unwrap_or(0);
        by_material.entry(mat).or_default().push([a, b, c]);
        valid_tris += 1;
        for pid in [pa, pb, pc] {
            if let Some(&prev) = pos_mat.get(&pid) {
                if prev != mat {
                    material_seams.insert(pid);
                }
            } else {
                pos_mat.insert(pid, mat);
            }
        }
    }
    if valid_tris == 0 {
        return Err(invalid("$.indices").expected(Expected::Triangles));
    }
    let (b_min, b_ext) = aabb_of_points(&source.positions);
    let b_max = b_min + b_ext;
    Ok(PreparedSource {
        vertices,
        pos_ids,
        by_material,
        material_seams,
        source_tri_count: valid_tris,
        bounds_min: b_min.to_array(),
        bounds_max: b_max.to_array(),
    })
}

fn make_dag_node(
    meshlet: MeshletData,
    lod_level: u32,
    material_id: u8,
    quant: QuantBounds,
) -> Result<DagNode, Error> {
    let raw_positions: Vec<Vec3> = meshlet.vertices.iter().map(|v| v.position).collect();
    let aabb_min = quant.min;
    let aabb_extent = quant.extent;
    let mut payload = Vec::new();
    encode_cluster_words(
        &meshlet.vertices,
        &meshlet.triangles,
        aabb_min,
        aabb_extent,
        &mut payload,
    )?;
    let meta = ClusterHeader::pack_meta(
        meshlet.vertices.len(),
        meshlet.triangles.len(),
        lod_level,
        material_id,
    );
    let temp_header = ClusterHeader {
        bound_sphere: [0.0; 4],
        lod_sphere: [0.0; 4],
        aabb_min: [aabb_min.x, aabb_min.y, aabb_min.z, 0.0],
        aabb_extent: [
            aabb_extent.x,
            aabb_extent.y,
            aabb_extent.z,
            ROOT_PARENT_ERROR,
        ],
        parent_lod_sphere: [0.0; 4],
        cone_and_meta: [0, meta, 0, 0],
        streaming_meta: [0, NO_FALLBACK_CLUSTER, FLAG_ROOT_CLUSTER, NO_FALLBACK_CLUSTER],
    };
    let decoded = decode_cluster_words(&temp_header, &payload)?;
    let decoded_positions: Vec<Vec3> = decoded.vertices.iter().map(|v| v.position).collect();
    let mut combined_points = Vec::with_capacity(raw_positions.len() + decoded_positions.len());
    combined_points.extend_from_slice(&raw_positions);
    combined_points.extend_from_slice(&decoded_positions);
    let bound_sphere = bounding_sphere_points(&combined_points);
    let cone = compute_normal_cone(&decoded_positions, &meshlet.triangles);
    let header = ClusterHeader {
        bound_sphere,
        lod_sphere: bound_sphere,
        aabb_min: [aabb_min.x, aabb_min.y, aabb_min.z, 0.0],
        aabb_extent: [
            aabb_extent.x,
            aabb_extent.y,
            aabb_extent.z,
            ROOT_PARENT_ERROR,
        ],
        parent_lod_sphere: bound_sphere,
        cone_and_meta: [cone, meta, 0, 0],
        streaming_meta: [0, NO_FALLBACK_CLUSTER, FLAG_ROOT_CLUSTER, NO_FALLBACK_CLUSTER],
    };
    Ok(DagNode {
        meshlet,
        header,
        payload,
    })
}

fn two_nearest_parents(
    child_center: Vec3,
    parent_ids: &[usize],
    nodes: &[DagNode],
) -> (u32, u32) {
    if parent_ids.len() <= 2 {
        return (
            parent_ids[0] as u32,
            parent_ids
                .get(1)
                .map_or(NO_FALLBACK_CLUSTER, |&p| p as u32),
        );
    }
    let mut ranked: Vec<(f32, usize)> = parent_ids
        .iter()
        .map(|&pid| {
            let s = nodes[pid].header.bound_sphere;
            let pc = Vec3::new(s[0], s[1], s[2]);
            ((pc - child_center).length_squared(), pid)
        })
        .collect();
    ranked.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    (ranked[0].1 as u32, ranked[1].1 as u32)
}

struct ReduceEnv<'a> {
    material_id: u8,
    material_seams: &'a HashSet<u32>,
    quant: QuantBounds,
    allow_weld: bool,
    relax_mat_seams: bool,
    max_parents: usize,
    mark_pinned: bool,
}

fn simplify_for_group(
    env: &ReduceEnv<'_>,
    child_meshlets: &[&MeshletData],
    hard_locked: &HashSet<u32>,
    child_count: usize,
) -> (Vec<MeshletData>, f32) {
    let in_tris: usize = child_meshlets.iter().map(|m| m.triangles.len()).sum();
    let base_parents = (child_count / 2).max(1);
    let target_parents = if env.max_parents > 0 {
        base_parents.min(env.max_parents)
    } else {
        base_parents
    };
    let target_tris = (in_tris / 2)
        .min(target_parents * (MAX_CLUSTER_TRIS - 6))
        .max(1);
    let target_verts = (target_parents * 50).max(3);
    let mut simplified =
        simplify_group(child_meshlets, hard_locked, target_tris, target_verts, true);
    let mut parts = partition_meshlets(
        &simplified.vertices,
        &simplified.pos_ids,
        &simplified.triangles,
    );
    let mut err = simplified.error;
    let over_cap = |len: usize| {
        len == 0 || len >= child_count || (env.max_parents > 0 && len > env.max_parents)
    };
    if over_cap(parts.len()) && simplified.triangles.len() > 2 {
        let retry = simplify_group(
            child_meshlets,
            hard_locked,
            (target_tris * 3 / 4).max(1),
            (target_verts * 4 / 5).max(3),
            true,
        );
        let retry_parts = partition_meshlets(&retry.vertices, &retry.pos_ids, &retry.triangles);
        if !retry_parts.is_empty() && retry_parts.len() < parts.len() {
            simplified = retry;
            err = simplified.error;
            parts = retry_parts;
        }
    }
    if over_cap(parts.len()) && env.allow_weld {
        let welded = cluster_weld_group(child_meshlets, hard_locked, target_tris, target_verts);
        let weld_parts = partition_meshlets(&welded.vertices, &welded.pos_ids, &welded.triangles);
        if !over_cap(weld_parts.len()) {
            parts = weld_parts;
            err = max(err, welded.error);
        }
    }
    (parts, err)
}

fn reduce_single_group(
    env: &ReduceEnv<'_>,
    child_ids: &[usize],
    level_pos_count: &HashMap<u32, u32>,
    nodes: &mut Vec<DagNode>,
    next_group_id: &mut u32,
) -> Result<Option<Vec<usize>>, Error> {
    if child_ids.len() < 2 {
        return Ok(None);
    }
    let mut group_pos_count: HashMap<u32, u32> = HashMap::with_hasher(FixedState::default());
    for &ci in child_ids {
        let mut uniq = nodes[ci].meshlet.pos_ids.clone();
        uniq.sort_unstable();
        uniq.dedup();
        for p in uniq {
            *group_pos_count.entry(p).or_insert(0) += 1;
        }
    }
    let mut hard_locked: HashSet<u32> = HashSet::with_hasher(FixedState::default());
    for (&p, &g_cnt) in &group_pos_count {
        let total_cnt = level_pos_count.get(&p).copied().unwrap_or(0);
        if total_cnt > g_cnt || (!env.relax_mat_seams && env.material_seams.contains(&p)) {
            hard_locked.insert(p);
        }
    }
    let child_meshlets: Vec<&MeshletData> =
        child_ids.iter().map(|&ci| &nodes[ci].meshlet).collect();
    let (parent_meshlets, simplify_error) =
        simplify_for_group(env, &child_meshlets, &hard_locked, child_ids.len());
    if parent_meshlets.is_empty()
        || parent_meshlets.len() >= child_ids.len()
        || (env.max_parents > 0 && parent_meshlets.len() > env.max_parents)
    {
        return Ok(None);
    }
    let mut parent_lod = 0_u32;
    let mut max_child_err = 0.0_f32;
    for &ci in child_ids {
        parent_lod = parent_lod.max(nodes[ci].header.lod_level() + 1);
        max_child_err = max(max_child_err, nodes[ci].header.self_error());
    }
    let parent_lod = parent_lod.min(255);
    let group_error = max(
        max_child_err.hypot(simplify_error),
        max_child_err * 1.001 + 1.0e-5,
    );
    let mut parent_ids = Vec::with_capacity(parent_meshlets.len());
    for pm in parent_meshlets {
        let id = nodes.len();
        nodes.push(make_dag_node(pm, parent_lod, env.material_id, env.quant)?);
        parent_ids.push(id);
    }
    let mut sphere_list = Vec::with_capacity(child_ids.len() + parent_ids.len());
    for &ci in child_ids {
        sphere_list.push(nodes[ci].header.lod_sphere);
    }
    for &pi in &parent_ids {
        sphere_list.push(nodes[pi].header.bound_sphere);
    }
    let group_lod_sphere = merge_spheres(&sphere_list);
    let gid = *next_group_id;
    *next_group_id += 1;
    let p_flag = FLAG_ROOT_CLUSTER | if env.mark_pinned { FLAG_PINNED_COARSE } else { 0 };
    for &pi in &parent_ids {
        let h = &mut nodes[pi].header;
        h.lod_sphere = group_lod_sphere;
        h.parent_lod_sphere = group_lod_sphere;
        h.aabb_min[3] = group_error;
        h.aabb_extent[3] = ROOT_PARENT_ERROR;
        h.streaming_meta[2] = p_flag;
        h.streaming_meta[3] = NO_FALLBACK_CLUSTER;
    }
    let mut parent_hits = vec![0_u32; parent_ids.len()];
    let mut assigned: Vec<(u32, u32)> = Vec::with_capacity(child_ids.len());
    for &ci in child_ids {
        let cs = nodes[ci].header.bound_sphere;
        let (fb0, fb1) = two_nearest_parents(Vec3::new(cs[0], cs[1], cs[2]), &parent_ids, nodes);
        if let Some(pos) = parent_ids.iter().position(|&p| p as u32 == fb0) {
            parent_hits[pos] += 1;
        }
        if fb1 != NO_FALLBACK_CLUSTER
            && let Some(pos) = parent_ids.iter().position(|&p| p as u32 == fb1)
        {
            parent_hits[pos] += 1;
        }
        assigned.push((fb0, fb1));
    }
    for (p_idx, &hits) in parent_hits.iter().enumerate() {
        if hits == 0 && !assigned.is_empty() {
            let pid = parent_ids[p_idx] as u32;
            let slot = p_idx % assigned.len();
            assigned[slot].1 = pid;
        }
    }
    for (&ci, (fb0, fb1)) in child_ids.iter().zip(assigned) {
        let h = &mut nodes[ci].header;
        h.aabb_extent[3] = group_error;
        h.parent_lod_sphere = group_lod_sphere;
        h.streaming_meta[0] = gid;
        h.streaming_meta[1] = fb0;
        h.streaming_meta[2] = if env.mark_pinned { FLAG_PINNED_COARSE } else { 0 };
        h.streaming_meta[3] = fb1;
    }
    Ok(Some(parent_ids))
}

fn reduce_material_dag(
    material_id: u8,
    mut active: Vec<usize>,
    material_seams: &HashSet<u32>,
    quant: QuantBounds,
    nodes: &mut Vec<DagNode>,
    next_group_id: &mut u32,
) -> Result<(), Error> {
    let mut pass = 0_u32;
    let mut stall_streak = 0_u32;
    let min_root_clusters = if active.len() > 64 { 24 } else { 1 };
    while active.len() > min_root_clusters && pass < 160 {
        let unified = active.len() <= 48;
        let (group_max, max_parents) = if unified {
            (active.len(), 0)
        } else if stall_streak >= 3 {
            (8, 0)
        } else if stall_streak == 2 {
            (8, 4)
        } else if stall_streak == 1 {
            (6, 3)
        } else {
            (4, 2)
        };
        let allow_weld = !unified && stall_streak >= 4;
        let mut level_pos_count: HashMap<u32, u32> =
            HashMap::with_capacity_and_hasher(active.len() * 32, FixedState::default());
        for &ci in &active {
            let mut uniq = nodes[ci].meshlet.pos_ids.clone();
            uniq.sort_unstable();
            uniq.dedup();
            for p in uniq {
                *level_pos_count.entry(p).or_insert(0) += 1;
            }
        }
        let inputs: Vec<GroupInput<'_>> = active
            .iter()
            .map(|&ci| {
                let s = nodes[ci].header.bound_sphere;
                GroupInput {
                    pos_ids: &nodes[ci].meshlet.pos_ids,
                    center: Vec3::new(s[0], s[1], s[2]),
                    radius: s[3],
                }
            })
            .collect();
        let groups = group_clusters(&inputs, group_max, pass);
        let mut next_active = Vec::with_capacity(active.len().div_ceil(2));
        let mut reduced_any = false;
        let env = ReduceEnv {
            material_id,
            material_seams,
            quant,
            allow_weld,
            relax_mat_seams: unified || stall_streak >= 1,
            max_parents,
            mark_pinned: unified,
        };
        for group in groups {
            let child_ids: Vec<usize> = group.iter().map(|&li| active[li]).collect();
            match reduce_single_group(
                &env,
                &child_ids,
                &level_pos_count,
                nodes,
                next_group_id,
            )? {
                Some(parent_ids) => {
                    reduced_any = true;
                    next_active.extend(parent_ids);
                }
                None => next_active.extend(child_ids),
            }
        }
        if !reduced_any {
            stall_streak += 1;
            pass += 1;
            if stall_streak >= 6 {
                break;
            }
            continue;
        }
        if next_active.len() * 100 > active.len() * 92 {
            stall_streak += 1;
        } else {
            stall_streak = 0;
        }
        active = next_active;
        pass += 1;
    }
    for &ri in &active {
        let gid = *next_group_id;
        *next_group_id += 1;
        let h = &mut nodes[ri].header;
        h.streaming_meta[0] = gid;
        h.streaming_meta[1] = NO_FALLBACK_CLUSTER;
        h.streaming_meta[2] = FLAG_ROOT_CLUSTER | FLAG_PINNED_COARSE;
        h.streaming_meta[3] = NO_FALLBACK_CLUSTER;
        h.aabb_extent[3] = ROOT_PARENT_ERROR;
        h.parent_lod_sphere = h.lod_sphere;
    }
    Ok(())
}

fn resolve_fallback_target(
    start: u32,
    my_page: u32,
    mat: u8,
    headers: &[ClusterHeader],
    raw_fb0: &[u32],
    mat_root: &HashMap<u8, u32>,
) -> u32 {
    let mut cur = start;
    let mut steps = 0_usize;
    while cur != NO_FALLBACK_CLUSTER && steps < headers.len() {
        if headers[cur as usize].page_id() < my_page {
            return cur;
        }
        cur = raw_fb0[cur as usize];
        steps += 1;
    }
    mat_root.get(&mat).copied().unwrap_or(0)
}

fn pack_pages_and_fallbacks(
    headers: &mut [ClusterHeader],
    payloads: &[Vec<u32>],
) -> (Vec<u32>, u32, u32) {
    let mut page_words = Vec::new();
    let mut page_id = 0_u32;
    let mut word_offset = 0_usize;
    page_words.resize(PAGE_WORDS, 0);
    for (idx, payload) in payloads.iter().enumerate() {
        let needed = payload.len();
        if word_offset + needed > PAGE_WORDS {
            page_id += 1;
            word_offset = 0;
            page_words.resize((page_id as usize + 1) * PAGE_WORDS, 0);
        }
        let base = (page_id as usize) * PAGE_WORDS + word_offset;
        page_words[base..base + needed].copy_from_slice(payload);
        headers[idx].cone_and_meta[2] = page_id;
        headers[idx].cone_and_meta[3] = word_offset as u32;
        word_offset += needed;
    }
    let page_count = page_id + 1;
    let mut max_root_page = 0_u32;
    let mut mat_root: HashMap<u8, u32> = HashMap::with_hasher(FixedState::default());
    for (idx, header) in headers.iter().enumerate() {
        if header.is_root() || header.is_pinned_coarse() {
            max_root_page = max_root_page.max(header.page_id());
        }
        if header.is_root() {
            mat_root.entry(header.material_id()).or_insert(idx as u32);
        }
    }
    let root_page_count = max_root_page + 1;
    let raw_fb0: Vec<u32> = headers.iter().map(|h| h.streaming_meta[1]).collect();
    let raw_fb1: Vec<u32> = headers.iter().map(|h| h.streaming_meta[3]).collect();
    for idx in 0..headers.len() {
        let my_page = headers[idx].page_id();
        if my_page < root_page_count {
            headers[idx].streaming_meta[2] |= FLAG_IN_ROOT_PAGE;
            headers[idx].streaming_meta[1] = NO_FALLBACK_CLUSTER;
            headers[idx].streaming_meta[3] = NO_FALLBACK_CLUSTER;
        } else {
            let mat = headers[idx].material_id();
            let fb0 = resolve_fallback_target(raw_fb0[idx], my_page, mat, headers, &raw_fb0, &mat_root);
            let fb1 = if raw_fb1[idx] != NO_FALLBACK_CLUSTER {
                let r1 = resolve_fallback_target(
                    raw_fb1[idx],
                    my_page,
                    mat,
                    headers,
                    &raw_fb0,
                    &mat_root,
                );
                if r1 == fb0 { NO_FALLBACK_CLUSTER } else { r1 }
            } else {
                NO_FALLBACK_CLUSTER
            };
            headers[idx].streaming_meta[1] = fb0;
            headers[idx].streaming_meta[3] = fb1;
        }
    }
    (page_words, page_count, root_page_count)
}

pub fn build(source: &SourceMesh) -> Result<BuiltMeshlet, Error> {
    let prepared = validate_and_prepare(source)?;
    let mesh_min = Vec3::from_array(prepared.bounds_min);
    let quant = QuantBounds {
        min: mesh_min,
        extent: (Vec3::from_array(prepared.bounds_max) - mesh_min).max(Vec3::ZERO),
    };
    let mut nodes: Vec<DagNode> = Vec::new();
    let mut next_group_id = 1_u32;
    for (&material_id, mat_tris) in &prepared.by_material {
        let leaf_meshlets = partition_meshlets(&prepared.vertices, &prepared.pos_ids, mat_tris);
        let mut active = Vec::with_capacity(leaf_meshlets.len());
        for meshlet in leaf_meshlets {
            let id = nodes.len();
            nodes.push(make_dag_node(meshlet, 0, material_id, quant)?);
            active.push(id);
        }
        reduce_material_dag(
            material_id,
            active,
            &prepared.material_seams,
            quant,
            &mut nodes,
            &mut next_group_id,
        )?;
    }
    let mut headers: Vec<ClusterHeader> = nodes.iter().map(|n| n.header).collect();
    let mut payloads: Vec<Vec<u32>> = nodes.into_iter().map(|n| n.payload).collect();
    let bvh = build_bvh_and_order(&mut headers, &mut payloads);
    let (page_words, page_count, root_page_count) = pack_pages_and_fallbacks(&mut headers, &payloads);
    let (compressed_pages, page_spans) = encode_compressed_pages(&page_words, page_count)?;
    let mut max_lod = 0_u32;
    let mut leaf_tris = 0_u32;
    let mut total_tris = 0_u32;
    let mut total_verts = 0_u32;
    let mut leaf_clusters = 0_u32;
    let mut root_clusters = 0_u32;
    let mut geom_words = 0_u64;
    for header in &headers {
        max_lod = max_lod.max(header.lod_level());
        let tc = header.tri_count();
        let vc = header.vert_count();
        total_tris += tc;
        total_verts += vc;
        geom_words += u64::from(header.word_count());
        if header.lod_level() == 0 {
            leaf_tris += tc;
            leaf_clusters += 1;
        }
        if header.is_root() {
            root_clusters += 1;
        }
    }
    let lod_levels = max_lod + 1;
    let manifest = MeshletManifest {
        source_tri_count: prepared.source_tri_count,
        lod_levels,
        root_page_count,
        page_count,
        bounds_min: prepared.bounds_min,
        bounds_max: prepared.bounds_max,
        clusters: headers,
        bvh,
        page_spans,
    };
    let manifest_bytes = manifest.byte_len() as u64;
    let page_blob_bytes = compressed_pages.len() as u64;
    let geometry_bytes = geom_words * 4;
    let bytes_per_cluster_tri = (geometry_bytes as f64) / f64::from(total_tris.max(1));
    let bytes_per_source_tri =
        (page_blob_bytes as f64) / f64::from(prepared.source_tri_count.max(1));
    let stats = BuildStats {
        source_tri_count: prepared.source_tri_count,
        leaf_tri_count: leaf_tris,
        total_cluster_tris: total_tris,
        total_cluster_verts: total_verts,
        cluster_count: manifest.clusters.len() as u32,
        leaf_cluster_count: leaf_clusters,
        root_cluster_count: root_clusters,
        lod_levels,
        bvh_node_count: manifest.bvh.len() as u32,
        page_count,
        root_page_count,
        geometry_bytes,
        manifest_bytes,
        page_blob_bytes,
        bytes_per_cluster_tri,
        bytes_per_source_tri,
    };
    Ok(BuiltMeshlet {
        manifest,
        page_words,
        compressed_pages,
        stats,
    })
}
