use engine_math::collections::{FixedState, HashMap};
use engine_math::{clamp, max};
use glam::Vec3;

use crate::bounds::{aabb_of_points, morton_3d};
use crate::format::{ClusterVertex, MAX_CLUSTER_TRIS, MAX_CLUSTER_VERTS};

#[derive(Clone, Debug)]
pub struct MeshletData {
    pub vertices: Vec<ClusterVertex>,
    pub pos_ids: Vec<u32>,
    pub triangles: Vec<[u8; 3]>,
}

pub fn weld_positions(positions: &[Vec3]) -> (Vec<u32>, u32) {
    let (min_p, extent) = aabb_of_points(positions);
    let diag = extent.length();
    let grid = max(diag * 1.0e-6, 1.0e-6);
    let inv_grid = 1.0 / grid;
    let mut map: HashMap<[i64; 3], u32> =
        HashMap::with_capacity_and_hasher(positions.len(), FixedState::default());
    let mut ids = Vec::with_capacity(positions.len());
    let mut next_id = 0_u32;
    for &p in positions {
        let rel = p - min_p;
        let key = [
            (rel.x * inv_grid).round() as i64,
            (rel.y * inv_grid).round() as i64,
            (rel.z * inv_grid).round() as i64,
        ];
        let id = *map.entry(key).or_insert_with(|| {
            let assigned = next_id;
            next_id += 1;
            assigned
        });
        ids.push(id);
    }
    (ids, next_id)
}

struct LocalGraph {
    tri_verts: Vec<[u32; 3]>,
    tri_pos: Vec<[u32; 3]>,
    tri_centers: Vec<Vec3>,
    tri_normals: Vec<Vec3>,
    local_to_src_vert: Vec<u32>,
    pos_offsets: Vec<usize>,
    pos_tris: Vec<usize>,
    live_tris: Vec<u32>,
    morton_order: Vec<usize>,
    norm_scale_sq: f32,
    disconnect_limit_sq: f32,
}

fn build_pos_adjacency(tri_pos: &[[u32; 3]], pos_count: usize) -> (Vec<u32>, Vec<usize>, Vec<usize>) {
    let mut live_tris = vec![0_u32; pos_count];
    for &[p0, p1, p2] in tri_pos {
        live_tris[p0 as usize] += 1;
        if p1 != p0 {
            live_tris[p1 as usize] += 1;
        }
        if p2 != p0 && p2 != p1 {
            live_tris[p2 as usize] += 1;
        }
    }
    let mut pos_offsets = vec![0_usize; pos_count + 1];
    for (p, &deg) in live_tris.iter().enumerate() {
        pos_offsets[p + 1] = pos_offsets[p] + deg as usize;
    }
    let mut pos_tris = vec![0_usize; pos_offsets[pos_count]];
    let mut write_cursor = pos_offsets[..pos_count].to_vec();
    for (ti, &[p0, p1, p2]) in tri_pos.iter().enumerate() {
        pos_tris[write_cursor[p0 as usize]] = ti;
        write_cursor[p0 as usize] += 1;
        if p1 != p0 {
            pos_tris[write_cursor[p1 as usize]] = ti;
            write_cursor[p1 as usize] += 1;
        }
        if p2 != p0 && p2 != p1 {
            pos_tris[write_cursor[p2 as usize]] = ti;
            write_cursor[p2 as usize] += 1;
        }
    }
    (live_tris, pos_offsets, pos_tris)
}

fn build_local_graph(
    vertices: &[ClusterVertex],
    pos_ids: &[u32],
    triangles: &[[u32; 3]],
) -> LocalGraph {
    let mut vert_map: HashMap<u32, u32> =
        HashMap::with_capacity_and_hasher(triangles.len(), FixedState::default());
    let mut pos_map: HashMap<u32, u32> =
        HashMap::with_capacity_and_hasher(triangles.len(), FixedState::default());
    let mut local_to_src_vert = Vec::new();
    let mut tri_verts = Vec::with_capacity(triangles.len());
    let mut tri_pos = Vec::with_capacity(triangles.len());
    let mut tri_centers = Vec::with_capacity(triangles.len());
    let mut tri_normals = Vec::with_capacity(triangles.len());
    let mut pos_count = 0_u32;
    let mut edge_sq_sum = 0.0_f64;
    for &[a, b, c] in triangles {
        let mut lv = [0_u32; 3];
        let mut lp = [0_u32; 3];
        for (slot, &sv) in [a, b, c].iter().enumerate() {
            lv[slot] = *vert_map.entry(sv).or_insert_with(|| {
                let id = local_to_src_vert.len() as u32;
                local_to_src_vert.push(sv);
                id
            });
            let gp = pos_ids[sv as usize];
            lp[slot] = *pos_map.entry(gp).or_insert_with(|| {
                let id = pos_count;
                pos_count += 1;
                id
            });
        }
        let pa = vertices[a as usize].position;
        let pb = vertices[b as usize].position;
        let pc = vertices[c as usize].position;
        let cross = (pb - pa).cross(pc - pa);
        let len_sq = cross.length_squared();
        let n = if len_sq.is_finite() && len_sq > 1.0e-18 {
            cross / len_sq.sqrt()
        } else {
            Vec3::ZERO
        };
        tri_verts.push(lv);
        tri_pos.push(lp);
        tri_centers.push((pa + pb + pc) * (1.0 / 3.0));
        tri_normals.push(n);
        edge_sq_sum += f64::from((pb - pa).length_squared());
    }
    let (live_tris, pos_offsets, pos_tris) = build_pos_adjacency(&tri_pos, pos_count as usize);
    let (c_min, c_ext) = aabb_of_points(&tri_centers);
    let mut morton_pairs: Vec<(u32, usize)> = tri_centers
        .iter()
        .enumerate()
        .map(|(ti, &c)| (morton_3d(c, c_min, c_ext), ti))
        .collect();
    morton_pairs.sort_unstable();
    let morton_order = morton_pairs.into_iter().map(|(_, ti)| ti).collect();
    let avg_edge_sq = (edge_sq_sum / (triangles.len().max(1) as f64)) as f32;
    let diag_sq = c_ext.length_squared();
    let norm_scale_sq = max(max(avg_edge_sq * 24.0, diag_sq * 1.0e-6), 1.0e-8);
    let disconnect_limit_sq = max(diag_sq * 0.25, avg_edge_sq * 256.0);
    LocalGraph {
        tri_verts,
        tri_pos,
        tri_centers,
        tri_normals,
        local_to_src_vert,
        pos_offsets,
        pos_tris,
        live_tris,
        morton_order,
        norm_scale_sq,
        disconnect_limit_sq,
    }
}

struct PartitionScratch {
    used_tri: Vec<bool>,
    vert_stamp: Vec<u32>,
    vert_local: Vec<u8>,
    pos_stamp: Vec<u32>,
    cand_stamp: Vec<u32>,
    frontier: Vec<usize>,
    candidates: Vec<usize>,
    cluster_tris: Vec<usize>,
    cluster_verts: Vec<u32>,
    morton_cursor: usize,
    stamp: u32,
}

impl PartitionScratch {
    fn new(tri_count: usize, vert_count: usize, pos_count: usize) -> Self {
        Self {
            used_tri: vec![false; tri_count],
            vert_stamp: vec![0; vert_count],
            vert_local: vec![0; vert_count],
            pos_stamp: vec![0; pos_count],
            cand_stamp: vec![0; tri_count],
            frontier: Vec::new(),
            candidates: Vec::new(),
            cluster_tris: Vec::with_capacity(MAX_CLUSTER_TRIS),
            cluster_verts: Vec::with_capacity(MAX_CLUSTER_VERTS),
            morton_cursor: 0,
            stamp: 0,
        }
    }
}

fn pick_seed(graph: &LocalGraph, scratch: &mut PartitionScratch) -> Option<usize> {
    let mut best_tri = None;
    let mut best_valence = u32::MAX;
    for &ti in &scratch.frontier {
        if scratch.used_tri[ti] {
            continue;
        }
        let [p0, p1, p2] = graph.tri_pos[ti];
        let val = graph.live_tris[p0 as usize]
            + graph.live_tris[p1 as usize]
            + graph.live_tris[p2 as usize];
        if val < best_valence {
            best_valence = val;
            best_tri = Some(ti);
        }
    }
    scratch.frontier.clear();
    if best_tri.is_some() {
        return best_tri;
    }
    while scratch.morton_cursor < graph.morton_order.len()
        && scratch.used_tri[graph.morton_order[scratch.morton_cursor]]
    {
        scratch.morton_cursor += 1;
    }
    if scratch.morton_cursor >= graph.morton_order.len() {
        return None;
    }
    let mut checked = 0_usize;
    let mut scan = scratch.morton_cursor;
    while scan < graph.morton_order.len() && checked < 16 {
        let ti = graph.morton_order[scan];
        scan += 1;
        if scratch.used_tri[ti] {
            continue;
        }
        checked += 1;
        let [p0, p1, p2] = graph.tri_pos[ti];
        let val = graph.live_tris[p0 as usize]
            + graph.live_tris[p1 as usize]
            + graph.live_tris[p2 as usize];
        if val < best_valence {
            best_valence = val;
            best_tri = Some(ti);
        }
    }
    best_tri
}

fn add_triangle_to_cluster(
    ti: usize,
    graph: &mut LocalGraph,
    scratch: &mut PartitionScratch,
    center_sum: &mut Vec3,
    normal_sum: &mut Vec3,
) {
    let stamp = scratch.stamp;
    scratch.used_tri[ti] = true;
    scratch.cluster_tris.push(ti);
    *center_sum += graph.tri_centers[ti];
    *normal_sum += graph.tri_normals[ti];
    for &lv in &graph.tri_verts[ti] {
        let idx = lv as usize;
        if scratch.vert_stamp[idx] != stamp {
            scratch.vert_stamp[idx] = stamp;
            scratch.vert_local[idx] = scratch.cluster_verts.len() as u8;
            scratch.cluster_verts.push(lv);
        }
    }
    let [p0, p1, p2] = graph.tri_pos[ti];
    for (slot, &p) in [p0, p1, p2].iter().enumerate() {
        if (slot == 1 && p == p0) || (slot == 2 && (p == p0 || p == p1)) {
            continue;
        }
        let pidx = p as usize;
        graph.live_tris[pidx] = graph.live_tris[pidx].saturating_sub(1);
        scratch.pos_stamp[pidx] = stamp;
        let start = graph.pos_offsets[pidx];
        let end = graph.pos_offsets[pidx + 1];
        for &nb in &graph.pos_tris[start..end] {
            if !scratch.used_tri[nb] && scratch.cand_stamp[nb] != stamp {
                scratch.cand_stamp[nb] = stamp;
                scratch.candidates.push(nb);
            }
        }
    }
}

fn pick_best_candidate(
    graph: &LocalGraph,
    scratch: &PartitionScratch,
    cluster_center: Vec3,
    cluster_normal: Vec3,
    tight_cones: bool,
) -> Option<usize> {
    let stamp = scratch.stamp;
    let remaining_verts = MAX_CLUSTER_VERTS - scratch.cluster_verts.len();
    let has_cn = cluster_normal.length_squared() > 1.0e-6;
    let mut best_tri = None;
    let mut best_score = f32::NEG_INFINITY;
    for &ti in &scratch.candidates {
        if scratch.used_tri[ti] {
            continue;
        }
        let [v0, v1, v2] = graph.tri_verts[ti];
        let mut new_verts = 0_usize;
        if scratch.vert_stamp[v0 as usize] != stamp {
            new_verts += 1;
        }
        if v1 != v0 && scratch.vert_stamp[v1 as usize] != stamp {
            new_verts += 1;
        }
        if v2 != v0 && v2 != v1 && scratch.vert_stamp[v2 as usize] != stamp {
            new_verts += 1;
        }
        if new_verts > remaining_verts {
            continue;
        }
        let [p0, p1, p2] = graph.tri_pos[ti];
        let mut shared_pos = 0_u32;
        let mut closing = 0_u32;
        for &p in &[p0, p1, p2] {
            if scratch.pos_stamp[p as usize] == stamp {
                shared_pos += 1;
            }
            if graph.live_tris[p as usize] == 1 {
                closing += 1;
            }
        }
        let topo = match shared_pos {
            3 => 1200.0,
            2 => 260.0,
            1 => 25.0,
            _ => 0.0,
        };
        let shared_verts = 3_usize.saturating_sub(new_verts) as f32;
        let dist_sq = (graph.tri_centers[ti] - cluster_center).length_squared();
        let tn = graph.tri_normals[ti];
        let cone_bonus = if tight_cones && has_cn && tn.length_squared() > 1.0e-6 {
            let d = clamp(cluster_normal.dot(tn), -1.0, 1.0);
            if d < 0.0 { d * 240.0 } else { d * 95.0 }
        } else if has_cn && tn.length_squared() > 1.0e-6 {
            clamp(cluster_normal.dot(tn), -1.0, 1.0) * 30.0
        } else {
            0.0
        };
        let score = topo + shared_verts * 55.0 + (closing as f32) * 70.0 + cone_bonus
            - (dist_sq / graph.norm_scale_sq) * 45.0;
        if score > best_score {
            best_score = score;
            best_tri = Some(ti);
        }
    }
    best_tri
}

fn pick_nearby_disconnected(
    graph: &LocalGraph,
    scratch: &mut PartitionScratch,
    cluster_center: Vec3,
    cluster_normal: Vec3,
    tight_cones: bool,
) -> Option<usize> {
    let max_disc_tris = if tight_cones { 96 } else { MAX_CLUSTER_TRIS };
    if scratch.cluster_tris.len() >= max_disc_tris
        || scratch.cluster_verts.len() + 3 > MAX_CLUSTER_VERTS
    {
        return None;
    }
    while scratch.morton_cursor < graph.morton_order.len()
        && scratch.used_tri[graph.morton_order[scratch.morton_cursor]]
    {
        scratch.morton_cursor += 1;
    }
    let has_cn = cluster_normal.length_squared() > 1.0e-6;
    let require_align = tight_cones && has_cn && scratch.cluster_tris.len() >= 12;
    let limit_sq = if tight_cones {
        graph.disconnect_limit_sq
    } else {
        f32::INFINITY
    };
    let mut best_tri = None;
    let mut best_d2 = limit_sq;
    let mut checked = 0_usize;
    let mut scan = scratch.morton_cursor;
    while scan < graph.morton_order.len() && checked < 128 {
        let ti = graph.morton_order[scan];
        scan += 1;
        if scratch.used_tri[ti] {
            continue;
        }
        checked += 1;
        if require_align {
            let tn = graph.tri_normals[ti];
            if tn.length_squared() > 1.0e-6 && cluster_normal.dot(tn) < 0.25 {
                continue;
            }
        }
        let d2 = (graph.tri_centers[ti] - cluster_center).length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_tri = Some(ti);
        }
    }
    best_tri
}

fn partition_meshlets_impl(
    vertices: &[ClusterVertex],
    pos_ids: &[u32],
    triangles: &[[u32; 3]],
    tight_cones: bool,
) -> Vec<MeshletData> {
    if triangles.is_empty() {
        return Vec::new();
    }
    let mut graph = build_local_graph(vertices, pos_ids, triangles);
    let mut scratch = PartitionScratch::new(
        triangles.len(),
        graph.local_to_src_vert.len(),
        graph.live_tris.len(),
    );
    let mut out = Vec::with_capacity(triangles.len().div_ceil(MAX_CLUSTER_TRIS));
    while let Some(seed) = pick_seed(&graph, &mut scratch) {
        scratch.stamp = scratch.stamp.wrapping_add(1);
        scratch.cluster_tris.clear();
        scratch.cluster_verts.clear();
        scratch.candidates.clear();
        let mut center_sum = Vec3::ZERO;
        let mut normal_sum = Vec3::ZERO;
        add_triangle_to_cluster(
            seed,
            &mut graph,
            &mut scratch,
            &mut center_sum,
            &mut normal_sum,
        );
        while scratch.cluster_tris.len() < MAX_CLUSTER_TRIS {
            let center = center_sum / (scratch.cluster_tris.len() as f32);
            let n_len_sq = normal_sum.length_squared();
            let avg_n = if n_len_sq > 1.0e-8 {
                normal_sum / n_len_sq.sqrt()
            } else {
                Vec3::ZERO
            };
            let next = pick_best_candidate(&graph, &scratch, center, avg_n, tight_cones)
                .or_else(|| pick_nearby_disconnected(&graph, &mut scratch, center, avg_n, tight_cones));
            let Some(chosen) = next else {
                break;
            };
            add_triangle_to_cluster(
                chosen,
                &mut graph,
                &mut scratch,
                &mut center_sum,
                &mut normal_sum,
            );
        }
        for &cand in &scratch.candidates {
            if !scratch.used_tri[cand] && scratch.frontier.len() < 64 {
                scratch.frontier.push(cand);
            }
        }
        let mut out_verts = Vec::with_capacity(scratch.cluster_verts.len());
        let mut out_pos = Vec::with_capacity(scratch.cluster_verts.len());
        for &lv in &scratch.cluster_verts {
            let sv = graph.local_to_src_vert[lv as usize] as usize;
            out_verts.push(vertices[sv]);
            out_pos.push(pos_ids[sv]);
        }
        let mut out_tris = Vec::with_capacity(scratch.cluster_tris.len());
        for &ti in &scratch.cluster_tris {
            let [v0, v1, v2] = graph.tri_verts[ti];
            out_tris.push([
                scratch.vert_local[v0 as usize],
                scratch.vert_local[v1 as usize],
                scratch.vert_local[v2 as usize],
            ]);
        }
        out.push(MeshletData {
            vertices: out_verts,
            pos_ids: out_pos,
            triangles: out_tris,
        });
    }
    out
}

pub fn partition_meshlets(
    vertices: &[ClusterVertex],
    pos_ids: &[u32],
    triangles: &[[u32; 3]],
) -> Vec<MeshletData> {
    partition_meshlets_impl(vertices, pos_ids, triangles, true)
}

pub fn partition_meshlets_compact(
    vertices: &[ClusterVertex],
    pos_ids: &[u32],
    triangles: &[[u32; 3]],
) -> Vec<MeshletData> {
    partition_meshlets_impl(vertices, pos_ids, triangles, false)
}
