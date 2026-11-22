use std::cmp::Ordering;
use std::collections::BinaryHeap;

use engine_math::collections::{FixedState, HashMap, HashSet};
use engine_math::{clamp, max};
use glam::{DVec3, Vec2, Vec3};

use crate::format::ClusterVertex;
use crate::partition::MeshletData;

pub struct SimplifiedGroup {
    pub vertices: Vec<ClusterVertex>,
    pub pos_ids: Vec<u32>,
    pub triangles: Vec<[u32; 3]>,
    pub error: f32,
}

#[derive(Clone, Copy, Default)]
struct Quadric {
    a00: f64,
    a01: f64,
    a02: f64,
    a03: f64,
    a11: f64,
    a12: f64,
    a13: f64,
    a22: f64,
    a23: f64,
    a33: f64,
    weight: f64,
}

impl Quadric {
    fn from_plane(n: DVec3, d: f64, weight: f64) -> Self {
        let (a, b, c) = (n.x, n.y, n.z);
        Self {
            a00: a * a * weight,
            a01: a * b * weight,
            a02: a * c * weight,
            a03: a * d * weight,
            a11: b * b * weight,
            a12: b * c * weight,
            a13: b * d * weight,
            a22: c * c * weight,
            a23: c * d * weight,
            a33: d * d * weight,
            weight,
        }
    }

    fn add(&mut self, other: &Self) {
        self.a00 += other.a00;
        self.a01 += other.a01;
        self.a02 += other.a02;
        self.a03 += other.a03;
        self.a11 += other.a11;
        self.a12 += other.a12;
        self.a13 += other.a13;
        self.a22 += other.a22;
        self.a23 += other.a23;
        self.a33 += other.a33;
        self.weight += other.weight;
    }

    fn sum(&self, other: &Self) -> Self {
        let mut out = *self;
        out.add(other);
        out
    }

    fn eval(&self, p: DVec3) -> f64 {
        let (x, y, z) = (p.x, p.y, p.z);
        let value = self.a00 * x * x
            + 2.0 * self.a01 * x * y
            + 2.0 * self.a02 * x * z
            + 2.0 * self.a03 * x
            + self.a11 * y * y
            + 2.0 * self.a12 * y * z
            + 2.0 * self.a13 * y
            + self.a22 * z * z
            + 2.0 * self.a23 * z
            + self.a33;
        if value > 0.0 { value } else { 0.0 }
    }

    fn solve_optimal(&self) -> Option<DVec3> {
        let c00 = self.a11 * self.a22 - self.a12 * self.a12;
        let c01 = self.a02 * self.a12 - self.a01 * self.a22;
        let c02 = self.a01 * self.a12 - self.a02 * self.a11;
        let c11 = self.a00 * self.a22 - self.a02 * self.a02;
        let c12 = self.a01 * self.a02 - self.a00 * self.a12;
        let c22 = self.a00 * self.a11 - self.a01 * self.a01;
        let det = self.a00 * c00 + self.a01 * c01 + self.a02 * c02;
        let trace = self.a00 + self.a11 + self.a22;
        let scale = trace * trace * trace;
        if !det.is_finite() || det.abs() <= 1.0e-12 * scale.max(1.0e-12) {
            return None;
        }
        let inv_det = -1.0 / det;
        let x = (c00 * self.a03 + c01 * self.a13 + c02 * self.a23) * inv_det;
        let y = (c01 * self.a03 + c11 * self.a13 + c12 * self.a23) * inv_det;
        let z = (c02 * self.a03 + c12 * self.a13 + c22 * self.a23) * inv_det;
        let p = DVec3::new(x, y, z);
        p.is_finite().then_some(p)
    }
}

#[derive(Clone, Copy)]
struct CornerAttr {
    normal: Vec3,
    uv: Vec2,
    color: [f32; 3],
    ao: f32,
}

#[derive(Clone, Copy)]
struct SimpTri {
    v: [u32; 3],
    attrs: [CornerAttr; 3],
    ref_n: DVec3,
    alive: bool,
}

#[derive(Clone, Copy)]
struct CollapseEntry {
    cost: f64,
    geom_err: f64,
    keep: u32,
    remove: u32,
    ver_keep: u32,
    ver_remove: u32,
    target_pos: DVec3,
}

impl PartialEq for CollapseEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost.to_bits() == other.cost.to_bits()
            && self.keep == other.keep
            && self.remove == other.remove
    }
}

impl Eq for CollapseEntry {}

impl PartialOrd for CollapseEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CollapseEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.keep.cmp(&self.keep))
            .then_with(|| other.remove.cmp(&self.remove))
    }
}

struct SimplifierState {
    positions: Vec<DVec3>,
    global_pos: Vec<u32>,
    plane_q: Vec<Quadric>,
    boundary_q: Vec<Quadric>,
    hard_locked: Vec<bool>,
    locked: Vec<bool>,
    uv_seam: Vec<bool>,
    alive_vert: Vec<bool>,
    ver: Vec<u32>,
    vert_tris: Vec<Vec<usize>>,
    tris: Vec<SimpTri>,
    live_tris: usize,
    live_verts: usize,
    max_error: f64,
}

struct QuadricAndLocks {
    vert_tris: Vec<Vec<usize>>,
    plane_q: Vec<Quadric>,
    boundary_q: Vec<Quadric>,
    hard_locked: Vec<bool>,
    locked: Vec<bool>,
    uv_seam: Vec<bool>,
}

fn build_quadrics_and_locks(
    positions: &[DVec3],
    global_pos: &[u32],
    tris: &[SimpTri],
    hard_locked_pos: &HashSet<u32>,
) -> QuadricAndLocks {
    let v_count = positions.len();
    let mut edge_counts: HashMap<(u32, u32), u32> =
        HashMap::with_capacity_and_hasher(tris.len() * 3, FixedState::default());
    let mut vert_tris = vec![Vec::new(); v_count];
    let mut plane_q = vec![Quadric::default(); v_count];
    let mut boundary_q = vec![Quadric::default(); v_count];
    for (ti, tri) in tris.iter().enumerate() {
        let [a, b, c] = tri.v;
        vert_tris[a as usize].push(ti);
        vert_tris[b as usize].push(ti);
        vert_tris[c as usize].push(ti);
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            *edge_counts.entry(key).or_insert(0) += 1;
        }
        let pa = positions[a as usize];
        let pb = positions[b as usize];
        let pc = positions[c as usize];
        let cross = (pb - pa).cross(pc - pa);
        let twice_area = cross.length();
        if twice_area > 1.0e-16 {
            let n = cross / twice_area;
            let d = -n.dot(pa);
            let q = Quadric::from_plane(n, d, 0.5 * twice_area);
            plane_q[a as usize].add(&q);
            plane_q[b as usize].add(&q);
            plane_q[c as usize].add(&q);
        }
    }
    let mut is_open = vec![false; v_count];
    for tri in tris {
        let [a, b, c] = tri.v;
        let pa = positions[a as usize];
        let pb = positions[b as usize];
        let pc = positions[c as usize];
        let cross = (pb - pa).cross(pc - pa);
        let twice_area = cross.length();
        let n = if twice_area > 1.0e-16 {
            cross / twice_area
        } else {
            continue;
        };
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            if edge_counts.get(&key).copied().unwrap_or(0) == 1 {
                is_open[u as usize] = true;
                is_open[v as usize] = true;
                let edge = positions[v as usize] - positions[u as usize];
                let bn = edge.cross(n);
                let bn_len = bn.length();
                if bn_len > 1.0e-16 {
                    let b_unit = bn / bn_len;
                    let bd = -b_unit.dot(positions[u as usize]);
                    let bq = Quadric::from_plane(b_unit, bd, 12.0 * edge.length_squared());
                    boundary_q[u as usize].add(&bq);
                    boundary_q[v as usize].add(&bq);
                }
            }
        }
    }
    let mut first_uv: Vec<Option<Vec2>> = vec![None; v_count];
    let mut uv_seam = vec![false; v_count];
    for tri in tris {
        for slot in 0..3 {
            let vi = tri.v[slot] as usize;
            let uv = tri.attrs[slot].uv;
            if let Some(prev) = first_uv[vi] {
                if (prev - uv).length_squared() > 1.0e-4 {
                    uv_seam[vi] = true;
                }
            } else {
                first_uv[vi] = Some(uv);
            }
        }
    }
    let hard_locked: Vec<bool> = global_pos
        .iter()
        .map(|gp| hard_locked_pos.contains(gp))
        .collect();
    let locked: Vec<bool> = (0..v_count)
        .map(|i| hard_locked[i] || is_open[i])
        .collect();
    QuadricAndLocks {
        vert_tris,
        plane_q,
        boundary_q,
        hard_locked,
        locked,
        uv_seam,
    }
}

fn init_state(
    group_clusters: &[&MeshletData],
    hard_locked_pos: &HashSet<u32>,
) -> SimplifierState {
    let mut pos_map: HashMap<u32, u32> = HashMap::with_hasher(FixedState::default());
    let mut positions = Vec::new();
    let mut global_pos = Vec::new();
    let mut tris = Vec::new();
    for meshlet in group_clusters {
        let mut local_ids = Vec::with_capacity(meshlet.vertices.len());
        for (vi, &gp) in meshlet.pos_ids.iter().enumerate() {
            let lid = *pos_map.entry(gp).or_insert_with(|| {
                let id = positions.len() as u32;
                let p = meshlet.vertices[vi].position;
                positions.push(DVec3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z)));
                global_pos.push(gp);
                id
            });
            local_ids.push(lid);
        }
        for &[i0, i1, i2] in &meshlet.triangles {
            let a = local_ids[usize::from(i0)];
            let b = local_ids[usize::from(i1)];
            let c = local_ids[usize::from(i2)];
            if a == b || b == c || c == a {
                continue;
            }
            let attr = |idx: u8| {
                let v = meshlet.vertices[usize::from(idx)];
                CornerAttr {
                    normal: v.normal,
                    uv: v.uv,
                    color: v.color,
                    ao: v.ao,
                }
            };
            let pa = positions[a as usize];
            let pb = positions[b as usize];
            let pc = positions[c as usize];
            let cross = (pb - pa).cross(pc - pa);
            let clen = cross.length();
            let ref_n = if clen > 1.0e-14 {
                cross / clen
            } else {
                DVec3::Y
            };
            tris.push(SimpTri {
                v: [a, b, c],
                attrs: [attr(i0), attr(i1), attr(i2)],
                ref_n,
                alive: true,
            });
        }
    }
    let v_count = positions.len();
    let ql = build_quadrics_and_locks(&positions, &global_pos, &tris, hard_locked_pos);
    let live_tris = tris.len();
    SimplifierState {
        positions,
        global_pos,
        plane_q: ql.plane_q,
        boundary_q: ql.boundary_q,
        hard_locked: ql.hard_locked,
        locked: ql.locked,
        uv_seam: ql.uv_seam,
        alive_vert: vec![true; v_count],
        ver: vec![0; v_count],
        vert_tris: ql.vert_tris,
        tris,
        live_tris,
        live_verts: v_count,
        max_error: 0.0,
    }
}

fn evaluate_edge(state: &SimplifierState, u: u32, v: u32) -> Option<CollapseEntry> {
    let lu = state.locked[u as usize];
    let lv = state.locked[v as usize];
    if lu && lv {
        return None;
    }
    let pq = state.plane_q[u as usize].sum(&state.plane_q[v as usize]);
    let bq = state.boundary_q[u as usize].sum(&state.boundary_q[v as usize]);
    let total_q = pq.sum(&bq);
    let pu = state.positions[u as usize];
    let pv = state.positions[v as usize];
    let (keep, remove, target_pos) = if lu {
        (u, v, pu)
    } else if lv {
        (v, u, pv)
    } else {
        let mid = (pu + pv) * 0.5;
        let edge_len = (pv - pu).length();
        let mut best_p = mid;
        let mut best_c = total_q.eval(mid);
        for candidate in [pu, pv] {
            let c = total_q.eval(candidate);
            if c < best_c {
                best_c = c;
                best_p = candidate;
            }
        }
        if let Some(opt) = total_q.solve_optimal()
            && (opt - mid).length() <= edge_len * 2.0 + 1.0e-6
            && total_q.eval(opt) < best_c
        {
            best_p = opt;
        }
        (u, v, best_p)
    };
    let edge_len = (pv - pu).length();
    let mut cost = total_q.eval(target_pos);
    let plane_sq = pq.eval(target_pos) / pq.weight.max(1.0e-12);
    let bnd_sq = if bq.weight > 1.0e-12 {
        bq.eval(target_pos) / bq.weight
    } else {
        0.0
    };
    let mut geom_err = plane_sq.max(bnd_sq * 0.25).sqrt();
    let seam_edge = state.uv_seam[u as usize] || state.uv_seam[v as usize];
    if seam_edge {
        geom_err = geom_err.max(0.25 * edge_len);
        cost += edge_len * edge_len * 36.0;
    } else {
        let mut max_duv_sq = 0.0_f32;
        for &ti in &state.vert_tris[u as usize] {
            let tri = &state.tris[ti];
            if !tri.alive {
                continue;
            }
            let mut ui = None;
            let mut vi = None;
            for slot in 0..3 {
                if tri.v[slot] == u {
                    ui = Some(slot);
                } else if tri.v[slot] == v {
                    vi = Some(slot);
                }
            }
            if let (Some(a), Some(b)) = (ui, vi) {
                max_duv_sq = max(
                    max_duv_sq,
                    (tri.attrs[a].uv - tri.attrs[b].uv).length_squared(),
                );
            }
        }
        if max_duv_sq > 0.04 {
            geom_err = geom_err.max(0.12 * edge_len);
            cost += edge_len * edge_len * 8.0;
        }
    }
    Some(CollapseEntry {
        cost,
        geom_err,
        keep,
        remove,
        ver_keep: state.ver[keep as usize],
        ver_remove: state.ver[remove as usize],
        target_pos,
    })
}

fn push_all_edges(state: &SimplifierState, heap: &mut BinaryHeap<CollapseEntry>) {
    let mut seen: HashSet<(u32, u32)> =
        HashSet::with_capacity_and_hasher(state.live_tris * 3, FixedState::default());
    for tri in &state.tris {
        if !tri.alive {
            continue;
        }
        let [a, b, c] = tri.v;
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            if seen.insert(key)
                && let Some(entry) = evaluate_edge(state, key.0, key.1)
            {
                heap.push(entry);
            }
        }
    }
}

fn link_condition_ok(state: &SimplifierState, keep: u32, remove: u32) -> bool {
    let mut adj_k: Vec<u32> = Vec::new();
    let mut shared_opp: Vec<u32> = Vec::new();
    for &ti in &state.vert_tris[keep as usize] {
        let tri = &state.tris[ti];
        if !tri.alive {
            continue;
        }
        let has_r = tri.v.contains(&remove);
        for &v in &tri.v {
            if v == keep || v == remove {
                continue;
            }
            if !adj_k.contains(&v) {
                adj_k.push(v);
            }
            if has_r && !shared_opp.contains(&v) {
                shared_opp.push(v);
            }
        }
    }
    if shared_opp.is_empty() || shared_opp.len() > 2 {
        return false;
    }
    for &ti in &state.vert_tris[remove as usize] {
        let tri = &state.tris[ti];
        if !tri.alive || tri.v.contains(&keep) {
            continue;
        }
        for &v in &tri.v {
            if v != remove && adj_k.contains(&v) && !shared_opp.contains(&v) {
                return false;
            }
        }
    }
    true
}

fn collapse_is_valid(
    state: &SimplifierState,
    keep: u32,
    remove: u32,
    target: DVec3,
    min_dot: f64,
    check_link: bool,
) -> bool {
    if check_link && !link_condition_ok(state, keep, remove) {
        return false;
    }
    let mut has_shared = false;
    let mut surviving_faces: Vec<[u32; 3]> = Vec::new();
    for &vert in &[keep, remove] {
        for &ti in &state.vert_tris[vert as usize] {
            let tri = &state.tris[ti];
            if !tri.alive {
                continue;
            }
            let has_k = tri.v.contains(&keep);
            let has_r = tri.v.contains(&remove);
            if has_k && has_r {
                has_shared = true;
                continue;
            }
            let p_old = [
                state.positions[tri.v[0] as usize],
                state.positions[tri.v[1] as usize],
                state.positions[tri.v[2] as usize],
            ];
            let p_new = [
                if tri.v[0] == keep || tri.v[0] == remove {
                    target
                } else {
                    p_old[0]
                },
                if tri.v[1] == keep || tri.v[1] == remove {
                    target
                } else {
                    p_old[1]
                },
                if tri.v[2] == keep || tri.v[2] == remove {
                    target
                } else {
                    p_old[2]
                },
            ];
            let n_old = (p_old[1] - p_old[0]).cross(p_old[2] - p_old[0]);
            let n_new = (p_new[1] - p_new[0]).cross(p_new[2] - p_new[0]);
            let old_len = n_old.length();
            let new_len = n_new.length();
            if new_len <= 1.0e-14 {
                return false;
            }
            if old_len > 1.0e-14 && n_old.dot(n_new) < min_dot * old_len * new_len {
                return false;
            }
            if tri.ref_n.dot(n_new) < min_dot * new_len {
                return false;
            }
            let vn = (tri.attrs[0].normal + tri.attrs[1].normal + tri.attrs[2].normal).as_dvec3();
            let vn_len = vn.length();
            if vn_len > 1.0e-6 && vn.dot(n_new) < (min_dot * 0.75) * vn_len * new_len {
                return false;
            }
            let mut key = [
                if tri.v[0] == remove { keep } else { tri.v[0] },
                if tri.v[1] == remove { keep } else { tri.v[1] },
                if tri.v[2] == remove { keep } else { tri.v[2] },
            ];
            key.sort_unstable();
            if surviving_faces.contains(&key) {
                return false;
            }
            surviving_faces.push(key);
        }
    }
    has_shared
}

fn blend_attr(a: CornerAttr, b: CornerAttr, t: f32) -> CornerAttr {
    if a.normal.dot(b.normal) < 0.7 || (a.uv - b.uv).length_squared() > 4.0 {
        return a;
    }
    let n = a.normal.lerp(b.normal, t);
    let normal = if n.length_squared() > 1.0e-12 {
        n.normalize()
    } else {
        a.normal
    };
    CornerAttr {
        normal,
        uv: a.uv.lerp(b.uv, t),
        color: [
            clamp(a.color[0] + (b.color[0] - a.color[0]) * t, 0.0, 1.0),
            clamp(a.color[1] + (b.color[1] - a.color[1]) * t, 0.0, 1.0),
            clamp(a.color[2] + (b.color[2] - a.color[2]) * t, 0.0, 1.0),
        ],
        ao: clamp(a.ao + (b.ao - a.ao) * t, 0.0, 1.0),
    }
}

fn apply_collapse(
    state: &mut SimplifierState,
    entry: CollapseEntry,
    heap: &mut BinaryHeap<CollapseEntry>,
) {
    let keep = entry.keep;
    let remove = entry.remove;
    let p_keep_old = state.positions[keep as usize];
    let p_rem_old = state.positions[remove as usize];
    let edge = p_rem_old - p_keep_old;
    let edge_len_sq = edge.length_squared();
    let t = if edge_len_sq > 1.0e-16 {
        clamp(((entry.target_pos - p_keep_old).dot(edge) / edge_len_sq) as f32, 0.0, 1.0)
    } else {
        0.0
    };
    let mut ref_keep_attr = None;
    let mut ref_rem_attr = None;
    for &ti in &state.vert_tris[keep as usize] {
        let tri = &mut state.tris[ti];
        if !tri.alive {
            continue;
        }
        let mut ki = None;
        let mut ri = None;
        for slot in 0..3 {
            if tri.v[slot] == keep {
                ki = Some(slot);
            } else if tri.v[slot] == remove {
                ri = Some(slot);
            }
        }
        if let (Some(ks), Some(rs)) = (ki, ri) {
            if ref_keep_attr.is_none() {
                ref_keep_attr = Some(tri.attrs[ks]);
                ref_rem_attr = Some(tri.attrs[rs]);
            }
            tri.alive = false;
            state.live_tris = state.live_tris.saturating_sub(1);
        }
    }
    let merged_attr = match (ref_keep_attr, ref_rem_attr) {
        (Some(ka), Some(ra)) => Some(blend_attr(ka, ra, t)),
        _ => None,
    };
    let rem_tris = std::mem::take(&mut state.vert_tris[remove as usize]);
    for ti in rem_tris {
        let tri = &mut state.tris[ti];
        if !tri.alive {
            continue;
        }
        for slot in 0..3 {
            if tri.v[slot] == remove {
                tri.v[slot] = keep;
                if let Some(ma) = merged_attr
                    && tri.attrs[slot].normal.dot(ma.normal) > 0.7
                    && (tri.attrs[slot].uv - ma.uv).length_squared() < 4.0
                {
                    tri.attrs[slot] = ma;
                }
            }
        }
        state.vert_tris[keep as usize].push(ti);
    }
    if let Some(ma) = merged_attr {
        for &ti in &state.vert_tris[keep as usize] {
            let tri = &mut state.tris[ti];
            if !tri.alive {
                continue;
            }
            for slot in 0..3 {
                if tri.v[slot] == keep
                    && tri.attrs[slot].normal.dot(ma.normal) > 0.7
                    && (tri.attrs[slot].uv - ma.uv).length_squared() < 4.0
                {
                    tri.attrs[slot] = ma;
                }
            }
        }
    }
    let q_p = state.plane_q[remove as usize];
    let q_b = state.boundary_q[remove as usize];
    state.plane_q[keep as usize].add(&q_p);
    state.boundary_q[keep as usize].add(&q_b);
    state.positions[keep as usize] = entry.target_pos;
    state.locked[keep as usize] |= state.locked[remove as usize];
    state.hard_locked[keep as usize] |= state.hard_locked[remove as usize];
    state.uv_seam[keep as usize] |= state.uv_seam[remove as usize];
    state.alive_vert[remove as usize] = false;
    state.live_verts = state.live_verts.saturating_sub(1);
    state.ver[keep as usize] = state.ver[keep as usize].wrapping_add(1);
    if entry.geom_err > state.max_error {
        state.max_error = entry.geom_err;
    }
    state.vert_tris[keep as usize].retain(|&ti| state.tris[ti].alive);
    if state.vert_tris[keep as usize].is_empty() && state.alive_vert[keep as usize] {
        state.alive_vert[keep as usize] = false;
        state.live_verts = state.live_verts.saturating_sub(1);
    }
    let mut neighbors = Vec::new();
    for &ti in &state.vert_tris[keep as usize] {
        for &v in &state.tris[ti].v {
            if v != keep && state.alive_vert[v as usize] && !neighbors.contains(&v) {
                neighbors.push(v);
            }
        }
    }
    for nb in neighbors {
        if let Some(next_entry) = evaluate_edge(state, keep, nb) {
            heap.push(next_entry);
        }
    }
}

fn run_collapse_pass(
    state: &mut SimplifierState,
    target_tris: usize,
    target_verts: usize,
    min_dot: f64,
    check_link: bool,
) {
    let mut heap = BinaryHeap::new();
    push_all_edges(state, &mut heap);
    while (state.live_tris > target_tris || state.live_verts > target_verts) && state.live_tris > 1
    {
        let Some(entry) = heap.pop() else {
            break;
        };
        if !state.alive_vert[entry.keep as usize]
            || !state.alive_vert[entry.remove as usize]
            || state.ver[entry.keep as usize] != entry.ver_keep
            || state.ver[entry.remove as usize] != entry.ver_remove
        {
            continue;
        }
        if !collapse_is_valid(
            state,
            entry.keep,
            entry.remove,
            entry.target_pos,
            min_dot,
            check_link,
        ) {
            continue;
        }
        apply_collapse(state, entry, &mut heap);
    }
}

fn export_simplified(state: &SimplifierState) -> SimplifiedGroup {
    let mut per_pos: Vec<Vec<u32>> = vec![Vec::new(); state.positions.len()];
    let mut vertices: Vec<ClusterVertex> = Vec::new();
    let mut pos_ids: Vec<u32> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(state.live_tris);
    for tri in &state.tris {
        if !tri.alive {
            continue;
        }
        let mut out_tri = [0_u32; 3];
        for (slot, out_slot) in out_tri.iter_mut().enumerate() {
            let lv = tri.v[slot] as usize;
            let attr = tri.attrs[slot];
            let mut found = None;
            for &vid in &per_pos[lv] {
                let existing = &vertices[vid as usize];
                if existing.normal.dot(attr.normal) > 0.96
                    && (existing.uv - attr.uv).length_squared() < 1.0e-3
                    && (existing.ao - attr.ao).abs() < 5.0e-2
                    && (existing.color[0] - attr.color[0]).abs() < 5.0e-2
                    && (existing.color[1] - attr.color[1]).abs() < 5.0e-2
                    && (existing.color[2] - attr.color[2]).abs() < 5.0e-2
                {
                    found = Some(vid);
                    break;
                }
            }
            let vid = match found {
                Some(id) => id,
                None => {
                    let id = vertices.len() as u32;
                    let p = state.positions[lv];
                    vertices.push(ClusterVertex {
                        position: Vec3::new(p.x as f32, p.y as f32, p.z as f32),
                        normal: if attr.normal.length_squared() > 1.0e-12 {
                            attr.normal.normalize()
                        } else {
                            Vec3::Y
                        },
                        uv: attr.uv,
                        color: [
                            clamp(attr.color[0], 0.0, 1.0),
                            clamp(attr.color[1], 0.0, 1.0),
                            clamp(attr.color[2], 0.0, 1.0),
                        ],
                        ao: clamp(attr.ao, 0.0, 1.0),
                    });
                    pos_ids.push(state.global_pos[lv]);
                    per_pos[lv].push(id);
                    id
                }
            };
            *out_slot = vid;
        }
        if out_tri[0] != out_tri[1] && out_tri[1] != out_tri[2] && out_tri[2] != out_tri[0] {
            triangles.push(out_tri);
        }
    }
    let error = max(state.max_error as f32, 0.0);
    SimplifiedGroup {
        vertices,
        pos_ids,
        triangles,
        error,
    }
}

pub fn simplify_group(
    group_clusters: &[&MeshletData],
    hard_locked_pos: &HashSet<u32>,
    target_tris: usize,
    target_verts: usize,
    allow_open_edge_fallback: bool,
) -> SimplifiedGroup {
    let mut state = init_state(group_clusters, hard_locked_pos);
    let initial_tris = state.live_tris;
    let clamped_tris = target_tris.max(1);
    let clamped_verts = target_verts.max(3);
    run_collapse_pass(&mut state, clamped_tris, clamped_verts, 0.40, true);
    if state.live_tris > clamped_tris || state.live_verts > clamped_verts {
        run_collapse_pass(&mut state, clamped_tris, clamped_verts, 0.20, false);
    }
    let needs_more = state.live_tris > clamped_tris || state.live_verts > clamped_verts;
    let stalled = needs_more
        && (state.live_tris * 5 > initial_tris * 4 || hard_locked_pos.is_empty());
    if allow_open_edge_fallback && stalled {
        let mut unlocked_any = false;
        for i in 0..state.locked.len() {
            if state.locked[i] && !state.hard_locked[i] {
                state.locked[i] = false;
                unlocked_any = true;
            }
        }
        if unlocked_any {
            run_collapse_pass(&mut state, clamped_tris, clamped_verts, 0.15, false);
        }
        if (state.live_tris > clamped_tris || state.live_verts > clamped_verts)
            && hard_locked_pos.is_empty()
        {
            run_collapse_pass(&mut state, clamped_tris, clamped_verts, 0.08, false);
        }
    }
    export_simplified(&state)
}

#[derive(Default)]
struct WeldCell {
    q: Quadric,
    pos_sum: DVec3,
    norm_sum: Vec3,
    uv_sum: Vec2,
    col_sum: [f32; 3],
    ao_sum: f32,
    weight_sum: f64,
    pos_id: u32,
}

fn finalize_welded(
    cells: &[WeldCell],
    raw_tris: Vec<([u32; 3], Vec3)>,
    cell: f32,
) -> (Vec<ClusterVertex>, Vec<u32>, Vec<[u32; 3]>, f32) {
    let mut vertices = Vec::with_capacity(cells.len());
    let mut pos_ids = Vec::with_capacity(cells.len());
    let mut max_err = 0.0_f64;
    for c in cells {
        let w = c.weight_sum.max(1.0e-12);
        let avg = c.pos_sum / w;
        let p = match c.q.solve_optimal() {
            Some(opt) if (opt - avg).length() <= f64::from(cell) * 1.25 => opt,
            _ => avg,
        };
        if c.q.weight > 1.0e-12 {
            let err = (c.q.eval(p) / c.q.weight).sqrt();
            if err > max_err {
                max_err = err;
            }
        }
        let wf = w as f32;
        let normal = if c.norm_sum.length_squared() > 1.0e-12 {
            c.norm_sum.normalize()
        } else {
            Vec3::Y
        };
        vertices.push(ClusterVertex {
            position: Vec3::new(p.x as f32, p.y as f32, p.z as f32),
            normal,
            uv: c.uv_sum / wf,
            color: [
                clamp(c.col_sum[0] / wf, 0.0, 1.0),
                clamp(c.col_sum[1] / wf, 0.0, 1.0),
                clamp(c.col_sum[2] / wf, 0.0, 1.0),
            ],
            ao: clamp(c.ao_sum / wf, 0.0, 1.0),
        });
        pos_ids.push(c.pos_id);
    }
    let mut seen: HashSet<[u32; 3]> =
        HashSet::with_capacity_and_hasher(raw_tris.len(), FixedState::default());
    let mut triangles = Vec::with_capacity(raw_tris.len());
    let mut smooth_n = vec![Vec3::ZERO; vertices.len()];
    for (mut tri, orig_n) in raw_tris {
        let mut key = tri;
        key.sort_unstable();
        if !seen.insert(key) {
            continue;
        }
        let p0 = vertices[tri[0] as usize].position;
        let p1 = vertices[tri[1] as usize].position;
        let p2 = vertices[tri[2] as usize].position;
        let mut n_new = (p1 - p0).cross(p2 - p0);
        if n_new.length_squared() <= 1.0e-14 {
            continue;
        }
        if n_new.dot(orig_n) < 0.0 {
            tri.swap(1, 2);
            n_new = -n_new;
        }
        smooth_n[tri[0] as usize] += n_new;
        smooth_n[tri[1] as usize] += n_new;
        smooth_n[tri[2] as usize] += n_new;
        triangles.push(tri);
    }
    for (v, sn) in vertices.iter_mut().zip(smooth_n) {
        if sn.length_squared() > 1.0e-14 {
            v.normal = sn.normalize();
        }
    }
    (vertices, pos_ids, triangles, max(max_err as f32, cell * 0.5))
}

fn build_welded_at_cell(
    group_clusters: &[&MeshletData],
    hard_locked: &HashSet<u32>,
    bmin: Vec3,
    cell: f32,
) -> (Vec<ClusterVertex>, Vec<u32>, Vec<[u32; 3]>, f32) {
    let inv = 1.0 / max(cell, 1.0e-5);
    let mut cell_map: HashMap<[i64; 3], u32> = HashMap::with_hasher(FixedState::default());
    let mut locked_map: HashMap<u32, u32> = HashMap::with_hasher(FixedState::default());
    let mut cells: Vec<WeldCell> = Vec::new();
    let mut raw_tris: Vec<([u32; 3], Vec3)> = Vec::new();
    for m in group_clusters {
        let mut local_to_cell = Vec::with_capacity(m.vertices.len());
        for (vi, v) in m.vertices.iter().enumerate() {
            let gp = m.pos_ids[vi];
            let cid = if hard_locked.contains(&gp) {
                *locked_map.entry(gp).or_insert_with(|| {
                    let id = cells.len() as u32;
                    cells.push(WeldCell {
                        pos_id: gp,
                        ..WeldCell::default()
                    });
                    id
                })
            } else {
                let rel = (v.position - bmin) * inv;
                let key = [
                    rel.x.floor() as i64,
                    rel.y.floor() as i64,
                    rel.z.floor() as i64,
                ];
                *cell_map.entry(key).or_insert_with(|| {
                    let id = cells.len() as u32;
                    cells.push(WeldCell {
                        pos_id: gp,
                        ..WeldCell::default()
                    });
                    id
                })
            };
            local_to_cell.push(cid);
        }
        for &[i0, i1, i2] in &m.triangles {
            let v0 = m.vertices[usize::from(i0)];
            let v1 = m.vertices[usize::from(i1)];
            let v2 = m.vertices[usize::from(i2)];
            let p0 = DVec3::new(f64::from(v0.position.x), f64::from(v0.position.y), f64::from(v0.position.z));
            let p1 = DVec3::new(f64::from(v1.position.x), f64::from(v1.position.y), f64::from(v1.position.z));
            let p2 = DVec3::new(f64::from(v2.position.x), f64::from(v2.position.y), f64::from(v2.position.z));
            let cross = (p1 - p0).cross(p2 - p0);
            let clen = cross.length();
            let area = (clen * 0.5).max(1.0e-8);
            let q = if clen > 1.0e-14 {
                let n = cross / clen;
                Quadric::from_plane(n, -n.dot(p0), area)
            } else {
                Quadric::default()
            };
            for (li, pv) in [(i0, p0), (i1, p1), (i2, p2)] {
                let vert = m.vertices[usize::from(li)];
                let c = &mut cells[local_to_cell[usize::from(li)] as usize];
                c.q.add(&q);
                c.pos_sum += pv * area;
                let af = area as f32;
                c.norm_sum += vert.normal * af;
                c.uv_sum += vert.uv * af;
                c.col_sum[0] += vert.color[0] * af;
                c.col_sum[1] += vert.color[1] * af;
                c.col_sum[2] += vert.color[2] * af;
                c.ao_sum += vert.ao * af;
                c.weight_sum += area;
            }
            let (c0, c1, c2) = (
                local_to_cell[usize::from(i0)],
                local_to_cell[usize::from(i1)],
                local_to_cell[usize::from(i2)],
            );
            if c0 != c1 && c1 != c2 && c2 != c0 {
                let orig_n = Vec3::new(cross.x as f32, cross.y as f32, cross.z as f32);
                raw_tris.push(([c0, c1, c2], orig_n));
            }
        }
    }
    finalize_welded(&cells, raw_tris, cell)
}

pub fn cluster_weld_group(
    group_clusters: &[&MeshletData],
    hard_locked: &HashSet<u32>,
    target_tris: usize,
    target_verts: usize,
) -> SimplifiedGroup {
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    for m in group_clusters {
        for v in &m.vertices {
            bmin = bmin.min(v.position);
            bmax = bmax.max(v.position);
        }
    }
    let diag = max((bmax - bmin).length(), 1.0e-4);
    let mut cell = diag / 36.0;
    let mut best = SimplifiedGroup {
        vertices: Vec::new(),
        pos_ids: Vec::new(),
        triangles: Vec::new(),
        error: 0.0,
    };
    for _ in 0..18 {
        let (verts, pids, tris, err) =
            build_welded_at_cell(group_clusters, hard_locked, bmin, cell);
        if tris.is_empty() {
            break;
        }
        let ok = tris.len() <= target_tris && verts.len() <= target_verts;
        best = SimplifiedGroup {
            vertices: verts,
            pos_ids: pids,
            triangles: tris,
            error: err,
        };
        if ok {
            break;
        }
        cell *= 1.35;
    }
    best
}
