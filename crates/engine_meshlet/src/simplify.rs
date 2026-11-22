use std::cmp::Ordering;
use std::collections::BinaryHeap;

use engine_math::collections::{FixedState, HashMap, HashSet};
use engine_math::{clamp, max, min};
use glam::{DVec3, Vec2, Vec3};

use crate::bounds::bounding_sphere_points;
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
        if !det.is_finite() || det.abs() <= 1.0e-10 * scale.max(1.0e-12) {
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
    is_seam: Vec<bool>,
    is_crease: Vec<bool>,
    is_open: Vec<bool>,
    is_corner: Vec<bool>,
    uv_seam: Vec<bool>,
    alive_vert: Vec<bool>,
    ver: Vec<u32>,
    vert_tris: Vec<Vec<usize>>,
    tris: Vec<SimpTri>,
    live_tris: usize,
    live_verts: usize,
    max_error: f64,
    group_radius: f64,
}

struct QuadricAndLocks {
    vert_tris: Vec<Vec<usize>>,
    plane_q: Vec<Quadric>,
    boundary_q: Vec<Quadric>,
    hard_locked: Vec<bool>,
    is_seam: Vec<bool>,
    is_crease: Vec<bool>,
    is_open: Vec<bool>,
    is_corner: Vec<bool>,
    uv_seam: Vec<bool>,
}

struct BoundaryScan {
    is_open: Vec<bool>,
    is_crease: Vec<bool>,
    is_corner: Vec<bool>,
    first_dir: Vec<Option<DVec3>>,
    crease_dir: Vec<Option<DVec3>>,
    open_deg: Vec<u32>,
    crease_deg: Vec<u32>,
    boundary_q: Vec<Quadric>,
}

impl BoundaryScan {
    fn new(v_count: usize) -> Self {
        Self {
            is_open: vec![false; v_count],
            is_crease: vec![false; v_count],
            is_corner: vec![false; v_count],
            first_dir: vec![None; v_count],
            crease_dir: vec![None; v_count],
            open_deg: vec![0; v_count],
            crease_deg: vec![0; v_count],
            boundary_q: vec![Quadric::default(); v_count],
        }
    }

    fn record_crease(&mut self, u: usize, v: usize, positions: &[DVec3]) {
        self.is_crease[u] = true;
        self.is_crease[v] = true;
        self.crease_deg[u] += 1;
        self.crease_deg[v] += 1;
        if self.crease_deg[u] > 2 {
            self.is_corner[u] = true;
        }
        if self.crease_deg[v] > 2 {
            self.is_corner[v] = true;
        }
        let edge = positions[v] - positions[u];
        let elen = edge.length();
        if elen > 1.0e-12 {
            let dir = edge / elen;
            for idx in [u, v] {
                if let Some(prev) = self.crease_dir[idx] {
                    if prev.dot(dir).abs() < 0.35 {
                        self.is_corner[idx] = true;
                    }
                } else {
                    self.crease_dir[idx] = Some(dir);
                }
            }
        }
    }

    fn record_edge(&mut self, u: usize, v: usize, positions: &[DVec3], n: DVec3) {
        self.is_open[u] = true;
        self.is_open[v] = true;
        self.open_deg[u] += 1;
        self.open_deg[v] += 1;
        if self.open_deg[u] > 2 {
            self.is_corner[u] = true;
        }
        if self.open_deg[v] > 2 {
            self.is_corner[v] = true;
        }
        let edge = positions[v] - positions[u];
        let elen = edge.length();
        if elen > 1.0e-12 {
            let dir = edge / elen;
            for idx in [u, v] {
                if let Some(prev) = self.first_dir[idx] {
                    if prev.dot(dir).abs() < 0.85 {
                        self.is_corner[idx] = true;
                    }
                } else {
                    self.first_dir[idx] = Some(dir);
                }
            }
        }
        let bn = edge.cross(n);
        let bn_len = bn.length();
        if bn_len > 1.0e-16 {
            let b_unit = bn / bn_len;
            let bd = -b_unit.dot(positions[u]);
            let b_weight = (elen * 8.0).max(elen * elen * 16.0).max(1.0);
            let bq = Quadric::from_plane(b_unit, bd, b_weight);
            self.boundary_q[u].add(&bq);
            self.boundary_q[v].add(&bq);
        }
    }
}

fn build_quadrics_and_locks(
    positions: &[DVec3],
    global_pos: &[u32],
    tris: &[SimpTri],
    hard_locked_pos: &HashSet<u32>,
    seam_locked_pos: &HashSet<u32>,
) -> QuadricAndLocks {
    let v_count = positions.len();
    let mut edge_info: HashMap<(u32, u32), (u32, DVec3)> =
        HashMap::with_capacity_and_hasher(tris.len() * 3, FixedState::default());
    let mut vert_tris = vec![Vec::new(); v_count];
    let mut plane_q = vec![Quadric::default(); v_count];
    for (ti, tri) in tris.iter().enumerate() {
        let [a, b, c] = tri.v;
        vert_tris[a as usize].push(ti);
        vert_tris[b as usize].push(ti);
        vert_tris[c as usize].push(ti);
        let pa = positions[a as usize];
        let pb = positions[b as usize];
        let pc = positions[c as usize];
        let cross = (pb - pa).cross(pc - pa);
        let twice_area = cross.length();
        let n = if twice_area > 1.0e-16 {
            let unit = cross / twice_area;
            let d = -unit.dot(pa);
            let q = Quadric::from_plane(unit, d, 0.5 * twice_area);
            plane_q[a as usize].add(&q);
            plane_q[b as usize].add(&q);
            plane_q[c as usize].add(&q);
            unit
        } else {
            DVec3::ZERO
        };
        for &(u, v) in &[(a, b), (b, c), (c, a)] {
            let key = if u < v { (u, v) } else { (v, u) };
            edge_info
                .entry(key)
                .and_modify(|e| e.0 += 1)
                .or_insert((1, n));
        }
    }
    let mut scan = BoundaryScan::new(v_count);
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
            if let Some(&(cnt, first_n)) = edge_info.get(&key) {
                if cnt == 1 {
                    scan.record_edge(u as usize, v as usize, positions, n);
                } else if cnt == 2 && first_n.dot(n) < 0.25 {
                    scan.record_crease(u as usize, v as usize, positions);
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
    let is_seam: Vec<bool> = global_pos
        .iter()
        .map(|gp| seam_locked_pos.contains(gp))
        .collect();
    QuadricAndLocks {
        vert_tris,
        plane_q,
        boundary_q: scan.boundary_q,
        hard_locked,
        is_seam,
        is_crease: scan.is_crease,
        is_open: scan.is_open,
        is_corner: scan.is_corner,
        uv_seam,
    }
}

fn init_state(
    group_clusters: &[&MeshletData],
    hard_locked_pos: &HashSet<u32>,
    seam_locked_pos: &HashSet<u32>,
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
    let pts_f32: Vec<Vec3> = positions
        .iter()
        .map(|p| Vec3::new(p.x as f32, p.y as f32, p.z as f32))
        .collect();
    let group_radius = f64::from(bounding_sphere_points(&pts_f32)[3]).max(0.1);
    let ql = build_quadrics_and_locks(
        &positions,
        &global_pos,
        &tris,
        hard_locked_pos,
        seam_locked_pos,
    );
    let live_tris = tris.len();
    SimplifierState {
        positions,
        global_pos,
        plane_q: ql.plane_q,
        boundary_q: ql.boundary_q,
        hard_locked: ql.hard_locked,
        is_seam: ql.is_seam,
        is_crease: ql.is_crease,
        is_open: ql.is_open,
        is_corner: ql.is_corner,
        uv_seam: ql.uv_seam,
        alive_vert: vec![true; v_count],
        ver: vec![0; v_count],
        vert_tris: ql.vert_tris,
        tris,
        live_tris,
        live_verts: v_count,
        max_error: 0.0,
        group_radius,
    }
}

#[derive(Clone, Copy)]
struct PassPolicy {
    min_dot: f64,
    check_link: bool,
    check_vn: bool,
    max_bnd_sq: f64,
    max_plane_sq: f64,
    max_perp_disp: f64,
    max_corner_edge: f64,
}

fn shared_edge_tri_count(state: &SimplifierState, u: u32, v: u32) -> u32 {
    let mut count = 0_u32;
    for &ti in &state.vert_tris[u as usize] {
        let tri = &state.tris[ti];
        if tri.alive && (tri.v[0] == v || tri.v[1] == v || tri.v[2] == v) {
            count += 1;
        }
    }
    count
}

fn pick_collapse_target(
    state: &SimplifierState,
    u: u32,
    v: u32,
    total_q: &Quadric,
    policy: PassPolicy,
) -> Option<(u32, u32, DVec3)> {
    let lu = state.hard_locked[u as usize];
    let lv = state.hard_locked[v as usize];
    if lu && lv {
        return None;
    }
    let pu = state.positions[u as usize];
    let pv = state.positions[v as usize];
    let edge_len = (pv - pu).length();
    let cu_flag = state.is_corner[u as usize];
    let cv_flag = state.is_corner[v as usize];
    let smu = state.is_seam[u as usize];
    let smv = state.is_seam[v as usize];
    let cru = state.is_crease[u as usize];
    let crv = state.is_crease[v as usize];
    let ou = state.is_open[u as usize];
    let ov = state.is_open[v as usize];
    if lu {
        if (cv_flag && edge_len > policy.max_corner_edge) || (smv && !smu) || (crv && !cru) {
            return None;
        }
        return Some((u, v, pu));
    }
    if lv {
        if (cu_flag && edge_len > policy.max_corner_edge) || (smu && !smv) || (cru && !crv) {
            return None;
        }
        return Some((v, u, pv));
    }
    if ou && ov && shared_edge_tri_count(state, u, v) >= 2 {
        return None;
    }
    if cu_flag && cv_flag {
        if edge_len > policy.max_corner_edge {
            return None;
        }
        return if total_q.eval(pu) <= total_q.eval(pv) {
            Some((u, v, pu))
        } else {
            Some((v, u, pv))
        };
    }
    if cu_flag {
        return (!(smv && !smu) && !(crv && !cru)).then_some((u, v, pu));
    }
    if cv_flag {
        return (!(smu && !smv) && !(cru && !crv)).then_some((v, u, pv));
    }
    if smu && smv {
        return if total_q.eval(pu) <= total_q.eval(pv) {
            Some((u, v, pu))
        } else {
            Some((v, u, pv))
        };
    }
    if smu && !smv {
        return (!ov || ou).then_some((u, v, pu));
    }
    if smv && !smu {
        return (!ou || ov).then_some((v, u, pv));
    }
    if ou && ov {
        return if total_q.eval(pu) <= total_q.eval(pv) {
            Some((u, v, pu))
        } else {
            Some((v, u, pv))
        };
    }
    if ou && !ov {
        return (!crv).then_some((u, v, pu));
    }
    if ov && !ou {
        return (!cru).then_some((v, u, pv));
    }
    if cru && crv {
        return if total_q.eval(pu) <= total_q.eval(pv) {
            Some((u, v, pu))
        } else {
            Some((v, u, pv))
        };
    }
    if cru != crv {
        return Some(if cru { (u, v, pu) } else { (v, u, pv) });
    }
    let su = state.uv_seam[u as usize];
    let sv = state.uv_seam[v as usize];
    if su != sv {
        return Some(if su { (u, v, pu) } else { (v, u, pv) });
    }
    let mid = (pu + pv) * 0.5;
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
        && (opt - mid).length() <= edge_len * 0.75 + 1.0e-6
        && total_q.eval(opt) < best_c
    {
        best_p = opt;
    }
    Some((u, v, best_p))
}

fn edge_attribute_metrics(state: &SimplifierState, u: u32, v: u32) -> (f32, f32) {
    let mut max_duv_sq = 0.0_f32;
    let mut min_ndot = 1.0_f32;
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
            min_ndot = min(min_ndot, tri.attrs[a].normal.dot(tri.attrs[b].normal));
        }
    }
    (max_duv_sq, min_ndot)
}

fn evaluate_edge(
    state: &SimplifierState,
    u: u32,
    v: u32,
    policy: PassPolicy,
) -> Option<CollapseEntry> {
    let pq = state.plane_q[u as usize].sum(&state.plane_q[v as usize]);
    let bq = state.boundary_q[u as usize].sum(&state.boundary_q[v as usize]);
    let total_q = pq.sum(&bq);
    let (keep, remove, target_pos) = pick_collapse_target(state, u, v, &total_q, policy)?;
    let pu = state.positions[u as usize];
    let pv = state.positions[v as usize];
    let edge_len = (pv - pu).length();
    let plane_sq = pq.eval(target_pos) / pq.weight.max(1.0e-12);
    let u_sq = state.plane_q[u as usize].eval(target_pos)
        / state.plane_q[u as usize].weight.max(1.0e-12);
    let v_sq = state.plane_q[v as usize].eval(target_pos)
        / state.plane_q[v as usize].weight.max(1.0e-12);
    if plane_sq > policy.max_plane_sq || u_sq.max(v_sq) > policy.max_plane_sq * 2.0 {
        return None;
    }
    let open_moved = (state.is_open[u as usize] && (target_pos - pu).length_squared() > 1.0e-18)
        || (state.is_open[v as usize] && (target_pos - pv).length_squared() > 1.0e-18);
    let seam_moved = (state.is_seam[u as usize] && (target_pos - pu).length_squared() > 1.0e-18)
        || (state.is_seam[v as usize] && (target_pos - pv).length_squared() > 1.0e-18);
    let bnd_err_sq = if open_moved {
        let eval_t = bq.eval(target_pos);
        let base_err = state.boundary_q[u as usize].eval(pu) + state.boundary_q[v as usize].eval(pv);
        let rms_sq = eval_t / bq.weight.max(1.0);
        let inc_sq = (eval_t - base_err).max(0.0) / bq.weight.max(1.0);
        inc_sq.max(rms_sq)
    } else {
        0.0
    };
    if open_moved && bnd_err_sq > policy.max_bnd_sq {
        return None;
    }
    if seam_moved && !open_moved && plane_sq > policy.max_bnd_sq {
        return None;
    }
    let mut cost = pq.eval(target_pos) + bnd_err_sq * 64.0;
    let mut geom_err = plane_sq.max(bnd_err_sq).sqrt();
    let (max_duv_sq, min_ndot) = edge_attribute_metrics(state, u, v);
    if min_ndot < 0.7 {
        let crease = f64::from(1.0 - max(min_ndot, -1.0));
        cost += edge_len * edge_len * crease * 6.0;
        geom_err = geom_err.max(0.08 * crease * edge_len);
    }
    let seam_edge = state.uv_seam[u as usize] || state.uv_seam[v as usize];
    if seam_edge {
        geom_err = geom_err.max(0.18 * edge_len);
        cost += edge_len * edge_len * 20.0;
    } else if max_duv_sq > 0.04 {
        geom_err = geom_err.max(0.10 * edge_len);
        cost += edge_len * edge_len * 6.0;
    }
    if state.is_open[u as usize] && state.is_open[v as usize] {
        cost += edge_len * edge_len * 1.5;
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

fn push_all_edges(
    state: &SimplifierState,
    policy: PassPolicy,
    heap: &mut BinaryHeap<CollapseEntry>,
) {
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
                && let Some(entry) = evaluate_edge(state, key.0, key.1, policy)
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

fn tri_max_edge(p: [DVec3; 3]) -> f64 {
    let e0 = (p[1] - p[0]).length();
    let e1 = (p[2] - p[1]).length();
    let e2 = (p[0] - p[2]).length();
    e0.max(e1).max(e2)
}

fn incident_geom_stats(state: &SimplifierState, keep: u32, remove: u32) -> (f64, f64) {
    let mut sum_area = 0.0_f64;
    let mut max_edge = 0.0_f64;
    for &vert in &[keep, remove] {
        for &ti in &state.vert_tris[vert as usize] {
            let tri = &state.tris[ti];
            if !tri.alive {
                continue;
            }
            let shared = tri.v.contains(&keep) && tri.v.contains(&remove);
            if shared && vert == remove {
                continue;
            }
            let p = [
                state.positions[tri.v[0] as usize],
                state.positions[tri.v[1] as usize],
                state.positions[tri.v[2] as usize],
            ];
            sum_area += 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length();
            max_edge = max_edge.max(tri_max_edge(p));
        }
    }
    (sum_area, max_edge)
}

fn collapse_is_valid(
    state: &SimplifierState,
    keep: u32,
    remove: u32,
    target: DVec3,
    policy: PassPolicy,
) -> bool {
    if policy.check_link && !link_condition_ok(state, keep, remove) {
        return false;
    }
    let (sum_old_area, old_max_edge) = incident_geom_stats(state, keep, remove);
    let max_allowed_edge = old_max_edge * 2.25 + 1.0e-4;
    let mut sum_new_area = 0.0_f64;
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
            let moved_old = if tri.v[0] == keep || tri.v[0] == remove {
                p_old[0]
            } else if tri.v[1] == keep || tri.v[1] == remove {
                p_old[1]
            } else {
                p_old[2]
            };
            if (target - moved_old).dot(tri.ref_n).abs() > policy.max_perp_disp {
                return false;
            }
            let old_area = 0.5 * old_len;
            let new_area = 0.5 * new_len;
            sum_new_area += new_area;
            let old_e = tri_max_edge(p_old);
            let new_e = tri_max_edge(p_new);
            if new_e > max_allowed_edge {
                return false;
            }
            let old_ratio = (old_e * old_e) / old_area.max(1.0e-18);
            let new_ratio = (new_e * new_e) / new_area.max(1.0e-18);
            if new_ratio > 24.0 && (new_area > old_area * 1.05 || new_ratio > old_ratio * 1.5) {
                return false;
            }
            if new_ratio > 48.0 && new_ratio > old_ratio * 1.05 {
                return false;
            }
            if old_len > 1.0e-14 && n_old.dot(n_new) < policy.min_dot * old_len * new_len {
                return false;
            }
            if tri.ref_n.dot(n_new) < policy.min_dot * new_len {
                return false;
            }
            if policy.check_vn {
                let vn = (tri.attrs[0].normal + tri.attrs[1].normal + tri.attrs[2].normal).as_dvec3();
                let vn_len = vn.length();
                if vn_len > 1.0e-6 && vn.dot(n_new) < (policy.min_dot * 0.75) * vn_len * new_len {
                    return false;
                }
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
    if sum_new_area > sum_old_area * 1.25 + 1.0e-8 {
        return false;
    }
    has_shared && !surviving_faces.is_empty()
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

fn cleanup_orphaned_verts(state: &mut SimplifierState, keep: u32, killed_opps: &[u32]) {
    state.vert_tris[keep as usize].retain(|&ti| state.tris[ti].alive);
    if state.vert_tris[keep as usize].is_empty() && state.alive_vert[keep as usize] {
        state.alive_vert[keep as usize] = false;
        state.live_verts = state.live_verts.saturating_sub(1);
    }
    for &ov in killed_opps {
        let oi = ov as usize;
        state.vert_tris[oi].retain(|&ti| state.tris[ti].alive);
        if state.vert_tris[oi].is_empty() && state.alive_vert[oi] {
            state.alive_vert[oi] = false;
            state.live_verts = state.live_verts.saturating_sub(1);
        }
    }
}

fn push_keep_neighbor_edges(
    state: &SimplifierState,
    keep: u32,
    policy: PassPolicy,
    heap: &mut BinaryHeap<CollapseEntry>,
) {
    let mut neighbors = Vec::new();
    for &ti in &state.vert_tris[keep as usize] {
        for &v in &state.tris[ti].v {
            if v != keep && state.alive_vert[v as usize] && !neighbors.contains(&v) {
                neighbors.push(v);
            }
        }
    }
    for nb in neighbors {
        if let Some(next_entry) = evaluate_edge(state, keep, nb, policy) {
            heap.push(next_entry);
        }
    }
}

fn apply_collapse(
    state: &mut SimplifierState,
    entry: CollapseEntry,
    policy: PassPolicy,
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
    let mut killed_opps: Vec<u32> = Vec::new();
    for &ti in &state.vert_tris[keep as usize] {
        let tri = &mut state.tris[ti];
        if !tri.alive {
            continue;
        }
        let mut ki = None;
        let mut ri = None;
        let mut opp = None;
        for slot in 0..3 {
            if tri.v[slot] == keep {
                ki = Some(slot);
            } else if tri.v[slot] == remove {
                ri = Some(slot);
            } else {
                opp = Some(tri.v[slot]);
            }
        }
        if let (Some(ks), Some(rs)) = (ki, ri) {
            if ref_keep_attr.is_none() {
                ref_keep_attr = Some(tri.attrs[ks]);
                ref_rem_attr = Some(tri.attrs[rs]);
            }
            if let Some(ov) = opp {
                killed_opps.push(ov);
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
    state.hard_locked[keep as usize] |= state.hard_locked[remove as usize];
    state.is_seam[keep as usize] |= state.is_seam[remove as usize];
    state.is_crease[keep as usize] |= state.is_crease[remove as usize];
    state.is_open[keep as usize] |= state.is_open[remove as usize];
    state.is_corner[keep as usize] |= state.is_corner[remove as usize];
    state.uv_seam[keep as usize] |= state.uv_seam[remove as usize];
    state.alive_vert[remove as usize] = false;
    state.live_verts = state.live_verts.saturating_sub(1);
    state.ver[keep as usize] = state.ver[keep as usize].wrapping_add(1);
    if entry.geom_err > state.max_error {
        state.max_error = entry.geom_err;
    }
    cleanup_orphaned_verts(state, keep, &killed_opps);
    push_keep_neighbor_edges(state, keep, policy, heap);
}

fn run_collapse_pass(
    state: &mut SimplifierState,
    target_tris: usize,
    target_verts: usize,
    policy: PassPolicy,
) {
    let mut heap = BinaryHeap::new();
    push_all_edges(state, policy, &mut heap);
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
        if !collapse_is_valid(state, entry.keep, entry.remove, entry.target_pos, policy) {
            continue;
        }
        apply_collapse(state, entry, policy, &mut heap);
    }
}

struct IslandComponent {
    tris: Vec<usize>,
    verts: Vec<u32>,
    area: f64,
    radius: f64,
}

fn collect_prunable_islands(
    state: &SimplifierState,
    relax_locks: bool,
) -> Vec<IslandComponent> {
    let mut visited = vec![false; state.tris.len()];
    let mut islands = Vec::new();
    let mut stack = Vec::new();
    let mut vert_mark = vec![0_u32; state.positions.len()];
    let mut stamp = 0_u32;
    for start in 0..state.tris.len() {
        if visited[start] || !state.tris[start].alive {
            continue;
        }
        stamp = stamp.wrapping_add(1);
        visited[start] = true;
        stack.clear();
        stack.push(start);
        let mut comp_tris = Vec::new();
        let mut comp_verts = Vec::new();
        let mut comp_pts = Vec::new();
        let mut has_hard_lock = false;
        let mut has_seam = false;
        let mut has_open = false;
        let mut area = 0.0_f64;
        while let Some(ti) = stack.pop() {
            comp_tris.push(ti);
            let [a, b, c] = state.tris[ti].v;
            let pa = state.positions[a as usize];
            let pb = state.positions[b as usize];
            let pc = state.positions[c as usize];
            area += 0.5 * (pb - pa).cross(pc - pa).length();
            for v in [a, b, c] {
                let vi = v as usize;
                if vert_mark[vi] != stamp {
                    vert_mark[vi] = stamp;
                    if state.hard_locked[vi] {
                        has_hard_lock = true;
                    }
                    if state.is_seam[vi] {
                        has_seam = true;
                    }
                    if state.is_open[vi] {
                        has_open = true;
                    }
                    comp_verts.push(v);
                    let p = state.positions[vi];
                    comp_pts.push(Vec3::new(p.x as f32, p.y as f32, p.z as f32));
                    for &nb in &state.vert_tris[vi] {
                        if !visited[nb] && state.tris[nb].alive {
                            visited[nb] = true;
                            stack.push(nb);
                        }
                    }
                }
            }
        }
        if !has_seam && !has_hard_lock {
            let bs = bounding_sphere_points(&comp_pts);
            let radius = f64::from(bs[3]);
            let mean_tri_area = area / (comp_tris.len().max(1) as f64);
            let is_leaf_card = has_open && radius <= 0.28 && area <= 0.10;
            let is_thin_wisp = relax_locks
                && radius > 0.28
                && area <= 0.25
                && mean_tri_area <= 0.005
                && (area / radius) <= 0.25;
            if is_leaf_card || is_thin_wisp {
                islands.push(IslandComponent {
                    tris: comp_tris,
                    verts: comp_verts,
                    area,
                    radius,
                });
            }
        }
    }
    islands.sort_unstable_by(|a, b| {
        a.area
            .total_cmp(&b.area)
            .then_with(|| a.radius.total_cmp(&b.radius))
            .then_with(|| a.tris.len().cmp(&b.tris.len()))
    });
    islands
}

fn prune_small_islands(
    state: &mut SimplifierState,
    target_tris: usize,
    target_verts: usize,
    relax_locks: bool,
) {
    let islands = collect_prunable_islands(state, relax_locks);
    if islands.len() <= 1 {
        return;
    }
    for island in islands {
        if state.live_tris <= target_tris && state.live_verts <= target_verts {
            break;
        }
        if island.tris.len() >= state.live_tris {
            break;
        }
        for ti in island.tris {
            if state.tris[ti].alive {
                state.tris[ti].alive = false;
                state.live_tris = state.live_tris.saturating_sub(1);
            }
        }
        for v in island.verts {
            let vi = v as usize;
            state.vert_tris[vi].retain(|&ti| state.tris[ti].alive);
            if state.vert_tris[vi].is_empty() && state.alive_vert[vi] {
                state.alive_vert[vi] = false;
                state.live_verts = state.live_verts.saturating_sub(1);
            }
        }
        let err = island.radius.max(island.area.sqrt());
        if err > state.max_error {
            state.max_error = err;
        }
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
    seam_locked_pos: &HashSet<u32>,
    relax_seams: bool,
    target_tris: usize,
    target_verts: usize,
) -> SimplifiedGroup {
    let mut state = init_state(group_clusters, hard_locked_pos, seam_locked_pos);
    let clamped_tris = target_tris.max(1);
    let clamped_verts = target_verts.max(3);
    let p1_rms = (state.group_radius * 0.01).clamp(0.01, 0.03);
    let p2_rms = (state.group_radius * 0.025).clamp(0.025, 0.05);
    let p3_rms = (state.group_radius * 0.05).clamp(0.04, 0.06);
    run_collapse_pass(
        &mut state,
        clamped_tris,
        clamped_verts,
        PassPolicy {
            min_dot: 0.30,
            check_link: true,
            check_vn: true,
            max_bnd_sq: 0.0001,
            max_plane_sq: p1_rms * p1_rms,
            max_perp_disp: p1_rms * 1.5,
            max_corner_edge: 0.05,
        },
    );
    if state.live_tris > clamped_tris || state.live_verts > clamped_verts {
        run_collapse_pass(
            &mut state,
            clamped_tris,
            clamped_verts,
            PassPolicy {
                min_dot: 0.08,
                check_link: true,
                check_vn: false,
                max_bnd_sq: 0.0009,
                max_plane_sq: p2_rms * p2_rms,
                max_perp_disp: p2_rms * 1.5,
                max_corner_edge: 0.12,
            },
        );
    }
    if state.live_tris > clamped_tris || state.live_verts > clamped_verts {
        prune_small_islands(&mut state, clamped_tris, clamped_verts, relax_seams);
    }
    if state.live_tris > clamped_tris || state.live_verts > clamped_verts {
        run_collapse_pass(
            &mut state,
            clamped_tris,
            clamped_verts,
            PassPolicy {
                min_dot: 0.05,
                check_link: true,
                check_vn: false,
                max_bnd_sq: 0.0025,
                max_plane_sq: p3_rms * p3_rms,
                max_perp_disp: p3_rms * 1.5,
                max_corner_edge: 0.28,
            },
        );
    }
    export_simplified(&state)
}
