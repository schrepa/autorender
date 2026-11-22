use std::f32::consts::{PI, TAU};
use std::path::Path;
use std::time::Instant;

use cook::Refusal;
use engine_math::{clamp, cos, max, min, sin};
use glam::Vec3;
use serde::Serialize;

use crate::gltf_load::load_gltf_mesh;
use crate::pbr_pack::PBR_MAGIC;

pub const GI_MAGIC: [u8; 4] = *b"GIP1";
pub const GI_VERSION: u32 = 1;
pub const GRID_X: usize = 16;
pub const GRID_Y: usize = 8;
pub const GRID_Z: usize = 10;
pub const GRID_MIN: [f32; 3] = [-17.0, 0.35, -8.8];
pub const GRID_MAX: [f32; 3] = [19.0, 14.5, 9.6];

const CHUNK_NAMES: [&str; 8] = [
    "sponza_lower_w.glb",
    "sponza_lower_e.glb",
    "sponza_upper_w.glb",
    "sponza_upper_e.glb",
    "sponza_roof.glb",
    "sponza_curtains.glb",
    "sponza_ivy_low.glb",
    "sponza_ivy_high.glb",
];

#[derive(Clone, Debug, Serialize)]
pub struct BakeGiOutput {
    pub status: &'static str,
    pub out_path: String,
    pub probe_count: u32,
    pub valid_probes: u32,
    pub triangle_count: u32,
    pub out_bytes: usize,
    pub elapsed_ms: f64,
}

#[derive(Clone, Copy)]
struct MatShade {
    albedo: Vec3,
    emissive: Vec3,
    flags: u32,
}

#[derive(Clone, Copy)]
struct RayTri {
    v0: Vec3,
    e1: Vec3,
    e2: Vec3,
    normal: Vec3,
    centroid: Vec3,
    bmin: Vec3,
    bmax: Vec3,
    mat: u8,
}

#[derive(Clone, Copy)]
struct BvhNode {
    bmin: Vec3,
    bmax: Vec3,
    left_or_first: u32,
    right: u32,
    count: u32,
}

struct SceneBvh {
    tris: Vec<RayTri>,
    nodes: Vec<BvhNode>,
    mats: Vec<MatShade>,
}

struct RayHit {
    t: f32,
    normal: Vec3,
    mat: u8,
    backface: bool,
}

fn read_u32_le(bytes: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}

fn read_f32_le(bytes: &[u8], off: usize) -> f32 {
    f32::from_bits(read_u32_le(bytes, off))
}

fn load_pbr_materials(path: &Path) -> Result<Vec<MatShade>, Refusal> {
    let bytes = std::fs::read(path)
        .map_err(|e| Refusal::new("ReadFailed", &path.display().to_string(), format!("{e}")))?;
    if bytes.len() < 32 || bytes[0..4] != PBR_MAGIC {
        return Err(Refusal::new(
            "InvalidPbr",
            &path.display().to_string(),
            "expected SPB2 header",
        ));
    }
    let mat_count = read_u32_le(&bytes, 8) as usize;
    let mut mats = Vec::with_capacity(mat_count);
    for i in 0..mat_count {
        let base = 32 + i * 48;
        if base + 48 > bytes.len() {
            break;
        }
        let albedo = Vec3::new(
            read_f32_le(&bytes, base),
            read_f32_le(&bytes, base + 4),
            read_f32_le(&bytes, base + 8),
        );
        let emissive = Vec3::new(
            read_f32_le(&bytes, base + 16),
            read_f32_le(&bytes, base + 20),
            read_f32_le(&bytes, base + 24),
        );
        let flags = read_u32_le(&bytes, base + 36);
        mats.push(MatShade {
            albedo,
            emissive,
            flags,
        });
    }
    Ok(mats)
}

fn load_scene_triangles(glbs_dir: &Path) -> Result<Vec<RayTri>, Refusal> {
    let mut tris = Vec::new();
    for name in CHUNK_NAMES {
        let path = glbs_dir.join(name);
        if !path.is_file() {
            return Err(Refusal::new(
                "MissingChunk",
                &path.display().to_string(),
                "expected Sponza GLB chunk",
            ));
        }
        let src = load_gltf_mesh(&path, None)?;
        for (ti, &[a, b, c]) in src.indices.as_chunks::<3>().0.iter().enumerate() {
            let v0 = src.positions[a as usize];
            let v1 = src.positions[b as usize];
            let v2 = src.positions[c as usize];
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let cross = e1.cross(e2);
            let len_sq = cross.length_squared();
            if !len_sq.is_finite() || len_sq <= 1.0e-12 {
                continue;
            }
            let normal = cross / len_sq.sqrt();
            let centroid = (v0 + v1 + v2) * (1.0 / 3.0);
            let bmin = v0.min(v1).min(v2);
            let bmax = v0.max(v1).max(v2);
            let mat = src.material_ids.get(ti).copied().unwrap_or(0);
            tris.push(RayTri {
                v0,
                e1,
                e2,
                normal,
                centroid,
                bmin,
                bmax,
                mat,
            });
        }
    }
    Ok(tris)
}

fn build_bvh_node(tris: &mut [RayTri], first: usize, count: usize, nodes: &mut Vec<BvhNode>) -> u32 {
    let idx = nodes.len() as u32;
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    let mut cmin = Vec3::splat(f32::INFINITY);
    let mut cmax = Vec3::splat(f32::NEG_INFINITY);
    for t in &tris[first..first + count] {
        bmin = bmin.min(t.bmin);
        bmax = bmax.max(t.bmax);
        cmin = cmin.min(t.centroid);
        cmax = cmax.max(t.centroid);
    }
    nodes.push(BvhNode {
        bmin,
        bmax,
        left_or_first: first as u32,
        right: 0,
        count: count as u32,
    });
    if count <= 8 {
        return idx;
    }
    let ext = cmax - cmin;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    if ext[axis] <= 1.0e-5 {
        return idx;
    }
    let split_pos = 0.5 * (cmin[axis] + cmax[axis]);
    let mut i = first;
    let mut j = first + count;
    while i < j {
        if tris[i].centroid[axis] < split_pos {
            i += 1;
        } else {
            j -= 1;
            tris.swap(i, j);
        }
    }
    let left_count = i - first;
    if left_count == 0 || left_count == count {
        return idx;
    }
    let left_idx = build_bvh_node(tris, first, left_count, nodes);
    let right_idx = build_bvh_node(tris, i, count - left_count, nodes);
    nodes[idx as usize].left_or_first = left_idx;
    nodes[idx as usize].right = right_idx;
    nodes[idx as usize].count = 0;
    idx
}

fn intersect_aabb(origin: Vec3, inv_dir: Vec3, bmin: Vec3, bmax: Vec3, t_max: f32) -> f32 {
    let t0 = (bmin - origin) * inv_dir;
    let t1 = (bmax - origin) * inv_dir;
    let tmin_v = t0.min(t1);
    let tmax_v = t0.max(t1);
    let t_enter = max(max(tmin_v.x, tmin_v.y), max(tmin_v.z, 0.0));
    let t_exit = min(min(tmax_v.x, tmax_v.y), min(tmax_v.z, t_max));
    if t_enter <= t_exit { t_enter } else { f32::INFINITY }
}

fn intersect_tri(origin: Vec3, dir: Vec3, tri: &RayTri, t_max: f32) -> Option<(f32, bool)> {
    let h = dir.cross(tri.e2);
    let a = tri.e1.dot(h);
    if a.abs() < 1.0e-7 {
        return None;
    }
    let f = 1.0 / a;
    let s = origin - tri.v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(tri.e1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = f * tri.e2.dot(q);
    if t > 1.0e-4 && t < t_max {
        Some((t, a < 0.0))
    } else {
        None
    }
}

impl SceneBvh {
    fn trace(&self, origin: Vec3, dir: Vec3, t_limit: f32) -> Option<RayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv_dir = Vec3::new(
            if dir.x.abs() > 1.0e-8 { 1.0 / dir.x } else { 1.0e8 },
            if dir.y.abs() > 1.0e-8 { 1.0 / dir.y } else { 1.0e8 },
            if dir.z.abs() > 1.0e-8 { 1.0 / dir.z } else { 1.0e8 },
        );
        let mut best_t = t_limit;
        let mut hit: Option<RayHit> = None;
        let mut stack = [0_u32; 64];
        let mut sp = 0_usize;
        if intersect_aabb(origin, inv_dir, self.nodes[0].bmin, self.nodes[0].bmax, best_t).is_finite()
        {
            stack[0] = 0;
            sp = 1;
        }
        while sp > 0 {
            sp -= 1;
            let node = self.nodes[stack[sp] as usize];
            if node.count > 0 {
                let first = node.left_or_first as usize;
                let end = first + node.count as usize;
                for tri in &self.tris[first..end] {
                    if let Some((t, backface)) = intersect_tri(origin, dir, tri, best_t) {
                        best_t = t;
                        hit = Some(RayHit {
                            t,
                            normal: if backface { -tri.normal } else { tri.normal },
                            mat: tri.mat,
                            backface,
                        });
                    }
                }
            } else {
                let left = node.left_or_first;
                let right = node.right;
                let tl = intersect_aabb(origin, inv_dir, self.nodes[left as usize].bmin, self.nodes[left as usize].bmax, best_t);
                let tr = intersect_aabb(origin, inv_dir, self.nodes[right as usize].bmin, self.nodes[right as usize].bmax, best_t);
                if tl < tr {
                    if tr < best_t && sp < 63 {
                        stack[sp] = right;
                        sp += 1;
                    }
                    if tl < best_t && sp < 63 {
                        stack[sp] = left;
                        sp += 1;
                    }
                } else {
                    if tl < best_t && sp < 63 {
                        stack[sp] = left;
                        sp += 1;
                    }
                    if tr < best_t && sp < 63 {
                        stack[sp] = right;
                        sp += 1;
                    }
                }
            }
        }
        hit
    }

    fn occluded(&self, origin: Vec3, dir: Vec3, t_limit: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv_dir = Vec3::new(
            if dir.x.abs() > 1.0e-8 { 1.0 / dir.x } else { 1.0e8 },
            if dir.y.abs() > 1.0e-8 { 1.0 / dir.y } else { 1.0e8 },
            if dir.z.abs() > 1.0e-8 { 1.0 / dir.z } else { 1.0e8 },
        );
        let mut stack = [0_u32; 64];
        let mut sp = 0_usize;
        if intersect_aabb(origin, inv_dir, self.nodes[0].bmin, self.nodes[0].bmax, t_limit).is_finite()
        {
            stack[0] = 0;
            sp = 1;
        }
        while sp > 0 {
            sp -= 1;
            let node = self.nodes[stack[sp] as usize];
            if node.count > 0 {
                let first = node.left_or_first as usize;
                let end = first + node.count as usize;
                for tri in &self.tris[first..end] {
                    if intersect_tri(origin, dir, tri, t_limit).is_some() {
                        return true;
                    }
                }
            } else {
                let left = node.left_or_first;
                let right = node.right;
                if intersect_aabb(origin, inv_dir, self.nodes[left as usize].bmin, self.nodes[left as usize].bmax, t_limit) < t_limit
                    && sp < 63
                {
                    stack[sp] = left;
                    sp += 1;
                }
                if intersect_aabb(origin, inv_dir, self.nodes[right as usize].bmin, self.nodes[right as usize].bmax, t_limit) < t_limit
                    && sp < 63
                {
                    stack[sp] = right;
                    sp += 1;
                }
            }
        }
        false
    }
}

fn eval_sky(dir: Vec3, sun_dir: Vec3) -> Vec3 {
    let up_t = clamp(dir.y, 0.0, 1.0).powf(0.45);
    let d_xz = Vec3::new(dir.x, 0.0, dir.z).normalize_or_zero();
    let s_xz = Vec3::new(sun_dir.x, 0.0, sun_dir.z).normalize_or_zero();
    let sun_az = clamp(d_xz.dot(s_xz) * 0.5 + 0.5, 0.0, 1.0);
    let horiz = Vec3::new(0.64, 0.76, 0.92).lerp(Vec3::new(0.98, 0.90, 0.76), sun_az.powf(1.4));
    let mut col = horiz.lerp(Vec3::new(0.16, 0.38, 0.82), up_t) * 1.45;
    if dir.y < 0.0 {
        let fade = clamp(-dir.y * 4.5, 0.0, 1.0);
        col = (horiz * 1.1).lerp(Vec3::new(0.28, 0.24, 0.19), fade);
    }
    col
}

fn fibonacci_dir(index: usize, count: usize) -> Vec3 {
    let golden = 2.399_963_2_f32;
    let y = 1.0 - (2.0 * (index as f32 + 0.5)) / (count as f32);
    let r = max(0.0, 1.0 - y * y).sqrt();
    let phi = (index as f32) * golden;
    Vec3::new(cos(phi) * r, y, sin(phi) * r)
}

fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    x
}

fn cosine_hemisphere(n: Vec3, seed: u32) -> Vec3 {
    let u1 = (hash_u32(seed) as f32) / (u32::MAX as f32);
    let u2 = (hash_u32(seed ^ 0x9E37_79B9) as f32) / (u32::MAX as f32);
    let r = u1.sqrt();
    let theta = TAU * u2;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = max(0.0, 1.0 - u1).sqrt();
    let up = if n.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let t = up.cross(n).normalize();
    let b = n.cross(t);
    (t * x + b * y + n * z).normalize()
}

fn backface_ratio(scene: &SceneBvh, pos: Vec3) -> f32 {
    let n = 48_usize;
    let mut back = 0_u32;
    for i in 0..n {
        let d = fibonacci_dir(i, n);
        if let Some(hit) = scene.trace(pos, d, 60.0) {
            let flags = scene.mats.get(usize::from(hit.mat)).map_or(0, |m| m.flags);
            if hit.backface && (flags & 1) == 0 {
                back += 1;
            }
        }
    }
    (back as f32) / (n as f32)
}

fn nudge_probe(scene: &SceneBvh, base: Vec3) -> Option<Vec3> {
    let mut best_p = base;
    let mut best_ratio = backface_ratio(scene, base);
    if best_ratio <= 0.15 {
        return Some(base);
    }
    let offsets = [
        Vec3::new(0.4, 0.0, 0.0),
        Vec3::new(-0.4, 0.0, 0.0),
        Vec3::new(0.0, 0.4, 0.0),
        Vec3::new(0.0, -0.3, 0.0),
        Vec3::new(0.0, 0.0, 0.4),
        Vec3::new(0.0, 0.0, -0.4),
        Vec3::new(0.75, 0.35, 0.0),
        Vec3::new(-0.75, 0.35, 0.0),
        Vec3::new(0.0, 0.35, 0.75),
        Vec3::new(0.0, 0.35, -0.75),
    ];
    for off in offsets {
        let cand = base + off;
        let ratio = backface_ratio(scene, cand);
        if ratio < best_ratio {
            best_ratio = ratio;
            best_p = cand;
            if best_ratio <= 0.12 {
                break;
            }
        }
    }
    (best_ratio <= 0.30).then_some(best_p)
}

fn trace_radiance(scene: &SceneBvh, pos: Vec3, dir: Vec3, sun_dir: Vec3, seed: u32) -> Vec3 {
    let sun_irrad = Vec3::new(1.00, 0.91, 0.76) * 11.5;
    let mut ro = pos;
    let mut rd = dir;
    let mut throughput = Vec3::ONE;
    let mut total = Vec3::ZERO;
    for bounce in 0..3_u32 {
        let Some(hit) = scene.trace(ro, rd, 80.0) else {
            total += throughput * eval_sky(rd, sun_dir);
            break;
        };
        let m = scene.mats.get(usize::from(hit.mat)).copied().unwrap_or(MatShade {
            albedo: Vec3::splat(0.5),
            emissive: Vec3::ZERO,
            flags: 0,
        });
        total += throughput * m.emissive;
        let hit_pos = ro + rd * hit.t + hit.normal * 0.015;
        let n_dot_l = max(hit.normal.dot(sun_dir), 0.0);
        let trans = if (m.flags & (2 | 4)) != 0 {
            max((-hit.normal).dot(sun_dir), 0.0) * 0.25
        } else {
            0.0
        };
        if (n_dot_l > 0.001 || trans > 0.001) && !scene.occluded(hit_pos, sun_dir, 80.0) {
            total += throughput * (m.albedo / PI) * sun_irrad * (n_dot_l + trans);
        }
        throughput *= m.albedo;
        if bounce == 2 || throughput.max_element() < 0.02 {
            break;
        }
        ro = hit_pos;
        rd = cosine_hemisphere(hit.normal, seed.wrapping_add(bounce * 7919));
    }
    total.min(Vec3::splat(16.0))
}

#[derive(Clone, Copy, Default)]
struct ProbeSh {
    r: [f32; 4],
    g: [f32; 4],
    b: [f32; 4],
    valid: bool,
}

fn bake_single_probe(scene: &SceneBvh, idx: usize, pos: Vec3, sun_dir: Vec3) -> ProbeSh {
    let Some(sample_pos) = nudge_probe(scene, pos) else {
        return ProbeSh::default();
    };
    let rays = 128_usize;
    let inv_n = 1.0 / (rays as f32);
    let mut r = [0.0_f32; 4];
    let mut g = [0.0_f32; 4];
    let mut b = [0.0_f32; 4];
    for ri in 0..rays {
        let dir = fibonacci_dir(ri, rays);
        let seed = ((idx as u32).wrapping_mul(1973)) ^ ((ri as u32).wrapping_mul(9277));
        let rad = trace_radiance(scene, sample_pos, dir, sun_dir, seed);
        r[0] += rad.x * inv_n;
        r[1] += rad.x * dir.x * inv_n;
        r[2] += rad.x * dir.y * inv_n;
        r[3] += rad.x * dir.z * inv_n;
        g[0] += rad.y * inv_n;
        g[1] += rad.y * dir.x * inv_n;
        g[2] += rad.y * dir.y * inv_n;
        g[3] += rad.y * dir.z * inv_n;
        b[0] += rad.z * inv_n;
        b[1] += rad.z * dir.x * inv_n;
        b[2] += rad.z * dir.y * inv_n;
        b[3] += rad.z * dir.z * inv_n;
    }
    ProbeSh {
        r,
        g,
        b,
        valid: true,
    }
}

fn fill_invalid_probes(probes: &mut [ProbeSh]) {
    let idx_of = |x: usize, y: usize, z: usize| (z * GRID_Y + y) * GRID_X + x;
    for _ in 0..6 {
        let prev = probes.to_vec();
        let mut changed = false;
        for z in 0..GRID_Z {
            for y in 0..GRID_Y {
                for x in 0..GRID_X {
                    let i = idx_of(x, y, z);
                    if prev[i].valid {
                        continue;
                    }
                    let mut acc = ProbeSh::default();
                    let mut count = 0_u32;
                    let neighbors = [
                        (x.wrapping_sub(1), y, z),
                        (x + 1, y, z),
                        (x, y.wrapping_sub(1), z),
                        (x, y + 1, z),
                        (x, y, z.wrapping_sub(1)),
                        (x, y, z + 1),
                    ];
                    for (nx, ny, nz) in neighbors {
                        if nx < GRID_X && ny < GRID_Y && nz < GRID_Z {
                            let nb = prev[idx_of(nx, ny, nz)];
                            if nb.valid {
                                for c in 0..4 {
                                    acc.r[c] += nb.r[c];
                                    acc.g[c] += nb.g[c];
                                    acc.b[c] += nb.b[c];
                                }
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        let inv = 1.0 / (count as f32);
                        for c in 0..4 {
                            acc.r[c] *= inv;
                            acc.g[c] *= inv;
                            acc.b[c] *= inv;
                        }
                        acc.valid = true;
                        probes[i] = acc;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x007F_FFFF;
    if exp == 255 {
        return sign | 0x7C00 | u16::from(frac != 0);
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7BFF;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = (frac | 0x0080_0000) >> (14 - new_exp);
        return sign | (m as u16);
    }
    sign | ((new_exp as u16) << 10) | ((frac >> 13) as u16)
}

fn serialize_gi(probes: &[ProbeSh]) -> Vec<u8> {
    let mut out = Vec::with_capacity(40 + probes.len() * 24);
    out.extend_from_slice(&GI_MAGIC);
    for u in [GI_VERSION, GRID_X as u32, GRID_Y as u32, GRID_Z as u32, 0] {
        out.extend_from_slice(&u.to_le_bytes());
    }
    for f in GRID_MIN {
        out.extend_from_slice(&f.to_le_bytes());
    }
    for f in GRID_MAX {
        out.extend_from_slice(&f.to_le_bytes());
    }
    for ch in 0..3 {
        for p in probes {
            let v = match ch {
                0 => p.r,
                1 => p.g,
                _ => p.b,
            };
            for f in v {
                out.extend_from_slice(&f32_to_f16(f).to_le_bytes());
            }
        }
    }
    out
}

pub fn bake_gi(glbs_dir: &Path, pbr_base_path: &Path, out_path: &Path) -> Result<BakeGiOutput, Refusal> {
    let start = Instant::now();
    let mats = load_pbr_materials(pbr_base_path)?;
    let mut tris = load_scene_triangles(glbs_dir)?;
    let tri_count = tris.len() as u32;
    let mut nodes = Vec::with_capacity(tris.len() / 2);
    let total_tris = tris.len();
    if total_tris > 0 {
        build_bvh_node(&mut tris, 0, total_tris, &mut nodes);
    }
    let scene = SceneBvh { tris, nodes, mats };
    let sun_dir = Vec3::new(0.18, 0.93, 0.32).normalize();
    let total_probes = GRID_X * GRID_Y * GRID_Z;
    let mut positions = Vec::with_capacity(total_probes);
    for z in 0..GRID_Z {
        let fz = (z as f32 + 0.5) / (GRID_Z as f32);
        let pz = GRID_MIN[2] + (GRID_MAX[2] - GRID_MIN[2]) * fz;
        for y in 0..GRID_Y {
            let fy = (y as f32 + 0.5) / (GRID_Y as f32);
            let py = GRID_MIN[1] + (GRID_MAX[1] - GRID_MIN[1]) * fy;
            for x in 0..GRID_X {
                let fx = (x as f32 + 0.5) / (GRID_X as f32);
                let px = GRID_MIN[0] + (GRID_MAX[0] - GRID_MIN[0]) * fx;
                positions.push(Vec3::new(px, py, pz));
            }
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let chunk = total_probes.div_ceil(threads.max(1));
    let mut probes = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for first in (0..total_probes).step_by(chunk) {
            let end = (first + chunk).min(total_probes);
            let scene_ref = &scene;
            let pos_slice = &positions[first..end];
            handles.push(scope.spawn(move || {
                let mut part = Vec::with_capacity(end - first);
                for (offset, &pos) in pos_slice.iter().enumerate() {
                    part.push(bake_single_probe(scene_ref, first + offset, pos, sun_dir));
                }
                part
            }));
        }
        let mut all = Vec::with_capacity(total_probes);
        for h in handles {
            if let Ok(part) = h.join() {
                all.extend(part);
            }
        }
        all
    });
    let valid_probes = probes.iter().filter(|p| p.valid).count() as u32;
    fill_invalid_probes(&mut probes);
    let bytes = serialize_gi(&probes);
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| Refusal::new("OutputDir", &parent.display().to_string(), format!("{e}")))?;
    }
    std::fs::write(out_path, &bytes)
        .map_err(|e| Refusal::new("WriteFailed", &out_path.display().to_string(), format!("{e}")))?;
    Ok(BakeGiOutput {
        status: "pass",
        out_path: out_path.display().to_string(),
        probe_count: total_probes as u32,
        valid_probes,
        triangle_count: tri_count,
        out_bytes: bytes.len(),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}
