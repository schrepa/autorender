const PAGE_WORDS: u32 = 4096u;
const CLUSTER_HEADER_WORDS: u32 = 7u;
const CLUSTER_DRAW_VERTS: u32 = 384u;
const MAX_PHASE_CLUSTERS: u32 = 131072u;
const MAX_VIS_CLUSTERS: u32 = 524288u;
const MAX_OCC_INSTANCES: u32 = 65536u;
const MAX_OCC_CLUSTERS: u32 = 262144u;
const DEDUP_MASK: u32 = 262143u;
const BVH_QUEUE_CAP: u32 = 262144u;
const CAND_CAP: u32 = 524288u;
const SW_DISPATCH_STRIDE: u32 = 16384u;
const UNMAPPED_PAGE: u32 = 0xFFFFFFFFu;
const NO_CHILD: u32 = 0xFFFFFFFFu;
const FLAG_DOUBLE_SIDED_SHIFTED: u32 = 0x00100000u;
const SHADOW_LIST_CAP: u32 = 131072u;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn transform_pos(pos_scale: vec4<f32>, rot_quat: vec4<f32>, p: vec3<f32>) -> vec3<f32> {
    return pos_scale.xyz + quat_rotate(rot_quat, p * pos_scale.w);
}

fn decode_qpos_u16(w0: u32, w1: u32, vi: u32) -> vec3<f32> {
    var qx = w0 & 0xFFFFu;
    var qy = w0 >> 16u;
    var qz = w1 & 0xFFFFu;
    if (vi & 1u) == 1u {
        qx = w0 >> 16u;
        qy = w1 & 0xFFFFu;
        qz = w1 >> 16u;
    }
    return vec3<f32>(f32(qx), f32(qy), f32(qz));
}

fn decode_pos_u16(w0: u32, w1: u32, vi: u32, origin: vec3<f32>, step: f32) -> vec3<f32> {
    return origin + decode_qpos_u16(w0, w1, vi) * step;
}

fn decode_octahedral_10(packed24: u32) -> vec3<f32> {
    let px = (f32(packed24 & 0x3FFu) - 511.0) * (1.0 / 511.0);
    let py = (f32((packed24 >> 10u) & 0x3FFu) - 511.0) * (1.0 / 511.0);
    let pz = 1.0 - abs(px) - abs(py);
    let fold = max(-pz, 0.0);
    let x = px + select(fold, -fold, px >= 0.0);
    let y = py + select(fold, -fold, py >= 0.0);
    let raw = vec3<f32>(x, y, pz);
    let l2 = dot(raw, raw);
    if l2 > 1.0e-20 {
        return raw * inverseSqrt(l2);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn decode_uv_12(packed24: u32, uv_origin: vec2<f32>, uv_step: f32) -> vec2<f32> {
    let qu = f32(packed24 & 0xFFFu);
    let qv = f32((packed24 >> 12u) & 0xFFFu);
    return uv_origin + vec2<f32>(qu, qv) * uv_step;
}

fn projected_error(
    cull_pos: vec3<f32>,
    near_z: f32,
    lod_scale: f32,
    world_c: vec3<f32>,
    world_r: f32,
    err: f32,
    inst_scale: f32,
) -> f32 {
    let dist = max(length(cull_pos - world_c) - world_r, near_z);
    return (err * inst_scale * lod_scale) / dist;
}

fn edge_yup_i32(a: vec2<i32>, b: vec2<i32>, c: vec2<i32>) -> i32 {
    return (b.x - a.x) * (a.y - c.y) - (a.y - b.y) * (c.x - a.x);
}

fn is_top_left_i32(a: vec2<i32>, b: vec2<i32>) -> bool {
    return (b.y > a.y) || (b.y == a.y && b.x < a.x);
}

fn dedup_hash_pair(inst_id: u32, cluster_id: u32) -> vec2<u32> {
    var h0 = (inst_id * 2654435761u) ^ (cluster_id * 2246822519u);
    h0 = (h0 ^ (h0 >> 16u)) * 0x45D9F3Bu;
    h0 = h0 ^ (h0 >> 16u);
    var h1 = (inst_id * 3266489917u + 0x9E3779B9u) ^ (cluster_id * 668265263u);
    h1 = (h1 ^ (h1 >> 15u)) * 0x27D4EB2Du;
    h1 = h1 ^ (h1 >> 16u);
    let tag = max(h1 & 0x07FFFFFFu, 1u);
    return vec2<u32>(h0 & DEDUP_MASK, tag);
}

fn hash_u32(v_in: u32) -> u32 {
    var v = v_in;
    v = (v ^ 61u) ^ (v >> 16u);
    v = v + (v << 3u);
    v = v ^ (v >> 4u);
    v = v * 0x27D4EB2Du;
    v = v ^ (v >> 15u);
    return v;
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = fract(vec3<f32>(h, h + 0.6666667, h + 0.3333333)) * 6.0;
    let rgb = clamp(abs(k - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return v * mix(vec3<f32>(1.0), rgb, s);
}

fn hash_color(id: u32) -> vec3<f32> {
    let h = hash_u32(id);
    let hue = f32(h & 0xFFFFu) * (1.0 / 65536.0);
    let sat = 0.62 + 0.28 * f32((h >> 16u) & 0xFFu) * (1.0 / 255.0);
    let val = 0.68 + 0.28 * f32((h >> 24u) & 0xFFu) * (1.0 / 255.0);
    return hsv_to_rgb(hue, sat, val);
}

fn lod_heat_color(lod: u32) -> vec3<f32> {
    let t = clamp(f32(lod) * 0.06, 0.0, 1.0);
    let c0 = vec3<f32>(0.06, 0.34, 0.92);
    let c1 = vec3<f32>(0.05, 0.78, 0.62);
    let c2 = vec3<f32>(0.96, 0.78, 0.12);
    let c3 = vec3<f32>(0.94, 0.18, 0.24);
    if t < 0.33 {
        return mix(c0, c1, t * 3.0);
    } else if t < 0.66 {
        return mix(c1, c2, (t - 0.33) * 3.0);
    }
    return mix(c2, c3, (t - 0.66) * 3.0);
}

fn ray_tri_bary_uv(
    d_vec: vec3<f32>,
    m_vec: vec3<f32>,
    q_vec: vec3<f32>,
    r: vec3<f32>,
) -> vec2<f32> {
    let det = dot(r, d_vec);
    if abs(det) <= 1.0e-12 {
        return vec2<f32>(0.3333333, 0.3333334);
    }
    let inv_det = 1.0 / det;
    return vec2<f32>(dot(r, m_vec), dot(r, q_vec)) * inv_det;
}

fn clamp_uv_grad(g: vec2<f32>) -> vec2<f32> {
    let l2 = dot(g, g);
    if l2 > 0.0625 {
        return g * (0.25 * inverseSqrt(l2));
    }
    return g;
}

fn build_tangent_frame(
    e1: vec3<f32>,
    e2: vec3<f32>,
    du1: vec2<f32>,
    du2: vec2<f32>,
    n: vec3<f32>,
) -> vec4<f32> {
    let det = du1.x * du2.y - du1.y * du2.x;
    if abs(det) <= 1.0e-8 {
        return vec4<f32>(0.0);
    }
    let inv_det = 1.0 / det;
    let t_raw = (e1 * du2.y - e2 * du1.y) * inv_det;
    let b_raw = (e2 * du1.x - e1 * du2.x) * inv_det;
    let t_ortho = t_raw - n * dot(n, t_raw);
    let t_l2 = dot(t_ortho, t_ortho);
    if t_l2 <= 1.0e-8 {
        return vec4<f32>(0.0);
    }
    let t = t_ortho * inverseSqrt(t_l2);
    let sign_b = select(-1.0, 1.0, dot(cross(n, t), b_raw) >= 0.0);
    return vec4<f32>(t, sign_b);
}
