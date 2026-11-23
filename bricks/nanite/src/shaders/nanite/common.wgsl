const PAGE_WORDS: u32 = 4096u;
const CLUSTER_HEADER_WORDS: u32 = 7u;
const CLUSTER_DRAW_VERTS: u32 = 384u;
const MAX_PHASE_CLUSTERS: u32 = 131072u;
const MAX_VIS_CLUSTERS: u32 = 524288u;
const MAX_OCC_INSTANCES: u32 = 4096u;
const MAX_OCC_CLUSTERS: u32 = 262144u;
const DEDUP_MASK: u32 = 262143u;
const BVH_QUEUE_CAP: u32 = 262144u;
const CAND_CAP: u32 = 524288u;
const SW_DISPATCH_STRIDE: u32 = 16384u;
const UNMAPPED_PAGE: u32 = 0xFFFFFFFFu;
const NO_CHILD: u32 = 0xFFFFFFFFu;
const SHADOW_C0_DIM: u32 = 2048u;
const SHADOW_C1_DIM: u32 = 2048u;
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
