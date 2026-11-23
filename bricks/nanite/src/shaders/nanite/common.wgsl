import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;

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

fn inst_to_world(inst: NaniteInstance, p: vec3<f32>) -> vec3<f32> {
    return inst.pos_scale.xyz + quat_rotate(inst.rot_quat, p * inst.pos_scale.w);
}

fn decode_qpos_u16(w0: u32, w1: u32, odd: bool) -> vec3<f32> {
    let qx = select(w0 & 0xFFFFu, w0 >> 16u, odd);
    let qy = select(w0 >> 16u, w1 & 0xFFFFu, odd);
    let qz = select(w1 & 0xFFFFu, w1 >> 16u, odd);
    return vec3<f32>(f32(qx), f32(qy), f32(qz));
}

fn decode_pos_u16(w0: u32, w1: u32, odd: bool, origin: vec3<f32>, step: f32) -> vec3<f32> {
    return origin + decode_qpos_u16(w0, w1, odd) * step;
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
    return select(vec3<f32>(0.0, 1.0, 0.0), raw * inverseSqrt(l2), l2 > 1.0e-20);
}

fn decode_uv_12(packed24: u32, uv_origin: vec2<f32>, uv_step: f32) -> vec2<f32> {
    let qu = f32(packed24 & 0xFFFu);
    let qv = f32((packed24 >> 12u) & 0xFFFu);
    return uv_origin + vec2<f32>(qu, qv) * uv_step;
}

fn cluster_vert_count(c: GpuClusterHeader) -> u32 {
    return c.errors_and_cone.w & 0xFFu;
}

fn cluster_tri_count(c: GpuClusterHeader) -> u32 {
    return (c.errors_and_cone.w >> 8u) & 0xFFu;
}

fn cluster_lod(c: GpuClusterHeader) -> u32 {
    return (c.errors_and_cone.w >> 16u) & 0xFFu;
}

fn cluster_mat(c: GpuClusterHeader) -> u32 {
    return (c.errors_and_cone.w >> 24u) & 0xFFu;
}

fn cluster_page_id(c: GpuClusterHeader) -> u32 {
    return c.page_and_fallback.x;
}

fn cluster_word_offset(c: GpuClusterHeader) -> u32 {
    return c.page_and_fallback.y & 0xFFFFu;
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
