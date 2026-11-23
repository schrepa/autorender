import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;

const PAGE_WORDS: u32 = 8192u;
const CLUSTER_HEADER_WORDS: u32 = 4u;
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

fn decode_pos(w0: u32, w1: u32, origin: vec3<f32>, step: f32) -> vec3<f32> {
    let qx = f32(w0 & 0xFFFFu);
    let qy = f32(w0 >> 16u);
    let qz = f32(w1 & 0xFFFFu);
    return origin + vec3<f32>(qx, qy, qz) * step;
}

fn decode_octahedral(w2: u32) -> vec3<f32> {
    let p = unpack2x16snorm(w2);
    let pz = 1.0 - abs(p.x) - abs(p.y);
    let fold = max(-pz, 0.0);
    let x = p.x + select(fold, -fold, p.x >= 0.0);
    let y = p.y + select(fold, -fold, p.y >= 0.0);
    let raw = vec3<f32>(x, y, pz);
    let l2 = dot(raw, raw);
    if l2 > 1.0e-20 {
        return raw * inverseSqrt(l2);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn decode_uv(w3: u32) -> vec2<f32> {
    return vec2<f32>(f32(w3 & 0xFFFFu), f32(w3 >> 16u)) * (1.0 / 2048.0);
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
