import package::generated::nanite::NaniteUniforms;
import package::generated::nanite::NaniteInstance;
import package::generated::nanite::GpuClusterHeader;

const PAGE_WORDS: u32 = 16384u;
const MAX_PHASE_CLUSTERS: u32 = 262144u;
const MAX_VIS_CLUSTERS: u32 = 524288u;
const MAX_OCC_INSTANCES: u32 = 4096u;
const MAX_OCC_CLUSTERS: u32 = 262144u;
const DEDUP_MASK: u32 = 262143u;
const BVH_QUEUE_CAP: u32 = 262144u;
const CAND_CAP: u32 = 524288u;
const SW_DISPATCH_STRIDE: u32 = 16384u;
const UNMAPPED_PAGE: u32 = 0xFFFFFFFFu;
const NO_FALLBACK: u32 = 0xFFFFFFFFu;
const NO_CHILD: u32 = 0xFFFFFFFFu;
const SHADOW_C0_DIM: u32 = 1024u;
const SHADOW_C1_DIM: u32 = 2048u;
const SHADOW_LIST_CAP: u32 = 131072u;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn inst_to_world(inst: NaniteInstance, p: vec3<f32>) -> vec3<f32> {
    return inst.pos_scale.xyz + quat_rotate(inst.rot_quat, p * inst.pos_scale.w);
}

fn decode_pos(w0: u32, w1: u32, aabb_min: vec3<f32>, aabb_extent: vec3<f32>) -> vec3<f32> {
    let qx = f32(w0 & 0x3FFFu) * (1.0 / 16383.0);
    let qy = f32((w0 >> 14u) & 0x3FFFu) * (1.0 / 16383.0);
    let gz = ((w0 >> 28u) & 0xFu) | ((w1 & 0x3FFu) << 4u);
    let qz = f32(gz) * (1.0 / 16383.0);
    return aabb_min + vec3<f32>(qx, qy, qz) * aabb_extent;
}

fn decode_color_ao(w1: u32) -> vec4<f32> {
    let r = f32((w1 >> 10u) & 0x3Fu) * (1.0 / 63.0);
    let g = f32((w1 >> 16u) & 0x3Fu) * (1.0 / 63.0);
    let b = f32((w1 >> 22u) & 0x1Fu) * (1.0 / 31.0);
    let ao = f32((w1 >> 27u) & 0x1Fu) * (1.0 / 31.0);
    return vec4<f32>(r * r, g * g, b * b, ao);
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
    return c.cone_and_meta.y & 0xFFu;
}

fn cluster_tri_count(c: GpuClusterHeader) -> u32 {
    return (c.cone_and_meta.y >> 8u) & 0xFFu;
}

fn cluster_mat(c: GpuClusterHeader) -> u32 {
    return (c.cone_and_meta.y >> 24u) & 0xFFu;
}

fn frustum_visible(u: NaniteUniforms, c: vec3<f32>, r: f32) -> bool {
    if dot(u.plane0.xyz, c) + u.plane0.w < -r { return false; }
    if dot(u.plane1.xyz, c) + u.plane1.w < -r { return false; }
    if dot(u.plane2.xyz, c) + u.plane2.w < -r { return false; }
    if dot(u.plane3.xyz, c) + u.plane3.w < -r { return false; }
    if dot(u.plane4.xyz, c) + u.plane4.w < -r { return false; }
    if dot(u.plane5.xyz, c) + u.plane5.w < -r { return false; }
    return true;
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

fn projected_min_error(
    cull_pos: vec3<f32>,
    near_z: f32,
    lod_scale: f32,
    world_c: vec3<f32>,
    world_r: f32,
    err: f32,
    inst_scale: f32,
) -> f32 {
    let dist = max(length(cull_pos - world_c) + world_r, near_z);
    return (err * inst_scale * lod_scale) / dist;
}

fn unpack_i8(word: u32, shift: u32) -> i32 {
    let raw = i32((word >> shift) & 0xFFu);
    return select(raw, raw - 256, raw >= 128);
}

fn cone_backface_culled(
    c: GpuClusterHeader,
    inst: NaniteInstance,
    world_c: vec3<f32>,
    world_r: f32,
    cull_pos: vec3<f32>,
) -> bool {
    let packed = c.cone_and_meta.x;
    let cutoff = unpack_i8(packed, 24u);
    if cutoff >= 127 {
        return false;
    }
    let to_c = world_c - cull_pos;
    let dist = length(to_c);
    if dist <= world_r {
        return false;
    }
    let ax = vec3<f32>(
        f32(unpack_i8(packed, 0u)),
        f32(unpack_i8(packed, 8u)),
        f32(unpack_i8(packed, 16u)),
    );
    let ax_l2 = dot(ax, ax);
    if ax_l2 <= 1.0e-4 {
        return false;
    }
    let local_axis = ax * inverseSqrt(ax_l2);
    let world_axis = quat_rotate(inst.rot_quat, local_axis);
    let sin_cone = f32(cutoff) * (1.0 / 127.0);
    return dot(to_c, world_axis) >= sin_cone * dist + world_r;
}

fn edge_yup(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return (b.x - a.x) * (a.y - c.y) - (a.y - b.y) * (c.x - a.x);
}

fn edge_yup_i32(a: vec2<i32>, b: vec2<i32>, c: vec2<i32>) -> i32 {
    return (b.x - a.x) * (a.y - c.y) - (a.y - b.y) * (c.x - a.x);
}

fn is_top_left_i32(a: vec2<i32>, b: vec2<i32>) -> bool {
    return (b.y > a.y) || (b.y == a.y && b.x < a.x);
}
