struct Particle {
    pos_foam: vec4<f32>,
    vel_dens: vec4<f32>,
    c_packed: vec4<u32>,
    extra: vec4<f32>,
}

struct SimParams {
    dims: vec4<u32>,
    physics: vec4<f32>,
    fluid: vec4<f32>,
    stir_pos: vec4<f32>,
    stir_vel: vec4<f32>,
    drop_info: vec4<u32>,
    bounds: vec4<f32>,
    time_info: vec4<f32>,
}

struct FrameUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    eye_time: vec4<f32>,
    viewport: vec4<f32>,
    splat: vec4<f32>,
    style: vec4<f32>,
    stir: vec4<f32>,
    basin: vec4<f32>,
    grid_dims: vec4<f32>,
}

struct UiQuad {
    rect: vec4<f32>,
    uv: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
}

struct UiBuffer {
    header: vec4<f32>,
    quads: array<UiQuad>,
}

const FIXED_SCALE: f32 = 65536.0;
const INV_FIXED_SCALE: f32 = 1.0 / 65536.0;

fn grid_to_world(g: vec3<f32>, basin: vec4<f32>, grid_dims: vec4<f32>) -> vec3<f32> {
    let cs = basin.w;
    return vec3<f32>(
        (g.x - grid_dims.x * 0.5) * cs,
        (g.y - 3.0) * cs + basin.z,
        (g.z - grid_dims.z * 0.5) * cs,
    );
}

fn cell_idx(x: u32, y: u32, z: u32, dims: vec3<u32>) -> u32 {
    return (z * dims.y + y) * dims.x + x;
}

fn hash_u32(v: u32) -> u32 {
    var x = v * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed) >> 8u) * (1.0 / 16777216.0);
}

fn unpack_c(packed: vec4<u32>, c22: f32) -> mat3x3<f32> {
    let a = unpack2x16float(packed.x);
    let b = unpack2x16float(packed.y);
    let c = unpack2x16float(packed.z);
    let d = unpack2x16float(packed.w);
    return mat3x3<f32>(
        vec3<f32>(a.x, a.y, b.x),
        vec3<f32>(b.y, c.x, c.y),
        vec3<f32>(d.x, d.y, c22),
    );
}

fn pack_c_vec(m: mat3x3<f32>) -> vec4<u32> {
    let c0 = clamp(m[0], vec3<f32>(-40.0), vec3<f32>(40.0));
    let c1 = clamp(m[1], vec3<f32>(-40.0), vec3<f32>(40.0));
    let c2 = clamp(m[2], vec3<f32>(-40.0), vec3<f32>(40.0));
    return vec4<u32>(
        pack2x16float(vec2<f32>(c0.x, c0.y)),
        pack2x16float(vec2<f32>(c0.z, c1.x)),
        pack2x16float(vec2<f32>(c1.y, c1.z)),
        pack2x16float(vec2<f32>(c2.x, c2.y)),
    );
}

fn encode_fixed(v: f32) -> i32 {
    return i32(clamp(v * FIXED_SCALE, -1073741824.0, 1073741824.0));
}

fn decode_fixed(v: i32) -> f32 {
    return f32(v) * INV_FIXED_SCALE;
}
