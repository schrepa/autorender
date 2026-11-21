struct SimParams {
    dims: vec4<u32>,
    physics: vec4<f32>,
    fluid: vec4<f32>,
    stir_pos: vec4<f32>,
    stir_vel: vec4<f32>,
    drop_info: vec4<u32>,
    bounds: vec4<f32>,
    time_info: vec4<f32>,
    eye_grid: vec4<f32>,
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
const BRICK_X: u32 = 21u;
const BRICK_Y: u32 = 12u;
const BRICK_Z: u32 = 14u;
const NUM_BRICKS: u32 = 3528u;

fn grid_to_world(g: vec3<f32>, basin: vec4<f32>, grid_dims: vec4<f32>) -> vec3<f32> {
    let cs = basin.w;
    return vec3<f32>(
        (g.x - grid_dims.x * 0.5) * cs,
        (g.y - 3.0) * cs + basin.z,
        (g.z - grid_dims.z * 0.5) * cs,
    );
}

fn brick_idx(bx: u32, by: u32, bz: u32) -> u32 {
    return (bz * BRICK_Y + by) * BRICK_X + bx;
}

fn cell_idx(x: u32, y: u32, z: u32) -> u32 {
    let brick = brick_idx(x >> 2u, y >> 2u, z >> 2u);
    return (brick << 6u) | ((z & 3u) << 4u) | ((y & 3u) << 2u) | (x & 3u);
}

fn hash_u32(v: u32) -> u32 {
    var x = v * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (x >> 22u) ^ x;
}

fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed) >> 8u) * (1.0 / 16777216.0);
}

fn pack_meta(foam: f32, energy: f32, ratio: f32, speed: f32) -> f32 {
    let v = clamp(
        vec4<f32>(foam, energy * 0.5, ratio * 0.33333334, speed * 0.4),
        vec4<f32>(0.0),
        vec4<f32>(1.0),
    );
    return bitcast<f32>(pack4x8unorm(v));
}

fn unpack_meta(w: f32) -> vec4<f32> {
    let v = unpack4x8unorm(bitcast<u32>(w));
    return vec4<f32>(v.x, v.y * 2.0, v.z * 3.0, v.w * 2.5);
}

fn pack_grid_vel(vel: vec3<f32>, mass: f32) -> vec2<u32> {
    return vec2<u32>(
        pack2x16float(vel.xy),
        pack2x16float(vec2<f32>(vel.z, mass)),
    );
}

fn unpack_grid_vel(g: vec2<u32>) -> vec4<f32> {
    let xy = unpack2x16float(g.x);
    let zm = unpack2x16float(g.y);
    return vec4<f32>(xy.x, xy.y, zm.x, zm.y);
}

fn encode_fixed(v: f32) -> i32 {
    return i32(clamp(v * FIXED_SCALE, -1073741824.0, 1073741824.0));
}

fn decode_fixed(v: i32) -> f32 {
    return f32(v) * INV_FIXED_SCALE;
}
