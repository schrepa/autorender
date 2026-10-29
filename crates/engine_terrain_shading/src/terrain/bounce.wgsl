import package::terrain::horizons::{terrain_horizon_records, terrain_horizon_sampler,
    terrain_horizons};

override BOUNCE: bool = true;

const BOUNCE_LOW: f32 = 1.5;
const BOUNCE_HIGH: f32 = 24.0;
const BOUNCE_WINDOW_ROW: u32 = 4096u;

fn bounce_layer(texel: vec2<f32>, first: u32, normal: vec3<f32>) -> vec4<f32> {
    let mean = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first, 0.0);
    let x = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first + 1u,
        0.0);
    let y = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first + 2u,
        0.0);
    let z = textureSampleLevel(terrain_horizons, terrain_horizon_sampler, texel, first + 3u,
        0.0);
    return mean + x * normal.x + y * normal.y + z * normal.z;
}

fn bounce_texel(at: u32, level: u32, uv: vec2<f32>) -> vec2<f32> {
    let header = terrain_horizon_records[at + 2u];
    let arranged = terrain_horizon_records[at + 3u];
    let columns = f32(u32(header.y) >> level);
    let window = min(header.w, columns);
    let packed = u32(terrain_horizon_records[at + 4u + level / 4u][level % 4u]);
    let first = vec2<f32>(f32(packed % BOUNCE_WINDOW_ROW), f32(packed / BOUNCE_WINDOW_ROW));
    let column = clamp(uv * columns - 0.5, vec2<f32>(0.0), vec2<f32>(columns - 1.0)) - first;
    if any(column < vec2<f32>(0.0)) || any(column > vec2<f32>(window - 1.0)) {
        return vec2<f32>(-1.0);
    }
    let row = u32(arranged.z);
    let region = vec2<f32>(f32(level % row), f32(level / row)) * header.w;
    return (region + column + 0.5) * arranged.x;
}

fn bounce_held(texel: vec2<f32>, first: u32, normal: vec3<f32>, above: f32) -> vec4<f32> {
    let held = bounce_layer(texel, first, normal);
    if above <= 0.0 {
        return held;
    }
    return mix(held, bounce_layer(texel, first + 4u, normal), above);
}
