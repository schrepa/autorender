import package::face::heightfield::{terrain, terrain_directory};

@group(1) @binding(22) var terrain_sky_sampler: sampler;
@group(1) @binding(30) var terrain_sky: texture_2d_array<f32>;

fn terrain_sky_uv(local: vec2<f32>) -> vec2<f32> {
    let header = terrain.directory.x;
    let extent = vec2<f32>(f32(terrain_directory[header + 1u]),
        f32(terrain_directory[header + 2u])) * 256.0 * terrain.anchor.w;
    return local / extent;
}

fn terrain_sky_map(local: vec2<f32>) -> vec4<f32> {
    if terrain.directory.w == 0u {
        return vec4<f32>(1.0, 0.5, 0.5, 0.5);
    }
    return textureSampleLevel(terrain_sky, terrain_sky_sampler, terrain_sky_uv(local),
        terrain.directory.w - 1u, 0.0);
}
