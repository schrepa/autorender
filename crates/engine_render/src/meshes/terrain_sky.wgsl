// Terrain's sky map in the mesh passes, after `terrain.wgsl`: where its
// ground sees the sky; the light the ground around gives back where it does
// not is bounce light's (`bounce_light`).

// Each terrain's sky map, cooked with its lighting: over the
// whole terrain, a layer from `terrain.directory.w` less one: the share of
// the sky its ground sees about its normal (red), and the turn of its normal
// toward the middle of the sky it sees (green, blue and alpha: x, y and z,
// from 0.5).
@group(1) @binding(30) var terrain_sky: texture_2d_array<f32>;

// Where terrain-local `local` lies on the terrain's sky map.
fn terrain_sky_uv(local: vec2<f32>) -> vec2<f32> {
    let header = terrain.directory.x;
    let extent = vec2<f32>(f32(terrain_directory[header + 1u]),
        f32(terrain_directory[header + 2u])) * 256.0 * terrain.anchor.w;
    return local / extent;
}

// The terrain's sky map at terrain-local `local` once it arrived: the share
// of the sky the ground sees (red), and how its normal turns toward the
// middle of the sky it sees (its bent normal less its normal: green, blue
// and alpha, from 0.5); before, all the sky and no turn.
fn terrain_sky_map(local: vec2<f32>) -> vec4<f32> {
    if terrain.directory.w == 0u {
        return vec4<f32>(1.0, 0.5, 0.5, 0.5);
    }
    return textureSampleLevel(terrain_sky, terrain_sky_sampler, terrain_sky_uv(local),
        terrain.directory.w - 1u, 0.0);
}
