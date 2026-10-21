// Terrain's sky map in the mesh passes, after `terrain.wgsl`: where its
// ground sees the sky, and past its bounce light's probes the light the
// ground around gives back where it does not.

// Each terrain's sky map, cooked with its lighting: over the
// whole terrain, two layers from `terrain.directory.w` less one: the share
// of the sky its ground sees about its normal (red), its hollows and ridges
// (green and blue); then the turn of its normal toward the middle of the sky
// it sees (red, green and blue: x, y and z, from 0.5).
@group(1) @binding(30) var terrain_sky: texture_2d_array<f32>;

// Where terrain-local `local` lies on the terrain's sky map.
fn terrain_sky_uv(local: vec2<f32>) -> vec2<f32> {
    let header = terrain.directory.x;
    let extent = vec2<f32>(f32(terrain_directory[header + 1u]),
        f32(terrain_directory[header + 2u])) * 256.0 * terrain.anchor.w;
    return local / extent;
}

// The terrain's sky map at terrain-local `local` once it arrived: the share
// of the sky the ground sees (red), and its large-scale variety,
// hollows (green) and ridges (blue); before, all the sky
// and level ground.
fn terrain_sky_map(local: vec2<f32>) -> vec4<f32> {
    if terrain.directory.w == 0u {
        return vec4<f32>(1.0, 0.5, 0.5, 1.0);
    }
    return textureSampleLevel(terrain_sky, terrain_normal_sampler, terrain_sky_uv(local),
        terrain.directory.w - 1u, 0.0);
}

// How the ground's normal at terrain-local `local` turns toward the middle
// of the sky it sees (its bent normal less its normal), once the sky map
// arrived; before, not at all.
fn terrain_sky_turn(local: vec2<f32>) -> vec3<f32> {
    if terrain.directory.w == 0u {
        return vec3<f32>(0.0);
    }
    let turn = textureSampleLevel(terrain_sky, terrain_normal_sampler, terrain_sky_uv(local),
        terrain.directory.w, 0.0).rgb;
    return turn * 2.0 - 1.0;
}

// The light the terrain around a hollow gives back per unit albedo for the
// sky it hides, where no probe relights (`bounce_light`): sunlit and skylit
// ground, as level ground would be.
fn terrain_bounce() -> vec3<f32> {
    return lighting.sun_color.rgb * (0.35 * saturate(lighting.sun.y))
        + sky_average(vec3<f32>(0.0, 1.0, 0.0)) * (0.5 * lighting.zenith.w);
}
