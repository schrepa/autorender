import package::generated::pieces::pieces;
import package::sky::air::{behind_clouds, cloud_dome_direction, cloud_dome_uv, haze};
import package::sky::common::air_sky;

@group(0) @binding(24) var sky_view_table: texture_2d<f32>;
@group(0) @binding(25) var sky_table_sampler: sampler;
@group(0) @binding(26) var sky_clouds: texture_2d_array<f32>;
@group(0) @binding(27) var sky_clouds_old: texture_2d_array<f32>;
@group(0) @binding(28) var sky_cloud_sampler: sampler;

const EYE_LAYER: u32 = 0u;
const MIRROR_LAYER: u32 = 1u;

fn physical_sky() -> bool {
    return pieces.sky.state.x > 0.5;
}

fn sky_air_light(direction: vec3<f32>) -> vec3<f32> {
    if physical_sky() {
        return air_sky(sky_view_table, sky_table_sampler, direction);
    }
    return haze(direction);
}

fn dome_at(dome: texture_2d_array<f32>, uv: vec2<f32>, layer: u32) -> vec4<f32> {
    let half_texel = 0.5 / vec2<f32>(textureDimensions(dome));
    return 0.25 * (textureSampleLevel(dome, sky_cloud_sampler, uv - half_texel, layer, 0.0)
        + textureSampleLevel(dome, sky_cloud_sampler, uv + half_texel, layer, 0.0)
        + textureSampleLevel(dome, sky_cloud_sampler,
            uv + vec2<f32>(half_texel.x, -half_texel.y), layer, 0.0)
        + textureSampleLevel(dome, sky_cloud_sampler,
            uv + vec2<f32>(-half_texel.x, half_texel.y), layer, 0.0));
}

fn sky_cloud(layer: u32, start: vec3<f32>, ray: vec3<f32>) -> vec4<f32> {
    if pieces.sky.domes_older.w < 0.5 {
        return dome_at(sky_clouds, cloud_dome_uv(ray), layer);
    }
    let mirrored = layer == MIRROR_LAYER;
    let newer_origin = select(pieces.sky.domes_newer.xyz, pieces.sky.domes_newer_mirror.xyz,
        mirrored);
    let older_origin = select(pieces.sky.domes_older.xyz, pieces.sky.domes_older_mirror.xyz,
        mirrored);
    let newer = dome_at(sky_clouds,
        cloud_dome_uv(cloud_dome_direction(newer_origin, start, ray)), layer);
    let older = dome_at(sky_clouds_old,
        cloud_dome_uv(cloud_dome_direction(older_origin, start, ray)), layer);
    return mix(older, newer, pieces.sky.domes_newer.w);
}

fn clouded(light: vec3<f32>, start: vec3<f32>, ray: vec3<f32>) -> vec3<f32> {
    return behind_clouds(light, sky_cloud(EYE_LAYER, start, ray), ray);
}

fn sky_along(start: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    return clouded(sky_air_light(direction), start, direction);
}
