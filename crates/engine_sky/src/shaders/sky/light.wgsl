import package::generated::sky::{EYE_LAYER, MIRROR_LAYER};
import package::sky::air::{behind_clouds, cloud_dome_direction, cloud_dome_uv};
import package::sky::common::air_sky;
import package::sky::piece::sky_piece;

@group(0) @binding(0) var sky_view_table: texture_2d<f32>;
@group(0) @binding(0) var sky_table_sampler: sampler;
@group(0) @binding(0) var sky_domes: texture_2d_array<f32>;
@group(0) @binding(0) var sky_cloud_sampler: sampler;

fn sky_air_light(direction: vec3<f32>) -> vec3<f32> {
    return air_sky(sky_view_table, sky_table_sampler, direction);
}

fn dome_at(uv: vec2<f32>, layer: u32) -> vec4<f32> {
    let half_texel = 0.5 / vec2<f32>(textureDimensions(sky_domes));
    return 0.25 * (textureSampleLevel(sky_domes, sky_cloud_sampler, uv - half_texel, layer, 0.0)
        + textureSampleLevel(sky_domes, sky_cloud_sampler, uv + half_texel, layer, 0.0)
        + textureSampleLevel(sky_domes, sky_cloud_sampler,
            uv + vec2<f32>(half_texel.x, -half_texel.y), layer, 0.0)
        + textureSampleLevel(sky_domes, sky_cloud_sampler,
            uv + vec2<f32>(-half_texel.x, half_texel.y), layer, 0.0));
}

fn sky_cloud_in(layer: u32, start: vec3<f32>, ray: vec3<f32>, shown: bool) -> vec4<f32> {
    if sky_piece.domes_older.w < 0.5 || !shown {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let mirrored = layer == MIRROR_LAYER;
    let newer_origin = select(sky_piece.domes_newer.xyz, sky_piece.domes_newer_mirror.xyz,
        mirrored);
    let older_origin = select(sky_piece.domes_older.xyz, sky_piece.domes_older_mirror.xyz,
        mirrored);
    let newer = dome_at(cloud_dome_uv(cloud_dome_direction(newer_origin, start, ray)),
        sky_piece.domes_layers.y + layer);
    let older = dome_at(cloud_dome_uv(cloud_dome_direction(older_origin, start, ray)),
        sky_piece.domes_layers.x + layer);
    return mix(older, newer, sky_piece.domes_newer.w);
}

fn sky_cloud(layer: u32, start: vec3<f32>, ray: vec3<f32>) -> vec4<f32> {
    return sky_cloud_in(layer, start, ray, true);
}

fn clouded_in(light: vec3<f32>, start: vec3<f32>, ray: vec3<f32>, shown: bool) -> vec3<f32> {
    return behind_clouds(light, sky_cloud_in(EYE_LAYER, start, ray, shown), ray);
}

fn clouded(light: vec3<f32>, start: vec3<f32>, ray: vec3<f32>) -> vec3<f32> {
    return clouded_in(light, start, ray, true);
}

fn sky_along_in(start: vec3<f32>, direction: vec3<f32>, shown: bool) -> vec3<f32> {
    return clouded_in(sky_air_light(direction), start, direction, shown);
}

fn sky_along(start: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    return sky_along_in(start, direction, true);
}
