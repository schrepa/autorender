@group(0) @binding(23) var<uniform> sky_air: SkyUniform;
@group(0) @binding(24) var sky_view_table: texture_2d<f32>;
@group(0) @binding(25) var sky_table_sampler: sampler;
@group(0) @binding(26) var sky_clouds: texture_2d_array<f32>;
@group(0) @binding(27) var sky_clouds_old: texture_2d_array<f32>;
@group(0) @binding(28) var sky_cloud_sampler: sampler;
@group(0) @binding(29) var<uniform> sky_domes: CloudDomes;

const EYE_LAYER: u32 = 0u;
const MIRROR_LAYER: u32 = 1u;

fn physical_sky() -> bool {
    return lighting.grade.z > 0.5;
}

fn sky_air_light(direction: vec3<f32>) -> vec3<f32> {
    if physical_sky() {
        return air_sky(sky_view_table, sky_table_sampler, sky_air, direction);
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
    if sky_domes.older.w < 0.5 {
        return dome_at(sky_clouds, cloud_dome_uv(ray), layer);
    }
    let mirrored = layer == MIRROR_LAYER;
    let newer_origin = select(sky_domes.newer.xyz, sky_domes.newer_mirror.xyz, mirrored);
    let older_origin = select(sky_domes.older.xyz, sky_domes.older_mirror.xyz, mirrored);
    let newer = dome_at(sky_clouds,
        cloud_dome_uv(cloud_dome_direction(sky_domes, newer_origin, start, ray)), layer);
    let older = dome_at(sky_clouds_old,
        cloud_dome_uv(cloud_dome_direction(sky_domes, older_origin, start, ray)), layer);
    return mix(older, newer, sky_domes.newer.w);
}

fn clouded(light: vec3<f32>, start: vec3<f32>, ray: vec3<f32>) -> vec3<f32> {
    return behind_clouds(light, sky_cloud(EYE_LAYER, start, ray), ray);
}

fn sky_along(start: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    return clouded(sky_air_light(direction), start, direction);
}

fn mirrored_along(start: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    return behind_clouds(sky_air_light(direction), sky_cloud(MIRROR_LAYER, start, direction),
        direction);
}
