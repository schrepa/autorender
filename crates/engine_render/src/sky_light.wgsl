// The sky's light along a view from where it starts (`sky_along`): the
// physically based sky's air from its sky-view table, or the analytic sky's
// haze, behind the clouds' shown domes, each read where the view meets
// their layer. The sky pass draws it behind the sun's disc and the night
// sky (`meshes/sky.wgsl`), the sky's environment map holds it
// (`environment.wgsl`), and smooth surfaces mirror it exactly, handing over
// to that map as they roughen (`meshes/shading.wgsl`'s `mirrored_sky`).
// Prepended after scene.wgsl and sky_common.wgsl to the mesh and
// environment shaders, whose first groups bind what it reads at these
// numbers (`bindings::CAMERA`, `bindings::ENVIRONMENT_SKY`).

// The physically based sky's uniform, its sky-view table and their sampler;
// the clouds' newer and older shown domes (clear ones without clouds), their
// sampler and their uniform.
@group(0) @binding(23) var<uniform> sky_air: SkyUniform;
@group(0) @binding(24) var sky_view_table: texture_2d<f32>;
@group(0) @binding(25) var sky_table_sampler: sampler;
@group(0) @binding(26) var sky_clouds: texture_2d<f32>;
@group(0) @binding(27) var sky_clouds_old: texture_2d<f32>;
@group(0) @binding(28) var sky_cloud_sampler: sampler;
@group(0) @binding(29) var<uniform> sky_domes: CloudDomes;

// Whether the sky is the physically based one.
fn physical_sky() -> bool {
    return lighting.grade.z > 0.5;
}

// The sky's own light along `direction`, without the sun's disc: the
// physically based sky's air from its sky-view table, or the analytic sky's
// haze.
fn sky_air_light(direction: vec3<f32>) -> vec3<f32> {
    if physical_sky() {
        return air_sky(sky_view_table, sky_table_sampler, sky_air, direction);
    }
    return haze(direction);
}

// A dome at `uv` through four filtered taps half a texel apart: a tent
// over the texels around, which softens the noise of the march's per-ray
// offsets where thin, brightly lit edges change from texel to texel.
fn dome_at(dome: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let half_texel = 0.5 / vec2<f32>(textureDimensions(dome));
    return 0.25 * (textureSampleLevel(dome, sky_cloud_sampler, uv - half_texel, 0.0)
        + textureSampleLevel(dome, sky_cloud_sampler, uv + half_texel, 0.0)
        + textureSampleLevel(dome, sky_cloud_sampler, uv + vec2<f32>(half_texel.x, -half_texel.y), 0.0)
        + textureSampleLevel(dome, sky_cloud_sampler, uv + vec2<f32>(-half_texel.x, half_texel.y), 0.0));
}

// The clouds along `ray` from `start`: their light and the share of the sky
// passing, the newer dome blending in over the older, each looked up where
// it sees what the ray does.
fn sky_cloud(start: vec3<f32>, ray: vec3<f32>) -> vec4<f32> {
    if sky_domes.older.w < 0.5 {
        return dome_at(sky_clouds, cloud_dome_uv(ray));
    }
    let newer = dome_at(sky_clouds,
        cloud_dome_uv(cloud_dome_direction(sky_domes, sky_domes.newer.xyz, start, ray)));
    let older = dome_at(sky_clouds_old,
        cloud_dome_uv(cloud_dome_direction(sky_domes, sky_domes.older.xyz, start, ray)));
    return mix(older, newer, sky_domes.newer.w);
}

// `light` from the sky along `ray` from `start` behind the clouds' domes.
fn clouded(light: vec3<f32>, start: vec3<f32>, ray: vec3<f32>) -> vec3<f32> {
    return behind_clouds(light, sky_cloud(start, ray), ray);
}

// The sky's light along `direction` from `start`: its air's behind its
// clouds, without the sun's disc, whose highlights surfaces shade
// themselves.
fn sky_along(start: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    return clouded(sky_air_light(direction), start, direction);
}
