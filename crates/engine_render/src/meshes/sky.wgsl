// The sky behind all geometry: one triangle over the viewport at the far
// plane, drawn after the meshes so depth testing leaves only uncovered pixels.
struct SkyOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) index: u32) -> SkyOutput {
    let ndc = vec2<f32>(f32((index << 1u) & 2u) * 2.0 - 1.0, f32(index & 2u) * 2.0 - 1.0);
    var output: SkyOutput;
    // The far plane: depth 0 with reversed depth (decision 042).
    output.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    output.ndc = ndc;
    return output;
}

// The clouds' newer and older shown domes (decisions 084 and 091), their
// sampler and their uniform: clear ones without clouds.
@group(1) @binding(38) var sky_clouds: texture_2d<f32>;
@group(1) @binding(39) var sky_cloud_sampler: sampler;
@group(1) @binding(44) var sky_clouds_old: texture_2d<f32>;
@group(1) @binding(45) var<uniform> sky_domes: CloudDomes;
// The physically based sky (decision 088): its uniform, sky-view and
// transmittance tables and their sampler.
@group(1) @binding(40) var<uniform> sky_air: SkyUniform;
@group(1) @binding(41) var sky_view_table: texture_2d<f32>;
@group(1) @binding(42) var sky_transmittance_table: texture_2d<f32>;
@group(1) @binding(43) var sky_table_sampler: sampler;

// Whether the sky is the physically based one (decision 088).
fn physical_sky() -> bool {
    return lighting.grade.z > 0.5;
}

// The sky's light along `ray` from its tables, with the sun's disc when
// `disc`.
fn air_light_along(ray: vec3<f32>, disc: bool) -> vec3<f32> {
    return air_sky(sky_view_table, sky_transmittance_table, sky_table_sampler, sky_air, ray, disc);
}

// The clouds along `ray`: their light and the share of the sky passing,
// the newer dome blending in over the older, each looked up where it sees
// what the ray does (decision 091).
fn sky_cloud(ray: vec3<f32>) -> vec4<f32> {
    if sky_domes.older.w < 0.5 {
        return textureSampleLevel(sky_clouds, sky_cloud_sampler, cloud_dome_uv(ray), 0.0);
    }
    let newer = textureSampleLevel(sky_clouds, sky_cloud_sampler,
        cloud_dome_uv(cloud_dome_direction(sky_domes, sky_domes.newer.xyz, ray)), 0.0);
    let older = textureSampleLevel(sky_clouds_old, sky_cloud_sampler,
        cloud_dome_uv(cloud_dome_direction(sky_domes, sky_domes.older.xyz, ray)), 0.0);
    return mix(older, newer, sky_domes.newer.w);
}

// The sky along `ray` behind its clouds; under the sea, the water's light.
// While the view marches its own clouds (decision 095), they cover it
// after it draws.
fn clouded_sky(ray: vec3<f32>) -> vec3<f32> {
    if lighting.underwater.w > 0.0 {
        return sky(ray);
    }
    var light: vec3<f32>;
    if physical_sky() {
        light = air_light_along(ray, true);
    } else {
        light = sky(ray);
    }
    if sky_domes.view.x > 0.5 {
        return light;
    }
    return behind_clouds(light, sky_cloud(ray), ray);
}

// The view ray through unjittered `ndc`.
fn sky_ray(ndc: vec2<f32>) -> vec3<f32> {
    let far = camera.inverse_view_projection * vec4<f32>(ndc, 0.0, 1.0);
    return stable_normalize(far.xyz / far.w - camera.position.xyz);
}

@fragment
fn fs_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(grade(clouded_sky(sky_ray(input.ndc)))), 1.0);
}

// Reflection probe captures (decision 043): the sky's haze without the sun's
// disc, as the sky's environment map holds it (surfaces shade the sun's
// highlights themselves).
@fragment
fn fs_probe_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    let ray = sky_ray(input.ndc);
    var light: vec3<f32>;
    if physical_sky() {
        light = air_light_along(ray, false);
    } else {
        light = haze(ray);
    }
    return vec4<f32>(behind_clouds(light, sky_cloud(ray), ray), 1.0);
}

// The sky's surface texel: facing up, fully rough (it reflects nothing).
const SKY_SURFACE = vec4<f32>(0.5, 0.5, 1.0, 0.0);

// Clouds in the view (decision 095; `clouds.wgsl`'s `march_view`): each
// marched block's light and the share of the scene passing, how far its
// clouds lie, the depth buffer and the blocks' sampler.
@group(1) @binding(46) var view_clouds: texture_2d<f32>;
@group(1) @binding(47) var view_cloud_front: texture_2d<f32>;
@group(1) @binding(48) var view_cloud_depth: texture_depth_2d;
@group(1) @binding(49) var view_cloud_sampler: sampler;

// The view's clouds over the pixel at `position` (framebuffer): the
// blocks' light and passing share, fading out where they lie beyond what
// the pixel shows, its block having marched past it.
fn view_clouds_at(position: vec2<f32>) -> vec4<f32> {
    let pixels = vec2<f32>(textureDimensions(view_cloud_depth));
    let uv = position / pixels;
    // Four filtered taps half a block apart: a tent over the blocks around,
    // softening their edges and the noise of their rays' offsets.
    let half_block = 0.5 / vec2<f32>(textureDimensions(view_clouds));
    let cloud = 0.25 * (
        textureSampleLevel(view_clouds, view_cloud_sampler, uv - half_block, 0.0)
        + textureSampleLevel(view_clouds, view_cloud_sampler, uv + half_block, 0.0)
        + textureSampleLevel(view_clouds, view_cloud_sampler,
            uv + vec2<f32>(half_block.x, -half_block.y), 0.0)
        + textureSampleLevel(view_clouds, view_cloud_sampler,
            uv + vec2<f32>(-half_block.x, half_block.y), 0.0));
    let depth = textureLoad(view_cloud_depth, vec2<i32>(position), 0);
    if depth <= 0.0 {
        return cloud;
    }
    let blocks = vec2<f32>(textureDimensions(view_cloud_front));
    let block = vec2<i32>(min(floor(uv * blocks), blocks - 1.0));
    let front = textureLoad(view_cloud_front, block, 0).r;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let point = camera.inverse_view_projection * vec4<f32>(ndc, depth, 1.0);
    let distance = length(point.xyz / point.w - camera.position.xyz);
    let shown = saturate((distance - 0.8 * front) / max(0.2 * front, 1.0));
    return mix(vec4<f32>(0.0, 0.0, 0.0, 1.0), cloud, shown);
}

// The view's clouds as a layer over the scene: their own light, graded,
// times the share they cover, with the share passing as alpha; none under
// the sea.
fn view_cloud_layer(position: vec2<f32>) -> vec4<f32> {
    let cloud = view_clouds_at(position);
    let covering = 1.0 - cloud.a;
    if covering <= 1.0e-4 || lighting.underwater.w > 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(grade(cloud.rgb / covering) * covering, cloud.a);
}

// The view's clouds over the canvas, encoded as it is.
@fragment
fn fs_view_clouds(input: SkyOutput) -> @location(0) vec4<f32> {
    let layer = view_cloud_layer(input.clip_position.xy);
    let covering = 1.0 - layer.a;
    if covering <= 1.0e-4 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(encoded(layer.rgb / covering) * covering, layer.a);
}

// The view's clouds over the screen effects' scene target.
@fragment
fn fs_view_clouds_post(input: SkyOutput) -> @location(0) vec4<f32> {
    return view_cloud_layer(input.clip_position.xy);
}

// Screen-effect frames: the sky has no ambient occlusion; its sun glows.
@fragment
fn fs_sky_post(input: SkyOutput) -> SceneTargets {
    let light = clouded_sky(sky_ray(input.ndc));
    var targets: SceneTargets;
    targets.color = vec4<f32>(grade(light), 0.0);
    targets.glow = vec2<f32>(glow_weight(light), 0.0);
    targets.surface = SKY_SURFACE;
    return targets;
}

// Temporal frames: the unjittered sky reprojects by the camera's turn alone.
@fragment
fn fs_sky_temporal(input: SkyOutput) -> TemporalTargets {
    let ray = sky_ray(input.ndc);
    let light = clouded_sky(ray);
    var targets: TemporalTargets;
    targets.color = vec4<f32>(grade(light), 0.0);
    targets.glow = vec2<f32>(glow_weight(light), 0.0);
    targets.surface = SKY_SURFACE;
    let previous = camera.previous_view_projection * vec4<f32>(ray, 0.0);
    targets.motion = (previous.xy / previous.w - input.ndc) * vec2<f32>(0.5, -0.5);
    if !(previous.w > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}
