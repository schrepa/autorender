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
    // The far plane: depth 0 with reversed depth.
    output.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    output.ndc = ndc;
    return output;
}

// The clouds' newer and older shown domes, their
// sampler and their uniform: clear ones without clouds.
@group(1) @binding(38) var sky_clouds: texture_2d<f32>;
@group(1) @binding(39) var sky_cloud_sampler: sampler;
@group(1) @binding(44) var sky_clouds_old: texture_2d<f32>;
@group(1) @binding(45) var<uniform> sky_domes: CloudDomes;
// The physically based sky: its uniform, sky-view and
// transmittance tables and their sampler.
@group(1) @binding(40) var<uniform> sky_air: SkyUniform;
@group(1) @binding(41) var sky_view_table: texture_2d<f32>;
@group(1) @binding(42) var sky_transmittance_table: texture_2d<f32>;
@group(1) @binding(43) var sky_table_sampler: sampler;

// Whether the sky is the physically based one.
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
// what the ray does.
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

// The sky along `ray` behind its clouds, with the night sky's; under the
// sea, the water's light. While the view marches its own clouds, they
// cover it after it draws.
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
    light = with_night_sky(light, ray);
    if sky_domes.view.x > 0.5 {
        return light;
    }
    return behind_clouds(light, sky_cloud(ray), ray);
}

// The night sky's stars: cells along each side of each face of the cube
// they are hashed on, the share of cells holding one, the light of the
// brightest, and the sky's luminance above which none shows (they fade in
// below it, fully from two fifths of it).
const STAR_CELLS: f32 = 160.0;
const STAR_SHARE: f32 = 0.4;
const STAR_LIGHT: f32 = 0.25;
const STAR_SKY: f32 = 0.1;
// The moon's angular radius (radians: twice its true quarter degree, the
// size the eye reads it at) and the light of its full face.
const MOON_RADIUS: f32 = 0.0091;
const MOON_LIGHT: f32 = 0.6;
// Cirrus: the layer's height and how far along a view it fades out by
// (world units), and the world units its strands drift each second.
const CIRRUS_HEIGHT: f32 = 9000.0;
const CIRRUS_REACH: f32 = 90000.0;
const CIRRUS_DRIFT: vec2<f32> = vec2<f32>(6.0, 1.5);

// `direction` turned by the unit quaternion `turn`.
fn turned(turn: vec4<f32>, direction: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(turn.xyz, direction);
    return direction + turn.w * t + cross(turn.xyz, t);
}

// Three values in [0, 1) hashed from `cell`.
fn sky_hash(cell: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(dither_hash(cell) >> vec3<u32>(8u)) * (1.0 / 16777216.0);
}

// Smooth value noise in [0, 1] over the plane, its lattice salted by
// `salt`.
fn sky_noise(p: vec2<f32>, salt: u32) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let u = f * f * (3.0 - 2.0 * f);
    let base = vec2<u32>(vec2<i32>(cell) & vec2<i32>(0xffff));
    let a = sky_hash(vec3<u32>(base, salt)).x;
    let b = sky_hash(vec3<u32>(base + vec2<u32>(1u, 0u), salt)).x;
    let c = sky_hash(vec3<u32>(base + vec2<u32>(0u, 1u), salt)).x;
    let d = sky_hash(vec3<u32>(base + vec2<u32>(1u, 1u), salt)).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// The stars' light along `star`, a direction in their own frame: a star in
// STAR_SHARE of the cells of a cube's faces, placed within its cell, its
// brightness a high power of a uniform value so faint stars far outnumber
// bright ones, bluish white to warm, drawn as a soft spot of `spread`
// radians (about a pixel) so it neither vanishes nor shimmers between
// pixels.
fn star_light(star: vec3<f32>, spread: f32) -> vec3<f32> {
    let a = abs(star);
    var axis = 0u;
    var uv = star.yz / a.x;
    if a.y >= a.x && a.y >= a.z {
        axis = 1u;
        uv = star.xz / a.y;
    } else if a.z >= a.x {
        axis = 2u;
        uv = star.xy / a.z;
    }
    let face = axis * 2u + select(0u, 1u, star[axis] < 0.0);
    let position = (uv * 0.5 + 0.5) * STAR_CELLS;
    let cell = floor(position);
    let hashed = sky_hash(vec3<u32>(vec2<u32>(cell), face));
    if hashed.x >= STAR_SHARE {
        return vec3<f32>(0.0);
    }
    let center = cell + 0.25 + 0.5 * hashed.yz;
    // A cell spans about a quarter turn over STAR_CELLS.
    let offset = (position - center) * (1.5707963 / STAR_CELLS);
    let spot = exp(-dot(offset, offset) / (2.0 * spread * spread));
    let bright = pow(hashed.x / STAR_SHARE, 8.0);
    let warm = fract(hashed.y * 7.31 + hashed.z * 3.17);
    let tint = mix(vec3<f32>(0.78, 0.86, 1.0), vec3<f32>(1.0, 0.86, 0.68), warm);
    return tint * (STAR_LIGHT * bright * spot);
}

// The moon along `ray`, standing along `moon` and lit by the sun along
// `sun`: its face as a sphere lit where the sun reaches it, which gives its
// phase, its seas darker, and earthshine faint on the rest; rgb its light,
// a how much of the ray its disc covers (soft at the rim).
fn moon_disc(ray: vec3<f32>, moon: vec3<f32>, sun: vec3<f32>) -> vec4<f32> {
    if dot(ray, moon) < cos(MOON_RADIUS * 1.1) {
        return vec4<f32>(0.0);
    }
    let right = normalize(cross(moon, vec3<f32>(1.0e-4, 1.0, 0.0)));
    let up = cross(right, moon);
    let local = vec2<f32>(dot(ray, right), dot(ray, up)) / MOON_RADIUS;
    let r2 = dot(local, local);
    let cover = 1.0 - smoothstep(0.92, 1.0, sqrt(r2));
    if cover <= 0.0 {
        return vec4<f32>(0.0);
    }
    // The near half's normal faces the eye.
    let normal = local.x * right + local.y * up - sqrt(max(1.0 - r2, 0.0)) * moon;
    let lit = saturate(dot(normal, sun) * 2.0);
    let seas = mix(0.6, 1.0, smoothstep(0.3, 0.7,
        0.65 * sky_noise(local * 2.2 + 7.0, 3u) + 0.35 * sky_noise(local * 5.3 + 2.0, 4u)));
    let light = vec3<f32>(1.0, 0.96, 0.9) * (seas * (lit + 0.02) * MOON_LIGHT);
    return vec4<f32>(light, cover);
}

// How much cirrus covers along `ray`: fibrous noise on a layer
// CIRRUS_HEIGHT above the eye, stretched along x and drifting with
// `seconds`, over `coverage` of the sky, fading toward the horizon.
fn cirrus_cover(ray: vec3<f32>, coverage: f32, seconds: f32) -> f32 {
    if ray.y <= 0.02 {
        return 0.0;
    }
    let along = CIRRUS_HEIGHT / ray.y;
    let fade = 1.0 - smoothstep(0.35 * CIRRUS_REACH, CIRRUS_REACH, along);
    var p = (ray.xz * along + CIRRUS_DRIFT * seconds) * vec2<f32>(1.0 / 7000.0, 1.0 / 1700.0);
    var sum = 0.0;
    var weight = 0.5;
    for (var octave = 0u; octave < 4u; octave++) {
        sum += sky_noise(p, 9u + octave) * weight;
        p = p * 2.03 + vec2<f32>(17.0, 31.0);
        weight *= 0.5;
    }
    let shape = saturate((sum / 0.9375 - (1.0 - coverage)) / 0.3);
    return shape * shape * fade * 0.6;
}

// The light cirrus sends along `ray`: the sun's (with the physically based
// sky, what reaches the layer's height, which outlasts sunset there),
// scattered forward toward it; the moon's, faintly; and the sky's.
fn cirrus_light(ray: vec3<f32>) -> vec3<f32> {
    let sun = lighting.sun.xyz;
    var sunlight = lighting.sun_color.rgb;
    if physical_sky() {
        let r = air_radius(sky_air) + CIRRUS_HEIGHT * 0.001;
        sunlight = sky_air.ozone.w * air_transmittance(sky_transmittance_table, sky_table_sampler,
            sky_air, r, sun.y);
    }
    let toward = max(dot(ray, sun), 0.0);
    let forward = 0.25 + 1.4 * pow(toward, 6.0);
    let moonlight = vec3<f32>(0.8, 0.85, 1.0) * (0.02 * MOON_LIGHT * lighting.night.y
        * lighting.night.w * saturate(lighting.moon.y * 4.0));
    return sunlight * (0.3 * forward) + moonlight + 0.5 * (lighting.zenith.rgb + lighting.horizon.rgb);
}

// `light`, the sky's along `ray`, with the night sky's: stars where the sky
// in front of them is dark and the moon, both beyond the air and fading
// into the horizon's haze, then cirrus over both.
fn with_night_sky(light: vec3<f32>, ray: vec3<f32>) -> vec3<f32> {
    let night = lighting.night;
    if all(night.xyz == vec3<f32>(0.0)) {
        return light;
    }
    var beyond = vec3<f32>(0.0);
    if night.x > 0.0 {
        var dark = 1.0 - smoothstep(0.4 * STAR_SKY, STAR_SKY, luminance(light));
        // The physically based sky's sun is the real one: no stars while it
        // is up, all of them once it is six degrees down.
        if physical_sky() {
            dark *= 1.0 - smoothstep(-0.1, 0.0, lighting.sun.y);
        }
        let spread = max(0.8 * camera.options.y, 1.0e-5);
        beyond = star_light(turned(lighting.star_turn, ray), spread) * (night.x * dark);
    }
    if night.y > 0.0 {
        let disc = moon_disc(ray, lighting.moon.xyz, lighting.sun.xyz);
        beyond = mix(beyond, disc.rgb * night.y, disc.a);
    }
    var sky = light + beyond * smoothstep(-0.01, 0.12, ray.y);
    if night.z > 0.0 {
        let cover = cirrus_cover(ray, night.z, camera.position.w);
        if cover > 0.0 {
            sky = mix(sky, cirrus_light(ray), cover);
        }
    }
    return sky;
}

// The view ray through unjittered `ndc`.
fn sky_ray(ndc: vec2<f32>) -> vec3<f32> {
    let far = camera.inverse_view_projection * vec4<f32>(ndc, 0.0, 1.0);
    return stable_normalize(far.xyz / far.w - camera.position.xyz);
}

@fragment
fn fs_sky(input: SkyOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(grade(clouded_sky(sky_ray(input.ndc))), input.clip_position.xy),
        1.0);
}

// Reflection probe captures: the sky's haze without the sun's
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

// Clouds in the view (`clouds.wgsl`'s `march_view`): each
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

// The view's clouds over the canvas, encoded as it is; pixels they leave
// clear keep the target as it was, unblended.
@fragment
fn fs_view_clouds(input: SkyOutput) -> @location(0) vec4<f32> {
    let layer = view_cloud_layer(input.clip_position.xy);
    let covering = 1.0 - layer.a;
    if covering <= 1.0e-4 {
        discard;
    }
    return vec4<f32>(encoded(layer.rgb / covering, input.clip_position.xy) * covering, layer.a);
}

// The view's clouds over the screen effects' scene target.
@fragment
fn fs_view_clouds_post(input: SkyOutput) -> @location(0) vec4<f32> {
    let layer = view_cloud_layer(input.clip_position.xy);
    if layer.a >= 1.0 - 1.0e-4 {
        discard;
    }
    if !dithering() {
        return layer;
    }
    let covering = 1.0 - layer.a;
    return vec4<f32>(stored(layer.rgb / covering, input.clip_position.xy) * covering, layer.a);
}

// Screen-effect frames: the sky has no ambient occlusion; its sun glows.
@fragment
fn fs_sky_post(input: SkyOutput) -> SceneTargets {
    let light = clouded_sky(sky_ray(input.ndc));
    var targets: SceneTargets;
    targets.color = vec4<f32>(stored(grade(light), input.clip_position.xy), 0.0);
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
    targets.color = vec4<f32>(stored(grade(light), input.clip_position.xy), 0.0);
    targets.glow = vec2<f32>(glow_weight(light), 0.0);
    targets.surface = SKY_SURFACE;
    let previous = camera.previous_view_projection * vec4<f32>(ray, 0.0);
    targets.motion = (previous.xy / previous.w - input.ndc) * vec2<f32>(0.5, -0.5);
    if !(previous.w > 0.0) {
        targets.motion = vec2<f32>(2.0);
    }
    return targets;
}
