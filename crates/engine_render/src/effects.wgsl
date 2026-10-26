// Screen effects shared by the composite of
// post.wgsl, the haze's march of haze.wgsl and the temporal resolve of
// temporal.wgsl, which are prepended with this source: the frame's effect
// settings and how the composite and resolve apply ambient occlusion,
// reflections, the haze and bloom. Frames draw their screen-sized targets
// at their render size, from the top-left corner, and the half-resolution
// ones at half of it: every read keeps within what the frame drew.

struct Post {
    // Unjittered clip to world space: positions from the depth buffer.
    inverse_view_projection: mat4x4<f32>,
    // Unjittered world to clip space: reflection rays onto the screen.
    view_projection: mat4x4<f32>,
    // xyz: the eye; w: render pixels per world unit at distance 1.
    eye: vec4<f32>,
    // The render size in pixels, and its inverse: the part of the
    // screen-sized targets the frame draws.
    size: vec4<f32>,
    // The screen-sized targets' size in pixels, the display's, and its
    // inverse.
    extent: vec4<f32>,
    // Ambient occlusion: its radius in world units, its strength, the
    // finest occupancy map's texel over the meters around the detail eye it
    // holds, which it hands over by (`occlusion_share`), and 1 when it runs.
    occlusion: vec4<f32>,
    // Strength and 1 when bloom runs; zw: size of a half-resolution texel in
    // texture coordinates.
    bloom: vec4<f32>,
    // Ray distance, strength, 1 when screen-space reflections run, and which
    // pixel of each 2x2 block the half-resolution reflections trace (0 to 3,
    // changing every frame with temporal antialiasing).
    reflections: vec4<f32>,
    // The haze: the turn of its march's jitter this frame, the
    // half-resolution texels across and down its history holds (0 without
    // one), and 1 when it runs.
    haze: vec4<f32>,
    // Last frame's unjittered world to clip space and its eye (xyz), which
    // the haze's history reprojects through.
    previous_view_projection: mat4x4<f32>,
    previous_eye: vec4<f32>,
    // The unit direction toward the sun; w: 1 when 8-bit color writes
    // dither.
    sun_direction: vec4<f32>,
    // Sky occlusion: the share of hidden sky's light taken away, the least
    // share any point keeps, and (w) 1 when it runs: the occlusion pass
    // then folds it into the occlusion's visibility.
    sky: vec4<f32>,
    // The detail eye ambient occlusion hands over by its distance from
    // (xyz); w unused.
    detail: vec4<f32>,
}

// The texels of the half-resolution targets the frame draws: half its
// render size, a texel at least.
fn half_size(settings: Post) -> vec2<i32> {
    return max(vec2<i32>(settings.size.xy) / 2, vec2<i32>(1));
}

// Texture coordinates in a half-resolution target `allocated` texels large of
// `uv` across the view (0 to 1), kept half a texel within the texels the
// frame drew.
fn half_uv(uv: vec2<f32>, allocated: vec2<u32>, settings: Post) -> vec2<f32> {
    let drawn = vec2<f32>(half_size(settings));
    return clamp(uv * drawn, vec2<f32>(0.5), drawn - 0.5) / vec2<f32>(allocated);
}

// How near a half-resolution texel's surface must lie to a pixel's, as a
// share of the pixel's inverse distance from the eye, for the pixel to
// take the texel's effects.
const EFFECTS_PLANE: f32 = 0.1;

// The inverse distance from the eye of what `depth` (reversed: 0 where
// only sky is) shows at `pixel`; 0 for sky, which lies infinitely far.
fn inverse_distance_at(depth: texture_depth_2d, pixel: vec2<i32>, settings: Post) -> f32 {
    let value = textureLoad(depth, pixel, 0);
    if value <= 0.0 {
        return 0.0;
    }
    let uv = (vec2<f32>(pixel) + 0.5) * settings.size.zw;
    let point = settings.inverse_view_projection
        * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, value, 1.0);
    return 1.0 / max(length(point.xyz / point.w - settings.eye.xyz), 1.0e-4);
}

// A half-resolution effect at full-resolution `pixel`, from `effects`, a
// target whose texels hold the inverse distance of their surface in w (as
// the filters of post.wgsl write it): the four texels around the pixel,
// weighted bilinearly and by how near their surface lies to the pixel's own
// (from `depth`), so no effect darkens or lights across a silhouette; where
// none lies near, the bilinear weights alone.
fn effect_at(effects: texture_2d<f32>, depth: texture_depth_2d, pixel: vec2<i32>,
        settings: Post) -> vec4<f32> {
    let size = half_size(settings);
    let position = (vec2<f32>(pixel) + 0.5) * vec2<f32>(size) / settings.size.xy - 0.5;
    let first = vec2<i32>(floor(position));
    let f = position - floor(position);
    let last = size - 1;
    let own = inverse_distance_at(depth, pixel, settings);
    let tolerance = max(EFFECTS_PLANE * own, 1.0e-7);
    var near = vec4<f32>(0.0);
    var near_total = 0.0;
    var plain = vec4<f32>(0.0);
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let texel = textureLoad(effects, clamp(first + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            let bilinear = select(1.0 - f.x, f.x, x == 1) * select(1.0 - f.y, f.y, y == 1);
            let weight = bilinear * saturate(1.25 - abs(texel.w - own) / tolerance);
            near += texel * weight;
            near_total += weight;
            plain += texel * bilinear;
        }
    }
    if near_total < 1.0e-3 {
        return plain;
    }
    return near / near_total;
}

// The ambient occlusion's visibility at full-resolution `pixel`, with what
// sky occlusion leaves of the sky's light, from the half-resolution filter
// of post.wgsl in `occlusion`; 1 without either.
fn occlusion_at(occlusion: texture_2d<f32>, depth: texture_depth_2d, pixel: vec2<i32>,
        settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 && settings.sky.w == 0.0 {
        return 1.0;
    }
    return effect_at(occlusion, depth, pixel, settings).x;
}

// The haze's exposed light at full-resolution `pixel`, from its march
// filtered over time in `hazed`; none without it.
fn haze_at(hazed: texture_2d<f32>, depth: texture_depth_2d, pixel: vec2<i32>,
        settings: Post) -> vec3<f32> {
    if settings.haze.w == 0.0 {
        return vec3<f32>(0.0);
    }
    return effect_at(hazed, depth, pixel, settings).rgb;
}

// The scene reflected at `uv` across the view replacing the share of `color`
// its reflectivity and confidence give it; misses keep the sky reflection
// already in `color`. The composite and the resolve apply it to the shaded
// scene before the effects that add light (`effect_light`); the resolve
// reflects each render pixel's color before blending it with the history,
// which then gathers the pixels each 2x2 block traces in turn.
fn reflected(color: vec3<f32>, reflections: texture_2d<f32>, linear: sampler,
        uv: vec2<f32>, settings: Post) -> vec3<f32> {
    if settings.reflections.z == 0.0 {
        return color;
    }
    let reflection = textureSampleLevel(reflections, linear,
        half_uv(uv, textureDimensions(reflections), settings), 0.0);
    return mix(color, reflection.rgb, saturate(reflection.a * settings.reflections.y));
}

// `color` with the haze's exposed light `light` laid over it: light added
// as a screen blend, and light taken away (where the haze's drifting air is
// clearer than the mean `atmospheric` lit) as a multiply, each share
// compressed as x / (1 + |x|), so neither passes white or black.
fn with_haze(color: vec3<f32>, light: vec3<f32>, settings: Post) -> vec3<f32> {
    if settings.haze.w == 0.0 {
        return color;
    }
    let share = light / (vec3<f32>(1.0) + abs(light));
    let added = vec3<f32>(1.0) - (vec3<f32>(1.0) - color) * (vec3<f32>(1.0) - share);
    return select(color * (vec3<f32>(1.0) + share), added, share >= vec3<f32>(0.0));
}

// Bloom light at `uv` over `color`, as a screen blend that never passes white.
fn glowing(color: vec3<f32>, bloom: texture_2d<f32>, linear: sampler, uv: vec2<f32>,
        settings: Post) -> vec3<f32> {
    if settings.bloom.y == 0.0 {
        return color;
    }
    let glow = max(textureSampleLevel(bloom, linear, uv, 0.0).rgb * settings.bloom.x,
        vec3<f32>(0.0));
    let light = glow / (vec3<f32>(1.0) + glow);
    return vec3<f32>(1.0) - (vec3<f32>(1.0) - color) * (vec3<f32>(1.0) - light);
}

// The share of a pixel's light the ambient occlusion leaves: its ambient
// share `ambient` (the scene target's alpha) darkened by the occlusion's
// visible share `visible` (`occlusion_at`), which holds what sky occlusion
// leaves of the sky's light too.
fn effect_shade(ambient: f32, visible: f32, settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 && settings.sky.w == 0.0 {
        return 1.0;
    }
    return 1.0 - ambient * (1.0 - visible);
}

// `color` with the effects that add light laid over it at `uv` across the
// view: the haze's light `light` (`haze_at`), then bloom. Both are smooth
// and reach the canvas only: the composite lays them over the reflected
// scene, the resolve over its blend with the history.
fn effect_light(color: vec3<f32>, light: vec3<f32>, bloom: texture_2d<f32>,
        linear: sampler, uv: vec2<f32>, settings: Post) -> vec3<f32> {
    let lit = with_haze(color, light, settings);
    return glowing(lit, bloom, linear, half_uv(uv, textureDimensions(bloom), settings), settings);
}
