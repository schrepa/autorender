// Screen effects shared by the composite of
// post.wgsl, the haze's march of haze.wgsl and the temporal resolve of
// temporal.wgsl, which are prepended with this source: the frame's effect
// settings and how the composite and resolve apply ambient occlusion,
// reflections, the haze and bloom. Frames draw their screen-sized targets,
// every effect's but bloom's among them, at their render size from the
// top-left corner, and bloom's levels at their share of it: every read
// keeps within what the frame drew.

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
    // Strength and 1 when bloom runs (zw unused).
    bloom: vec4<f32>,
    // Ray distance, strength and 1 when reflections run, and the terrains
    // their world trace walks (reflections.wgsl).
    reflections: vec4<f32>,
    // The haze: the turn of its march's jitter this frame, the pixels
    // across and down its history holds (0 without one), and 1 when it
    // runs.
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

// Texture coordinates of the screen-sized targets at `pixel` (render
// pixels, continuous), kept half a pixel within the pixels the frame drew.
fn drawn_uv(pixel: vec2<f32>, settings: Post) -> vec2<f32> {
    return clamp(pixel, vec2<f32>(0.5), settings.size.xy - 0.5) * settings.extent.zw;
}

// The texels the frame drew of a target `size` texels large made at a
// fraction of the display's size (a level of a chain): all of them where it
// draws the whole display, else as many as the render size's share of the
// display, rounded down, which its passes draw at least.
fn drawn_texels(size: vec2<f32>, settings: Post) -> vec2<f32> {
    let share = max(floor(size * settings.size.xy * settings.extent.zw), vec2<f32>(1.0));
    return select(share, size, settings.size.xy == settings.extent.xy);
}

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

// The ambient occlusion's visibility at `pixel`, with what sky occlusion
// leaves of the sky's light, from the filter of post.wgsl in `occlusion`;
// 1 without either.
fn occlusion_at(occlusion: texture_2d<f32>, pixel: vec2<i32>, settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 && settings.sky.w == 0.0 {
        return 1.0;
    }
    return textureLoad(occlusion, pixel, 0).x;
}

// The haze's exposed light at `pixel`, from its march filtered over time
// in `hazed`; none without it.
fn haze_at(hazed: texture_2d<f32>, pixel: vec2<i32>, settings: Post) -> vec3<f32> {
    if settings.haze.w == 0.0 {
        return vec3<f32>(0.0);
    }
    return textureLoad(hazed, pixel, 0).rgb;
}

// The scene reflected at `uv` across the view: the reflections there,
// traced at every render pixel and read between them bilinearly,
// premultiplied by how surely their rays met the scene, take away the
// share of `color` its mirrored environment made up (a) and add the
// scene's light the surface mirrors in its place (rgb), both by the
// reflections' strength, within the graded range the scene holds; misses
// keep the sky the surface mirrors, already in `color` (`finish`: exact
// where it is smooth, its environment map's as it roughens). The composite
// and the resolve apply it to the shaded scene before the effects that add
// light (`effect_light`); the resolve reflects each render pixel's color
// before blending it with the history.
fn reflected(color: vec3<f32>, reflections: texture_2d<f32>, linear: sampler,
        uv: vec2<f32>, settings: Post) -> vec3<f32> {
    if settings.reflections.z == 0.0 {
        return color;
    }
    let reflection = textureSampleLevel(reflections, linear,
        drawn_uv(uv * settings.size.xy, settings), 0.0);
    let strength = settings.reflections.y;
    return min(color * (1.0 - saturate(reflection.a * strength)) + reflection.rgb * strength,
        vec3<f32>(1.0));
}

// `color` with the haze's exposed light `light` laid over it: light added
// as a screen blend, and light taken away (where the haze's drifting air is
// clearer than the mean `atmospheric` lit, or sees less of the sky than it
// counts) as a multiply, each share compressed as x / (1 + |x|), so neither
// passes white or black.
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
// view: the haze's light `light` (`haze_at`), then bloom, its first level
// read half a texel within the texels the frame drew of it. Both are smooth
// and reach the canvas only: the composite lays them over the reflected
// scene, the resolve over its blend with the history.
fn effect_light(color: vec3<f32>, light: vec3<f32>, bloom: texture_2d<f32>,
        linear: sampler, uv: vec2<f32>, settings: Post) -> vec3<f32> {
    let lit = with_haze(color, light, settings);
    let size = vec2<f32>(textureDimensions(bloom));
    let drawn = drawn_texels(size, settings);
    return glowing(lit, bloom, linear, clamp(uv * drawn, vec2<f32>(0.5), drawn - 0.5) / size,
        settings);
}
