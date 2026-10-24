// Screen effects shared by the composite of
// post.wgsl, the sun shafts of shafts.wgsl and the temporal resolve of
// temporal.wgsl, which are prepended with this source: the frame's effect
// settings and how the composite and resolve apply ambient occlusion,
// reflections, sun shafts and bloom. Frames draw their screen-sized targets
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
    // Strength, forward scattering, gathering distance, and 1 when sun
    // shafts run.
    shafts: vec4<f32>,
    // The sun's exposed color: light the shafts scatter.
    sun: vec4<f32>,
    // Contact shadows: each ray's reach in meters, the share
    // of sunlight a hit takes away, the distance within which surfaces
    // trace, and 1 when they run.
    contact: vec4<f32>,
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

// The full-resolution pixel quarter-resolution sun shaft texel `texel`
// traces: the second of each 4x4 block's diagonal, centered
// at 4 * texel + 1.5.
fn shaft_pixel(texel: vec2<i32>) -> vec2<i32> {
    return texel * 4 + vec2<i32>(1);
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

// The ambient occlusion's visibility (x), the sun shafts' light (y) and the
// contact shadows' visibility (z) at full-resolution `pixel`, which the
// half-resolution filter of post.wgsl writes together with the inverse
// distance of each texel's surface (w): the four texels around the pixel,
// weighted bilinearly and by how near their surface lies to the pixel's
// own (from `depth`), so no effect darkens or lights across a silhouette;
// where none lies near, the bilinear weights alone.
fn screen_effects(effects: texture_2d<f32>, depth: texture_depth_2d, pixel: vec2<i32>,
        settings: Post) -> vec3<f32> {
    if settings.occlusion.w == 0.0 && settings.shafts.w == 0.0 && settings.contact.w == 0.0
        && settings.sky.w == 0.0 {
        return vec3<f32>(1.0, 0.0, 1.0);
    }
    let size = half_size(settings);
    let position = (vec2<f32>(pixel) + 0.5) * vec2<f32>(size) / settings.size.xy - 0.5;
    let first = vec2<i32>(floor(position));
    let f = position - floor(position);
    let last = size - 1;
    let own = inverse_distance_at(depth, pixel, settings);
    let tolerance = max(EFFECTS_PLANE * own, 1.0e-7);
    var near = vec3<f32>(0.0);
    var near_total = 0.0;
    var plain = vec3<f32>(0.0);
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let texel = textureLoad(effects, clamp(first + vec2<i32>(x, y), vec2<i32>(0), last), 0);
            let bilinear = select(1.0 - f.x, f.x, x == 1) * select(1.0 - f.y, f.y, y == 1);
            let weight = bilinear * saturate(1.25 - abs(texel.w - own) / tolerance);
            near += texel.rgb * weight;
            near_total += weight;
            plain += texel.rgb * bilinear;
        }
    }
    if near_total < 1.0e-3 {
        return plain;
    }
    return near / near_total;
}

// The factor darkening a pixel whose light has an ambient share of
// `ambient` (the scene target's alpha) by the occlusion's `visible` share,
// which holds what sky occlusion leaves of the sky's light too.
fn occlusion_factor(ambient: f32, visible: f32, settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 && settings.sky.w == 0.0 {
        return 1.0;
    }
    return 1.0 - ambient * (1.0 - visible);
}

// The factor darkening the rest of that light, the sun's, by the contact
// shadows' `visible` share.
fn contact_factor(ambient: f32, visible: f32, settings: Post) -> f32 {
    if settings.contact.w == 0.0 {
        return 1.0;
    }
    return 1.0 - (1.0 - ambient) * (1.0 - visible);
}

// The scene reflected at `uv` replacing the share of `color`
// its reflectivity and confidence give it; misses keep the sky reflection
// already in `color`.
fn reflected(color: vec3<f32>, reflections: texture_2d<f32>, linear: sampler,
        uv: vec2<f32>, settings: Post) -> vec3<f32> {
    if settings.reflections.z == 0.0 {
        return color;
    }
    let reflection = textureSampleLevel(reflections, linear, uv, 0.0);
    return mix(color, reflection.rgb, saturate(reflection.a * settings.reflections.y));
}

// Sunlight scattered toward the eye, `amount` of it, added as
// a screen blend.
fn shafted(color: vec3<f32>, amount: f32, settings: Post) -> vec3<f32> {
    if settings.shafts.w == 0.0 {
        return color;
    }
    let light = settings.sun.rgb * max(amount, 0.0);
    let added = light / (vec3<f32>(1.0) + light);
    return vec3<f32>(1.0) - (vec3<f32>(1.0) - color) * (vec3<f32>(1.0) - added);
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

// The share of a pixel's light the ambient occlusion and the contact
// shadows leave: its ambient share `ambient` (the scene target's alpha)
// darkened by the occlusion, the rest by the contact shadows, as
// `screen_effects` gives them in `effects`.
fn effect_shade(ambient: f32, effects: vec3<f32>, settings: Post) -> f32 {
    return occlusion_factor(ambient, effects.x, settings)
        * contact_factor(ambient, effects.z, settings);
}

// `color` with the effects that add light laid over it at `uv` across the
// view: the reflections, the sun shafts' light in `effects`, then bloom.
fn effect_light(color: vec3<f32>, effects: vec3<f32>, reflections: texture_2d<f32>,
        bloom: texture_2d<f32>, linear: sampler, uv: vec2<f32>, settings: Post) -> vec3<f32> {
    let mirrored = reflected(color, reflections, linear,
        half_uv(uv, textureDimensions(reflections), settings), settings);
    let lit = shafted(mirrored, effects.y, settings);
    return glowing(lit, bloom, linear, half_uv(uv, textureDimensions(bloom), settings), settings);
}
