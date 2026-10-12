// Screen effects shared by the composite of
// post.wgsl, the sun shafts of shafts.wgsl and the temporal resolve of
// temporal.wgsl, which are prepended with this source: the frame's effect
// settings and how the composite and resolve apply ambient occlusion,
// reflections, sun shafts and bloom.

struct Post {
    // Unjittered clip to world space: positions from the depth buffer.
    inverse_view_projection: mat4x4<f32>,
    // Unjittered world to clip space: reflection rays onto the screen.
    view_projection: mat4x4<f32>,
    // xyz: the eye; w: depth-buffer pixels per world unit at distance 1.
    eye: vec4<f32>,
    // Scene target size in pixels, and its inverse.
    size: vec4<f32>,
    // Radius in world units, strength, unused, and 1 when ambient occlusion
    // runs.
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
    // The unit direction toward the sun.
    sun_direction: vec4<f32>,
}

// The full-resolution pixel quarter-resolution sun shaft texel `texel`
// traces: the second of each 4x4 block's diagonal, centered
// at 4 * texel + 1.5.
fn shaft_pixel(texel: vec2<i32>) -> vec2<i32> {
    return texel * 4 + vec2<i32>(1);
}

// The ambient occlusion's visibility (x), the sun shafts' light (y) and the
// contact shadows' visibility (z) at `uv`, which the half-resolution filter
// of post.wgsl writes together: four bilinear taps around the pixel smooth
// them over three texels.
fn screen_effects(effects: texture_2d<f32>, linear: sampler, uv: vec2<f32>,
        settings: Post) -> vec3<f32> {
    if settings.occlusion.w == 0.0 && settings.shafts.w == 0.0 && settings.contact.w == 0.0 {
        return vec3<f32>(1.0, 0.0, 1.0);
    }
    let t = settings.bloom.zw * 0.5;
    return (textureSampleLevel(effects, linear, uv - t, 0.0).rgb
        + textureSampleLevel(effects, linear, uv + vec2<f32>(t.x, -t.y), 0.0).rgb
        + textureSampleLevel(effects, linear, uv + vec2<f32>(-t.x, t.y), 0.0).rgb
        + textureSampleLevel(effects, linear, uv + t, 0.0).rgb) * 0.25;
}

// The factor darkening a pixel whose light has an ambient share of
// `ambient` (the scene target's alpha) by the occlusion's `visible` share.
fn occlusion_factor(ambient: f32, visible: f32, settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 {
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

// `color` with the effects that add light laid over it at `uv`: the
// reflections, the sun shafts' light in `effects`, then bloom.
fn effect_light(color: vec3<f32>, effects: vec3<f32>, reflections: texture_2d<f32>,
        bloom: texture_2d<f32>, linear: sampler, uv: vec2<f32>, settings: Post) -> vec3<f32> {
    let mirrored = reflected(color, reflections, linear, uv, settings);
    let lit = shafted(mirrored, effects.y, settings);
    return glowing(lit, bloom, linear, uv, settings);
}
