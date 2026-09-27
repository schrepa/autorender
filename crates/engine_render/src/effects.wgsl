// Screen effects (decisions 021, 024 and 025) shared by the composite of
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
    // Ray distance, strength, 1 when screen-space reflections run, unused.
    reflections: vec4<f32>,
    // Strength, forward scattering, gathering distance, and 1 when sun
    // shafts run.
    shafts: vec4<f32>,
    // The sun's exposed color: light the shafts scatter.
    sun: vec4<f32>,
}

// The factor darkening a pixel whose light has an ambient share of
// `ambient` (the scene target's alpha) by the half-resolution occlusion at
// `uv`. Four bilinear taps around the pixel blur the samples' noise over
// three texels.
fn occlusion_factor(ambient: f32, occlusion: texture_2d<f32>, linear: sampler,
        uv: vec2<f32>, settings: Post) -> f32 {
    if settings.occlusion.w == 0.0 {
        return 1.0;
    }
    let t = settings.bloom.zw * 0.5;
    let visible = (textureSampleLevel(occlusion, linear, uv - t, 0.0).r
        + textureSampleLevel(occlusion, linear, uv + vec2<f32>(t.x, -t.y), 0.0).r
        + textureSampleLevel(occlusion, linear, uv + vec2<f32>(-t.x, t.y), 0.0).r
        + textureSampleLevel(occlusion, linear, uv + t, 0.0).r) * 0.25;
    return 1.0 - ambient * (1.0 - visible);
}

// The scene reflected at `uv` (decision 025) replacing the share of `color`
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

// Sunlight scattered toward the eye at `uv` (decision 024), blurred over
// three half-resolution texels, added as a screen blend.
fn shafted(color: vec3<f32>, shafts: texture_2d<f32>, linear: sampler, uv: vec2<f32>,
        settings: Post) -> vec3<f32> {
    if settings.shafts.w == 0.0 {
        return color;
    }
    let t = settings.bloom.zw * 0.5;
    let amount = (textureSampleLevel(shafts, linear, uv - t, 0.0).r
        + textureSampleLevel(shafts, linear, uv + vec2<f32>(t.x, -t.y), 0.0).r
        + textureSampleLevel(shafts, linear, uv + vec2<f32>(-t.x, t.y), 0.0).r
        + textureSampleLevel(shafts, linear, uv + t, 0.0).r) * 0.25;
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

fn encode_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

// One triangle covering the viewport: (-1, -1), (3, -1) and (-1, 3).
@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}
