// Screen effects (decision 021) shared by the composite of post.wgsl and the
// temporal resolve of temporal.wgsl, which is prepended with this source:
// the frame's effect settings and how both apply ambient occlusion and bloom.

struct Post {
    // Unjittered clip to world space: positions from the depth buffer.
    inverse_view_projection: mat4x4<f32>,
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
