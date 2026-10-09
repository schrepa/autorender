// Weighted blended transparency: the composite of the
// accumulated transparent surfaces and particles over the opaque color.
// Blending computes average * (1 - revealed) + opaque * revealed, and keeps
// only the opaque share of the ambient light alpha, which ambient occlusion
// darkens. Pixels nothing transparent covered are discarded.

@group(0) @binding(0) var accumulated: texture_2d<f32>;
@group(0) @binding(1) var revealed: texture_2d<f32>;

// Frames with screen effects composite into the sRGB scene target, which
// encodes as it stores; others into the UNORM canvas, encoded here.
override TARGET_IS_SRGB: bool = false;

// One triangle covering the viewport: (-1, -1), (3, -1) and (-1, 3).
@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn encode_srgb(color: vec3<f32>) -> vec3<f32> {
    let low = color * 12.92;
    let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, color <= vec3<f32>(0.0031308));
}

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let reveal = textureLoad(revealed, pixel, 0).r;
    if reveal >= 1.0 {
        discard;
    }
    let sum = textureLoad(accumulated, pixel, 0);
    var color = min(sum.rgb / max(sum.a, 1.0e-5), vec3<f32>(1.0));
    if !TARGET_IS_SRGB {
        color = encode_srgb(color);
    }
    return vec4<f32>(color, 1.0 - reveal);
}
