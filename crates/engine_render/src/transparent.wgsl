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
