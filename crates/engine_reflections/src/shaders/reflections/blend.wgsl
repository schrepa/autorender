import package::engine::transparency::cover;
import package::generated::pieces::reflections::piece;

@group(1) @binding(0) var reflections: texture_2d<f32>;
@group(1) @binding(1) var reflected_target: texture_2d<f32>;
@group(1) @binding(2) var blended_mask: texture_2d<f32>;

@fragment
fn fs_reflections_blend(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let traced = textureLoad(reflections, pixel, 0);
    let replaced = textureLoad(reflected_target, pixel, 0).rgb;
    let seen = (1.0 - cover(blended_mask, pixel)) * piece().applied.y;
    return vec4<f32>((traced.rgb - replaced * traced.a) * seen, 0.0);
}
