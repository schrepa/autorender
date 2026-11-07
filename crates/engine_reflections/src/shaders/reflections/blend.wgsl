import package::face::transparency::revealed;
import package::generated::pieces::reflections::piece;
import package::generated::targets::{blended_cover, output_reflected};

@group(1) @binding(0) var reflections: texture_2d<f32>;

@fragment
fn fs_reflections_blend(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let traced = textureLoad(reflections, pixel, 0);
    let replaced = textureLoad(output_reflected, pixel, 0).rgb;
    let seen = revealed(pixel).x * (1.0 - blended_cover(pixel)) * piece().applied.y;
    return vec4<f32>((traced.rgb - replaced * traced.a) * seen, 0.0);
}
