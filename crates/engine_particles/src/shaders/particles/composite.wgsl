@group(1) @binding(0) var accumulated: texture_2d<f32>;
@group(1) @binding(1) var revealed: texture_2d<f32>;

@fragment
fn fs_composite(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let reveal = textureLoad(revealed, pixel, 0).r;
    if reveal >= 1.0 {
        discard;
    }
    let sum = textureLoad(accumulated, pixel, 0);
    return vec4<f32>(sum.rgb / max(sum.a, 1.0e-5), 1.0 - reveal);
}
