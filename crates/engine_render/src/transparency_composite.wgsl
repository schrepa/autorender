import package::engine::transparency::cover;

@group(2) @binding(0) var transparency_cover: texture_2d<f32>;
@group(2) @binding(1) var transparency_sum: texture_2d<f32>;

@fragment
fn fs_transparency(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let covered = cover(transparency_cover, pixel);
    if covered <= 0.0 {
        discard;
    }
    let held = min(pixel, vec2<i32>(textureDimensions(transparency_sum)) - 1);
    let sum = textureLoad(transparency_sum, held, 0);
    return vec4<f32>(sum.rgb / max(sum.a, 1.0e-5), covered);
}
