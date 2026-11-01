@group(0) @binding(0) var transparent_revealed: texture_2d<f32>;
@group(0) @binding(0) var<uniform> transparent_drawn: vec4<u32>;

fn revealed(pixel: vec2<i32>) -> vec2<f32> {
    if transparent_drawn.x == 0u {
        return vec2<f32>(1.0);
    }
    let covered = min(pixel, vec2<i32>(textureDimensions(transparent_revealed)) - 1);
    return textureLoad(transparent_revealed, covered, 0).xy;
}
