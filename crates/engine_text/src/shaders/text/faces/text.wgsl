@group(0) @binding(0) var text_atlas: texture_2d<f32>;

fn glyph_coverage(texel: vec2<i32>) -> f32 {
    let held = clamp(texel, vec2<i32>(0), vec2<i32>(textureDimensions(text_atlas)) - 1);
    return textureLoad(text_atlas, held, 0).r;
}
