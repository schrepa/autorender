import package::engine::post::pixel::PostPixel;

@group(0) @binding(0) var resolved_color: texture_2d<f32>;

fn resolved(at: PostPixel) -> vec4<f32> {
    return textureLoad(resolved_color, vec2<i32>(at.position), 0);
}
