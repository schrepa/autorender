import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};

fn transparency(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> f32 {
    return shaded.opacity;
}

fn cover(mask: texture_2d<f32>, pixel: vec2<i32>) -> f32 {
    let covered = min(pixel, vec2<i32>(textureDimensions(mask)) - 1);
    return textureLoad(mask, covered, 0).r;
}
