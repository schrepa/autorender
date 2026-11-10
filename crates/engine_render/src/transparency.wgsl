import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};

const WEIGHT_NEAR: f32 = 5.0;
const WEIGHT_FAR: f32 = 200.0;
const WEIGHT_LEAST: f32 = 0.01;
const WEIGHT_MOST: f32 = 3000.0;

fn transparency(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> f32 {
    return shaded.opacity;
}

fn accumulated(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<f32> {
    let opacity = shaded.opacity;
    let near = point.distance / WEIGHT_NEAR;
    let far = point.distance / WEIGHT_FAR;
    let far3 = far * far * far;
    let weight = opacity
        * clamp(10.0 / (1.0e-5 + near * near + far3 * far3), WEIGHT_LEAST, WEIGHT_MOST);
    return vec4<f32>(shaded.linear * opacity, opacity) * weight;
}

fn cover(mask: texture_2d<f32>, pixel: vec2<i32>) -> f32 {
    let covered = min(pixel, vec2<i32>(textureDimensions(mask)) - 1);
    return textureLoad(mask, covered, 0).r;
}
