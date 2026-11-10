import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};
import package::engine::view::transparency_weighting;

fn transparency(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> f32 {
    return shaded.opacity;
}

fn accumulated(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<f32> {
    let opacity = shaded.opacity;
    let weighting = transparency_weighting();
    let near = point.distance / weighting.x;
    let far = point.distance / weighting.y;
    let far3 = far * far * far;
    let weight = opacity
        * clamp(10.0 / (1.0e-5 + near * near + far3 * far3), weighting.z, weighting.w);
    return vec4<f32>(shaded.linear * opacity, opacity) * weight;
}

fn cover(mask: texture_2d<f32>, pixel: vec2<i32>) -> f32 {
    let covered = min(pixel, vec2<i32>(textureDimensions(mask)) - 1);
    return textureLoad(mask, covered, 0).r;
}
