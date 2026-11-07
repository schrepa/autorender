import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};

fn normals(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> vec4<f32> {
    return vec4<f32>(point.normal * 0.5 + 0.5, 1.0);
}
