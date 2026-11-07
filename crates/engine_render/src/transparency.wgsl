import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};

fn transparency(point: SurfacePoint, shaded: Shaded, at: ViewPixel) -> f32 {
    return shaded.opacity;
}
