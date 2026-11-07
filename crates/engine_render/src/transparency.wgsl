import package::engine::shading::{Shaded, ViewPixel};

fn transparency(shaded: Shaded, at: ViewPixel) -> f32 {
    return shaded.opacity;
}
