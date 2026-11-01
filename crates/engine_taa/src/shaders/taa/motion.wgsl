import package::engine::shading::{Shaded, ViewPixel};

fn motion(shaded: Shaded, at: ViewPixel) -> vec2<f32> {
    var moved = (at.previous.xy / at.previous.w - at.clip.xy / at.clip.w) * vec2<f32>(0.5, -0.5);
    if !(at.previous.w > 0.0) {
        moved = vec2<f32>(2.0);
    }
    return moved;
}
