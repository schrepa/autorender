import package::engine::shading::{Shaded, ViewPixel};
import package::engine::view::camera;

fn motion(shaded: Shaded, at: ViewPixel) -> vec2<f32> {
    let current = vec2<f32>(at.clip.x * camera.jitter.z - 1.0,
        1.0 - at.clip.y * camera.jitter.w) - camera.jitter.xy;
    var moved = (at.previous.xy / at.previous.w - current) * vec2<f32>(0.5, -0.5);
    if !(at.previous.w > 0.0) {
        moved = vec2<f32>(2.0);
    }
    return moved;
}
