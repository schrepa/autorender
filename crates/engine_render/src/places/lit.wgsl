import package::engine::shading::{Shaded, SurfacePoint};

alias LightInputs = u32;

alias LightOutputs = u32;

fn widened(point: SurfacePoint, turn_dx: vec3<f32>, turn_dy: vec3<f32>) -> SurfacePoint {
    return point;
}

fn finish(point: SurfacePoint, pixel: vec2<f32>, added: vec3<f32>) -> Shaded {
    var shaded: Shaded;
    return shaded;
}

fn lit_debug(point: SurfacePoint) -> u32 {
    return 0u;
}

fn lit_card(point: SurfacePoint, thickness: f32, seen: f32) -> SurfacePoint {
    return point;
}
