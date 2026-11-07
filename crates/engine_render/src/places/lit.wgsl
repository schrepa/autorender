import package::engine::shading::{Shaded, SurfacePoint};

fn local_light(point: SurfacePoint, pixel: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn widened(point: SurfacePoint, turn_dx: vec3<f32>, turn_dy: vec3<f32>) -> SurfacePoint {
    return point;
}

fn finish(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    var shaded: Shaded;
    return shaded;
}

fn lit_debug(point: SurfacePoint) -> u32 {
    return 0u;
}

fn lit_toward(point: SurfacePoint) -> vec3<f32> {
    return vec3<f32>(0.0);
}
