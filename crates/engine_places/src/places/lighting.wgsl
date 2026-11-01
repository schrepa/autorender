import package::engine::shading::{Shaded, SurfacePoint, flat};

fn highlights() -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn specular() -> f32 {
    return 0.0;
}

fn distribution(n_h: f32, alpha: f32) -> f32 {
    return 0.0;
}

fn visibility(n_v: f32, n_l: f32, alpha: f32) -> f32 {
    return 0.0;
}

fn fresnel(f0: vec3<f32>, cosine: f32) -> vec3<f32> {
    return f0;
}

fn environment_brdf(f0: vec3<f32>, roughness: f32, n_v: f32) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn sun_visibility(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        distance: f32, on_terrain: bool, crown: f32) -> f32 {
    return 1.0;
}

fn sun_debug(world: vec3<f32>, normal: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>,
        on_terrain: bool) -> u32 {
    return 0u;
}

fn local_light(point: SurfacePoint, pixel: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn widened(point: SurfacePoint, turn_dx: vec3<f32>, turn_dy: vec3<f32>) -> SurfacePoint {
    return point;
}

fn finish(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    return flat(vec4<f32>(point.albedo, point.opacity));
}

fn finish_ground(point: SurfacePoint, local: vec3<f32>) -> Shaded {
    return flat(vec4<f32>(point.albedo, point.opacity));
}
