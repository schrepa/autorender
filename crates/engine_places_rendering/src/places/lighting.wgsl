import package::engine::shading::SurfacePoint;
import package::place::sun_blockers::SunAt;

const SURFACE_THIN: u32 = 0u;

struct Reflecting {
    light: vec3<f32>,
    share: f32,
    across: f32,
}

fn highlights() -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn specular() -> f32 {
    return 0.0;
}

fn specular_widening() -> vec2<f32> {
    return vec2<f32>(0.0);
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

fn surface_sun(point: SurfacePoint) -> SunAt {
    var at: SunAt;
    return at;
}

fn sun_visibility(at: SunAt) -> f32 {
    return 1.0;
}

fn sun_debug(at: SunAt) -> u32 {
    return 0u;
}

fn reflecting(point: SurfacePoint) -> Reflecting {
    var result: Reflecting;
    return result;
}

fn given_reflectance(point: SurfacePoint, reflectance: vec3<f32>, diffuse: vec3<f32>)
        -> SurfacePoint {
    return point;
}
