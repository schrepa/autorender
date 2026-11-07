import package::engine::shading::SurfacePoint;
import package::place::sun_blockers::SunAt;

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

