import package::surfaces::materials::{SURFACE_EMISSION, alpha_cutoff, emissive_factor};
import package::surfaces::shading::{Shaded, SurfaceInput, SurfacePoint};

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return input.color.rgb;
}

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.color.a;
}

fn cutoff(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return alpha_cutoff();
}

fn kind(input: SurfaceInput, point: SurfacePoint) -> u32 {
    return point.kind;
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return point.geometric;
}

fn emissive(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return point.base_color * (input.surface.z * SURFACE_EMISSION) + emissive_factor();
}

fn output(input: SurfaceInput, point: SurfacePoint, shaded: Shaded) -> Shaded {
    return shaded;
}
