import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::mapped::base_color as mapped_base_color;
import package::engine::mapped::metalness as mapped_metalness;
import package::engine::mapped::normal as mapped_normal;
import package::engine::mapped::roughness as mapped_roughness;
import package::scatter::grounding::{grounded_color, grounded_metalness, grounded_normal,
    grounded_roughness};

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return grounded_color(input, mapped_base_color(input, point));
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return grounded_normal(input, mapped_normal(input, point));
}

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return grounded_roughness(input, mapped_roughness(input, point));
}

fn metalness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return grounded_metalness(input, mapped_metalness(input, point));
}
