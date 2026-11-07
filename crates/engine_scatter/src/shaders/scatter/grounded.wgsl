import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::standard::base_color as standard_base_color;
import package::engine::standard::metalness as standard_metalness;
import package::engine::standard::normal as standard_normal;
import package::engine::standard::roughness as standard_roughness;
import package::scatter::grounding::{grounded_color, grounded_metalness, grounded_normal,
    grounded_roughness};

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return grounded_color(input, standard_base_color(input, point));
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return grounded_normal(input, standard_normal(input, point));
}

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return grounded_roughness(input, standard_roughness(input, point));
}

fn metalness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return grounded_metalness(input, standard_metalness(input, point));
}
