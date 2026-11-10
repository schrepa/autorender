import package::engine::materials::{MAP_METALLIC_ROUGHNESS, map_present, map_uv,
    metallic_roughness_map, metallic_roughness_sampler};
import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::mapped::base_color as mapped_base_color;
import package::engine::mapped::normal as mapped_normal;
import package::scatter::grounding::{grounded_color, grounded_metalness, grounded_normal,
    grounded_roughness};

fn metal_rough(input: SurfaceInput) -> vec4<f32> {
    if !map_present(MAP_METALLIC_ROUGHNESS) {
        return vec4<f32>(1.0);
    }
    return textureSampleBias(metallic_roughness_map, metallic_roughness_sampler,
        map_uv(MAP_METALLIC_ROUGHNESS, input.uv, input.uv1), input.bias);
}

fn mapped_roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.surface.x * metal_rough(input).g;
}

fn mapped_metalness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.surface.y * metal_rough(input).b;
}

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
