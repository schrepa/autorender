import package::engine::materials::{MAP_METALLIC_ROUGHNESS, MAP_OCCLUSION, VALUE_OCCLUSION_STRENGTH,
    map_present, map_uv, material_value, metallic_roughness_map, metallic_roughness_sampler,
    occlusion_map, occlusion_sampler};
import package::engine::shading::{SurfaceInput, SurfacePoint};

fn metal_rough(input: SurfaceInput) -> vec4<f32> {
    if !map_present(MAP_METALLIC_ROUGHNESS) {
        return vec4<f32>(1.0);
    }
    return textureSampleBias(metallic_roughness_map, metallic_roughness_sampler,
        map_uv(MAP_METALLIC_ROUGHNESS, input.uv, input.uv1), input.bias);
}

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.surface.x * metal_rough(input).g;
}

fn metalness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return input.surface.y * metal_rough(input).b;
}

fn occlusion(input: SurfaceInput, point: SurfacePoint) -> f32 {
    if !map_present(MAP_OCCLUSION) {
        return 1.0;
    }
    let sampled = textureSampleBias(occlusion_map, occlusion_sampler,
        map_uv(MAP_OCCLUSION, input.uv, input.uv1), input.bias).r;
    return 1.0 + material_value(VALUE_OCCLUSION_STRENGTH) * (sampled - 1.0);
}
