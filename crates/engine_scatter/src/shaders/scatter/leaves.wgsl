import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::standard::base_color as standard_base_color;
import package::engine::standard::metalness as standard_metalness;
import package::engine::standard::normal as standard_normal;
import package::engine::standard::roughness as standard_roughness;
import package::face::foliage::foliage;
import package::scatter::grounding::{grounded_color, grounded_metalness, grounded_normal,
    grounded_roughness};

fn leafiness(color: vec3<f32>) -> f32 {
    return saturate(4.0 * (color.g - max(color.r, color.b)) / max(color.g, 1.0e-4));
}

fn plant_tint(plant: u32, leaf: f32) -> vec3<f32> {
    let strength = foliage().y * leaf;
    if strength <= 0.0 {
        return vec3<f32>(1.0);
    }
    let along = f32(plant >> 24u) / 255.0;
    let gradient = mix(vec3<f32>(1.16, 1.05, 0.64), vec3<f32>(0.80, 0.97, 1.08), along);
    return mix(vec3<f32>(1.0), gradient, strength);
}

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    let color = standard_base_color(input, point);
    return grounded_color(input, color * plant_tint(input.object.w, leafiness(color)));
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

fn transmission(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return foliage().x * leafiness(standard_base_color(input, point));
}
