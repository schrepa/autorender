import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::mapped::base_color as mapped_base_color;
import package::engine::mapped::cutoff as mapped_cutoff;
import package::engine::mapped::mask_coverage;
import package::engine::mapped::normal as mapped_normal;
import package::engine::view::fade_noise;
import package::face::foliage::foliage;
import package::face::lighting::SURFACE_THIN;
import package::scatter::grounded::{mapped_metalness, mapped_roughness};
import package::scatter::grounding::{grounded_color, grounded_metalness, grounded_normal,
    grounded_roughness};
import package::scatter::values::placed_values_of;

fn leafiness(color: vec3<f32>) -> f32 {
    return saturate(4.0 * (color.g - max(color.r, color.b)) / max(color.g, 1.0e-4));
}

fn plant_tint(plant: u32, leaf: f32) -> vec3<f32> {
    let strength = foliage().y * leaf;
    let along = f32(plant >> 24u) / 255.0;
    let gradient = mix(vec3<f32>(1.16, 1.05, 0.64), vec3<f32>(0.80, 0.97, 1.08), along);
    return mix(vec3<f32>(1.0), gradient, strength);
}

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    let color = mapped_base_color(input, point);
    let plant = placed_values_of(input.kind, input.object).tint;
    return grounded_color(input, color * plant_tint(plant, leafiness(color)));
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

fn transmission(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return foliage().x * leafiness(mapped_base_color(input, point));
}

fn cutoff(input: SurfaceInput, point: SurfacePoint) -> f32 {
    if foliage().z == 0.0 {
        return mapped_cutoff(input, point);
    }
    return max(fade_noise(input.pixel), 1.0 / 255.0) / mask_coverage(input);
}

fn kind(input: SurfaceInput, point: SurfacePoint) -> u32 {
    return SURFACE_THIN;
}

fn direct_softness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return leafiness(mapped_base_color(input, point));
}
