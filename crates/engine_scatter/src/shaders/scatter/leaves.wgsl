import package::engine::materials::{MAP_NORMAL, VALUE_BRIGHTNESS_VARIATION, VALUE_CROWN_SHADE,
    VALUE_CROWN_SOFTNESS, VALUE_HUE_VARIATION, VALUE_LIGHT_THROUGH, map_present, map_uv,
    material_value, normal_map, normal_sampler, normal_scale};
import package::engine::math::PI;
import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::mapped::base_color as mapped_base_color;
import package::engine::mapped::cutoff as mapped_cutoff;
import package::engine::mapped::{base_texel, deflected_normal, mask_coverage};
import package::engine::view::{fade_noise, stable_normalize};
import package::face::foliage::foliage;
import package::face::lighting::SURFACE_THIN;
import package::generated::kinds::KIND_PLACED;
import package::scatter::grounded::{mapped_metalness, mapped_roughness};
import package::scatter::grounding::{grounded_color, grounded_metalness, grounded_normal,
    grounded_roughness, prepare_grounding};
import package::scatter::values::placed_values_of;

fn prepare(input: SurfaceInput, point: SurfacePoint) {
    prepare_grounding(input);
}

const GREY: vec3<f32> = vec3<f32>(0.57735027);

struct PlantLook {
    through: f32,
    softness: f32,
    hue: f32,
    brightness: f32,
}

fn stated_look() -> PlantLook {
    return PlantLook(material_value(VALUE_LIGHT_THROUGH), material_value(VALUE_CROWN_SOFTNESS),
        material_value(VALUE_HUE_VARIATION), material_value(VALUE_BRIGHTNESS_VARIATION));
}

fn carried_look(packed: vec2<u32>) -> PlantLook {
    let first = unpack2x16unorm(packed.x);
    let second = unpack2x16unorm(packed.y);
    return PlantLook(first.x, first.y, second.x, second.y);
}

fn varied(color: vec3<f32>, plant: u32, look: PlantLook) -> vec3<f32> {
    let rolls = vec2<f32>(f32(plant >> 24u), f32((plant >> 16u) & 255u)) / 127.5 - 1.0;
    let share = foliage().y;
    let angle = look.hue * share * rolls.x * PI * 2.0;
    let turned = color * cos(angle) + cross(GREY, color) * sin(angle)
        + GREY * (dot(GREY, color) * (1.0 - cos(angle)));
    return max(turned, vec3<f32>(0.0)) * (1.0 + look.brightness * share * rolls.y);
}

fn light_through(look: PlantLook, crown: f32) -> f32 {
    return look.through * foliage().x * crown;
}

fn plant_color(input: SurfaceInput, color: vec3<f32>) -> vec3<f32> {
    if input.kind != KIND_PLACED {
        return color;
    }
    return varied(color, placed_values_of(input.kind, input.object).tint, stated_look());
}

fn base_color(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return grounded_color(input, plant_color(input, mapped_base_color(input, point)));
}

fn opacity(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return base_texel(input).a;
}

fn crown_normal(input: SurfaceInput) -> vec3<f32> {
    var shading = stable_normalize(input.normal);
    if map_present(MAP_NORMAL) {
        let uv = map_uv(MAP_NORMAL, input.uv, input.uv1);
        let sampled = textureSampleBias(normal_map, normal_sampler, uv, input.bias).xyz
            * 2.0 - 1.0;
        let plane = cross(input.world_dx, input.world_dy);
        let seen = select(-1.0, 1.0, dot(plane, input.eye - input.world) >= 0.0);
        let face = stable_normalize(plane) * seen;
        let relief = deflected_normal(face, input.world_dx, input.world_dy, dpdx(uv), dpdy(uv),
            sampled, normal_scale()) - face;
        shading = stable_normalize(shading + relief);
    }
    return shading;
}

fn normal(input: SurfaceInput, point: SurfacePoint) -> vec3<f32> {
    return grounded_normal(input, crown_normal(input));
}

fn roughness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return grounded_roughness(input, mapped_roughness(input, point));
}

fn metalness(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return grounded_metalness(input, mapped_metalness(input, point));
}

fn crown_light(input: SurfaceInput) -> f32 {
    return 1.0 - material_value(VALUE_CROWN_SHADE) * (1.0 - input.color.a);
}

fn occlusion(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return crown_light(input);
}

fn transmission(input: SurfaceInput, point: SurfacePoint) -> f32 {
    return light_through(stated_look(), crown_light(input));
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
    return material_value(VALUE_CROWN_SOFTNESS);
}
