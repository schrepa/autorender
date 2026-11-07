import package::engine::materials::{ALPHA_MASK, MAP_BASE_COLOR, MASK_MIP_SCALE, base_color_map,
    map_uv, material_alpha};
import package::engine::shading::{SurfaceInput, SurfacePoint, derived};
import package::engine::slots::{slot_base_color, slot_cutoff, slot_emissive, slot_kind,
    slot_light_inputs, slot_metalness, slot_normal, slot_occlusion, slot_opacity, slot_roughness,
    slot_transmission};
import package::engine::surfaces::{SURFACE_SOLID, SURFACE_THIN};
import package::engine::view::stable_normalize;

fn geometric_normal(surface_normal: vec3<f32>, front: bool) -> vec3<f32> {
    let geometric = stable_normalize(surface_normal);
    return select(-geometric, geometric, front);
}

fn surface_kind() -> u32 {
    return select(SURFACE_SOLID, SURFACE_THIN, material_alpha() == ALPHA_MASK);
}

fn placed(input: SurfaceInput) -> SurfacePoint {
    var point: SurfacePoint;
    point.world = input.world;
    point.world_dx = input.world_dx;
    point.world_dy = input.world_dy;
    point.geometric = geometric_normal(input.normal, input.front);
    point.normal = point.geometric;
    let to_eye = input.eye - input.world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.kind = surface_kind();
    point.occlusion = 1.0;
    return point;
}

fn surface_point(input: SurfaceInput) -> SurfacePoint {
    var point = placed(input);
    point.base_color = slot_base_color(input, point);
    point.opacity = slot_opacity(input, point);
    point.normal = slot_normal(input, point);
    point.normal_dx = dpdx(point.normal);
    point.normal_dy = dpdy(point.normal);
    point.roughness = slot_roughness(input, point);
    point.metalness = slot_metalness(input, point);
    point = derived(point);
    point.emissive = slot_emissive(input, point);
    point.occlusion = slot_occlusion(input, point);
    point.transmission = slot_transmission(input, point);
    point.kind = slot_kind(input, point);
    point.inputs = slot_light_inputs(input, point);
    return point;
}

fn mask_coverage(input: SurfaceInput) -> f32 {
    let uv = map_uv(MAP_BASE_COLOR, input.uv, input.uv1);
    let size = vec2<f32>(textureDimensions(base_color_map)) * exp2(input.bias);
    let x = dpdx(uv) * size;
    let y = dpdy(uv) * size;
    return 1.0 + MASK_MIP_SCALE * max(0.5 * log2(max(dot(x, x), dot(y, y))), 0.0);
}

fn masked_out(input: SurfaceInput, point: SurfacePoint) -> bool {
    return point.opacity * mask_coverage(input) < slot_cutoff(input, point);
}

fn opacity_cut(input: SurfaceInput) -> bool {
    var point = placed(input);
    point.opacity = slot_opacity(input, point);
    return masked_out(input, point);
}
