import package::engine::materials::material_model;
import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::slots::{derived, open_slots, slot_base_color, slot_cutoff, slot_emissive,
    slot_kind, slot_normal, slot_opacity, standard_point};
import package::engine::surfaces::SURFACE_SOLID;
import package::engine::view::stable_normalize;

fn geometric_normal(surface_normal: vec3<f32>, front: bool) -> vec3<f32> {
    let geometric = stable_normalize(surface_normal);
    return select(-geometric, geometric, front);
}

fn placed(input: SurfaceInput) -> SurfacePoint {
    var point = standard_point();
    point.world = input.world;
    point.world_dx = input.world_dx;
    point.world_dy = input.world_dy;
    point.geometric = geometric_normal(input.normal, input.front);
    point.normal = point.geometric;
    let to_eye = input.eye - input.world;
    point.distance = length(to_eye);
    point.view = stable_normalize(to_eye);
    point.kind = SURFACE_SOLID;
    return point;
}

fn surface_point(input: SurfaceInput) -> SurfacePoint {
    var point = placed(input);
    point.base_color = slot_base_color(input, point);
    point.opacity = slot_opacity(input, point);
    point.normal = slot_normal(input, point);
    point.normal_dx = dpdx(point.normal);
    point.normal_dy = dpdy(point.normal);
    point = open_slots(input, point);
    point = derived(point);
    point.emissive = slot_emissive(input, point);
    point.kind = slot_kind(input, point);
    point.lighting = material_model();
    return point;
}

fn masked_out(input: SurfaceInput, point: SurfacePoint) -> bool {
    return point.opacity < slot_cutoff(input, point);
}

fn opacity_cut(input: SurfaceInput) -> bool {
    var point = placed(input);
    point.opacity = slot_opacity(input, point);
    return masked_out(input, point);
}
