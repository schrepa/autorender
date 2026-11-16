import package::engine::drawn::drawn_records;
import package::engine::materials::alpha_test;
import package::engine::geometry::declarations::fade_hidden;
import package::engine::geometry::vertex::output_color;
import package::engine::shading::{Shaded, SurfaceInput, SurfacePoint, ViewPixel};
import package::engine::surface::{masked_out, opacity_cut, surface_point};
import package::engine::view::{camera, material_bias, view_pixel};
import package::face::debug::{DebugTargets, debug_record_with_kind, debug_targets};
import package::face::debug_modes::point_source;
import package::generated::writes::{Outputs, blended_outputs, outputs, slot_shaded};

struct SurfaceVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
}

struct SurfaceExtraVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
}

struct ShownVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

struct ShownExtraVaryings {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

struct KindFragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    material: vec4<u32>,
    pixel: vec2<f32>,
    kind: u32,
    object: u32,
}

struct KindShaded {
    point: SurfacePoint,
    shaded: Shaded,
}

struct KindPoint {
    point: SurfacePoint,
    pixel: vec2<f32>,
    at: ViewPixel,
}

fn surface_input(input: KindFragment, front: bool) -> SurfaceInput {
    var surface: SurfaceInput;
    surface.color = input.color;
    surface.normal = input.normal;
    surface.world = input.world;
    surface.world_dx = dpdx(input.world);
    surface.world_dy = dpdy(input.world);
    surface.uv = input.uv;
    surface.uv1 = input.uv1;
    surface.surface = unpack4x8unorm(input.material.z);
    surface.pixel = input.pixel;
    surface.front = front;
    surface.eye = camera.position.xyz;
    surface.bias = material_bias();
    surface.time = camera.position.w;
    surface.kind = input.kind;
    surface.object = input.object;
    return surface;
}

fn lit_point(surface: SurfaceInput) -> SurfacePoint {
    let point = surface_point(surface);
    if alpha_test() && masked_out(surface, point) {
        discard;
    }
    return point;
}

fn fragment_values(clip_position: vec4<f32>, normal_u: vec4<f32>, world_v: vec4<f32>,
        material: vec4<u32>) -> KindFragment {
    var values: KindFragment;
    values.color = output_color(material);
    values.normal = normal_u.xyz;
    values.world = world_v.xyz;
    values.uv = vec2<f32>(normal_u.w, world_v.w);
    values.uv1 = values.uv;
    values.material = material;
    values.pixel = clip_position.xy;
    return values;
}

fn extra_values(clip_position: vec4<f32>, normal_u: vec4<f32>, world_v: vec4<f32>,
        material: vec4<u32>, color: vec4<f32>, uv1: vec2<f32>) -> KindFragment {
    var values = fragment_values(clip_position, normal_u, world_v, material);
    values.color *= color;
    values.uv1 = uv1;
    return values;
}

fn surface_fragment(input: SurfaceVaryings) -> KindFragment {
    return fragment_values(input.clip_position, input.normal_u, input.world_v, input.material);
}

fn extra_fragment(input: SurfaceExtraVaryings) -> KindFragment {
    return extra_values(input.clip_position, input.normal_u, input.world_v, input.material,
        input.color, input.uv1);
}

fn object_fragment(values: KindFragment, object: vec4<u32>) -> KindFragment {
    if fade_hidden(values.material.w, values.pixel) {
        discard;
    }
    var placed = values;
    placed.kind = object.w;
    placed.object = object.y;
    return placed;
}

fn shown_object(input: ShownVaryings) -> KindFragment {
    return object_fragment(fragment_values(input.clip_position, input.normal_u, input.world_v,
        input.material), input.object);
}

fn shown_extra_object(input: ShownExtraVaryings) -> KindFragment {
    return object_fragment(extra_values(input.clip_position, input.normal_u, input.world_v,
        input.material, input.color, input.uv1), input.object);
}

fn masked_depth(input: KindFragment) {
    if alpha_test() && opacity_cut(surface_input(input, true)) {
        discard;
    }
}

fn shade(input: KindFragment, front: bool) -> KindShaded {
    let surface = surface_input(input, front);
    let point = lit_point(surface);
    return KindShaded(point, slot_shaded(surface, point, input.pixel));
}

fn drawn_pixel(clip_position: vec4<f32>, previous: vec3<f32>) -> ViewPixel {
    return view_pixel(clip_position, vec4<f32>(previous.xy, 0.0, previous.z));
}

fn shown_targets(input: KindFragment, object: vec4<u32>, front: bool) -> DebugTargets {
    let point = lit_point(surface_input(input, front));
    let record = debug_record_with_kind(object.x, drawn_records[object.y].kind);
    return debug_targets(record, object.y, object.z, point.normal, point_source(0u, point));
}

fn object_targets(input: KindFragment, object: vec4<u32>, front: bool) -> DebugTargets {
    let point = lit_point(surface_input(input, front));
    return debug_targets(debug_record_with_kind(object.x, object.w), object.y, object.z,
        point.normal, point_source(0u, point));
}

@fragment
fn fs_kind_masked_depth(input: SurfaceVaryings) {
    masked_depth(surface_fragment(input));
}

@fragment
fn fs_kind_masked_depth_extra(input: SurfaceExtraVaryings) {
    masked_depth(extra_fragment(input));
}

@fragment
fn fs_kind_lit(input: SurfaceVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(surface_fragment(input), front);
    return outputs(shaded.point, shaded.shaded, drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_kind_lit_extra(input: SurfaceExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(extra_fragment(input), front);
    return outputs(shaded.point, shaded.shaded, drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_kind_blended(input: SurfaceVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(surface_fragment(input), front);
    return blended_outputs(shaded.point, shaded.shaded,
        drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_kind_blended_extra(input: SurfaceExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(extra_fragment(input), front);
    return blended_outputs(shaded.point, shaded.shaded,
        drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_kind_shown(input: ShownVaryings, @builtin(front_facing) front: bool) -> DebugTargets {
    let values = fragment_values(input.clip_position, input.normal_u, input.world_v,
        input.material);
    return shown_targets(values, input.object, front);
}

@fragment
fn fs_kind_shown_extra(input: ShownExtraVaryings, @builtin(front_facing) front: bool)
        -> DebugTargets {
    let values = extra_values(input.clip_position, input.normal_u, input.world_v,
        input.material, input.color, input.uv1);
    return shown_targets(values, input.object, front);
}

@fragment
fn fs_kind_object_masked_depth(input: ShownVaryings) {
    masked_depth(shown_object(input));
}

@fragment
fn fs_kind_object_masked_depth_extra(input: ShownExtraVaryings) {
    masked_depth(shown_extra_object(input));
}

@fragment
fn fs_kind_object_lit(input: ShownVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(shown_object(input), front);
    return outputs(shaded.point, shaded.shaded, drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_kind_object_lit_extra(input: ShownExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(shown_extra_object(input), front);
    return outputs(shaded.point, shaded.shaded, drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_kind_object_blended(input: ShownVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(shown_object(input), front);
    return blended_outputs(shaded.point, shaded.shaded,
        drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_kind_object_blended_extra(input: ShownExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(shown_extra_object(input), front);
    return blended_outputs(shaded.point, shaded.shaded,
        drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_kind_object_shown(input: ShownVaryings, @builtin(front_facing) front: bool)
        -> DebugTargets {
    return object_targets(shown_object(input), input.object, front);
}

@fragment
fn fs_kind_object_shown_extra(input: ShownExtraVaryings, @builtin(front_facing) front: bool)
        -> DebugTargets {
    return object_targets(shown_extra_object(input), input.object, front);
}
