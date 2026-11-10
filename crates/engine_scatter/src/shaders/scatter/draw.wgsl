import package::engine::drawn::{GeometryExtra, GeometryVertex, INPUT_DRAW, INPUT_SLOT};
import package::engine::materials::alpha_test;
import package::engine::mesh::declarations::{AffinePose, faded_entry_fade, faded_slot_slot,
    fade_hidden};
import package::engine::mesh::vertex::{output_color, pose_matrix, transform_vertex};
import package::engine::shading::{PositionInput, Shaded, SurfaceInput, SurfacePoint};
import package::engine::slots::{shade, slot_output, slot_position};
import package::engine::surface::{masked_out, opacity_cut, surface_point};
import package::engine::view::{camera, detail_across, material_bias, view_clip, view_pixel};
import package::face::debug::{DebugTargets, debug_record_with_kind, debug_targets};
import package::face::debug_modes::point_source;
import package::generated::kinds::KIND_PLACED;
import package::generated::writes::{Outputs, outputs};

@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<vec4<u32>>;

struct PlacedVertex {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

struct PlacedExtendedVertex {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

struct PlacedShaded {
    point: SurfacePoint,
    shaded: Shaded,
}

fn placed_position(model: mat4x4<f32>, position: vec3<f32>, world: vec3<f32>, slot: u32,
        time: f32) -> vec3<f32> {
    let origin = model[3].xyz;
    var input: PositionInput;
    input.local = position;
    input.world = world;
    input.origin = origin;
    input.time = time;
    input.pixels = 1.0 / max(detail_across(origin), 1.0e-6);
    input.kind = KIND_PLACED;
    input.object = slot;
    return slot_position(input);
}

fn placed_vertex(input: GeometryVertex, slot: u32, fade: u32, draw: vec4<u32>)
        -> PlacedVertex {
    let model = pose_matrix(poses[slot]);
    let color = colors[slot];
    let vertex = transform_vertex(input.position, input.normal, color.xy, color.z, input.uv0,
        model);
    let world = placed_position(model, input.position, vertex.world_v.xyz, slot,
        camera.position.w);
    let before = placed_position(model, input.position, vertex.world_v.xyz, slot,
        camera.previous.w);
    var output: PlacedVertex;
    output.clip_position = view_clip(world);
    output.normal_u = vertex.normal_u;
    output.world_v = vec4<f32>(world, vertex.world_v.w);
    output.material = vec4<u32>(color.xy, color.z, fade);
    output.previous = (camera.previous_view_projection * vec4<f32>(before, 1.0)).xyw;
    output.object = vec4<u32>(draw.x, slot, 0u, 0u);
    return output;
}

fn placed_extended_vertex(input: GeometryVertex, extra: GeometryExtra, slot: u32, fade: u32,
        draw: vec4<u32>) -> PlacedExtendedVertex {
    let vertex = placed_vertex(input, slot, fade, draw);
    return PlacedExtendedVertex(vertex.clip_position, vertex.normal_u, vertex.world_v,
        vertex.material, vertex.previous, extra.color, extra.uv1, vertex.object);
}

fn extended_of(input: PlacedVertex) -> PlacedExtendedVertex {
    return PlacedExtendedVertex(input.clip_position, input.normal_u, input.world_v,
        input.material, input.previous, vec4<f32>(1.0), vec2<f32>(input.normal_u.w,
        input.world_v.w), input.object);
}

fn placed_surface(input: PlacedExtendedVertex, front: bool) -> SurfaceInput {
    if fade_hidden(input.material.w, input.clip_position.xy) {
        discard;
    }
    var surface: SurfaceInput;
    surface.color = output_color(input.material) * input.color;
    surface.normal = input.normal_u.xyz;
    surface.world = input.world_v.xyz;
    surface.world_dx = dpdx(input.world_v.xyz);
    surface.world_dy = dpdy(input.world_v.xyz);
    surface.uv = vec2<f32>(input.normal_u.w, input.world_v.w);
    surface.uv1 = input.uv1;
    surface.surface = unpack4x8unorm(input.material.z);
    surface.pixel = input.clip_position.xy;
    surface.front = front;
    surface.eye = camera.position.xyz;
    surface.bias = material_bias();
    surface.time = camera.position.w;
    surface.kind = KIND_PLACED;
    surface.object = input.object.y;
    return surface;
}

fn placed_point(surface: SurfaceInput) -> SurfacePoint {
    let point = surface_point(surface);
    if alpha_test() && masked_out(surface, point) {
        discard;
    }
    return point;
}

fn placed_shade(input: PlacedExtendedVertex, front: bool) -> PlacedShaded {
    let surface = placed_surface(input, front);
    let point = placed_point(surface);
    return PlacedShaded(point, slot_output(surface, point, shade(point, surface.pixel)));
}

fn lit(input: PlacedExtendedVertex, front: bool) -> Outputs {
    let shaded = placed_shade(input, front);
    return outputs(shaded.point, shaded.shaded, view_pixel(input.clip_position,
        vec4<f32>(input.previous.xy, 0.0, input.previous.z)));
}

fn masked_depth(input: PlacedExtendedVertex) {
    if alpha_test() && opacity_cut(placed_surface(input, true)) {
        discard;
    }
}

fn debug_object(input: PlacedExtendedVertex, front: bool) -> DebugTargets {
    let point = placed_point(placed_surface(input, front));
    return debug_targets(debug_record_with_kind(input.object.x, KIND_PLACED), input.object.y,
        input.object.z, point.normal, point_source(0u, point));
}

@vertex
fn vs_object(input: GeometryVertex, @location(INPUT_SLOT) slot: u32,
        @location(INPUT_DRAW) draw: vec4<u32>) -> PlacedVertex {
    return placed_vertex(input, slot, 0u, draw);
}

@vertex
fn vs_object_fading(input: GeometryVertex, @location(INPUT_SLOT) slot: u32,
        @location(INPUT_DRAW) draw: vec4<u32>) -> PlacedVertex {
    return placed_vertex(input, faded_slot_slot(slot), faded_entry_fade(slot), draw);
}

@vertex
fn vs_object_extended(input: GeometryVertex, extra: GeometryExtra,
        @location(INPUT_SLOT) slot: u32, @location(INPUT_DRAW) draw: vec4<u32>)
        -> PlacedExtendedVertex {
    return placed_extended_vertex(input, extra, slot, 0u, draw);
}

@vertex
fn vs_object_extended_fading(input: GeometryVertex, extra: GeometryExtra,
        @location(INPUT_SLOT) slot: u32, @location(INPUT_DRAW) draw: vec4<u32>)
        -> PlacedExtendedVertex {
    return placed_extended_vertex(input, extra, faded_slot_slot(slot), faded_entry_fade(slot),
        draw);
}

@fragment
fn fs_object(input: PlacedVertex, @builtin(front_facing) front: bool) -> Outputs {
    return lit(extended_of(input), front);
}

@fragment
fn fs_object_extended(input: PlacedExtendedVertex, @builtin(front_facing) front: bool)
        -> Outputs {
    return lit(input, front);
}

@fragment
fn fs_masked_depth(input: PlacedVertex) {
    masked_depth(extended_of(input));
}

@fragment
fn fs_masked_depth_extended(input: PlacedExtendedVertex) {
    masked_depth(input);
}

@fragment
fn fs_debug_object(input: PlacedVertex, @builtin(front_facing) front: bool) -> DebugTargets {
    return debug_object(extended_of(input), front);
}

@fragment
fn fs_debug_object_extended(input: PlacedExtendedVertex, @builtin(front_facing) front: bool)
        -> DebugTargets {
    return debug_object(input, front);
}
