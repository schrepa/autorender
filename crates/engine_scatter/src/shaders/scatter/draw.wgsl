import package::engine::drawn::{GeometryExtra, GeometryVertex, INPUT_DRAW, INPUT_SLOT};
import package::engine::mesh::declarations::{AffinePose, faded_entry_fade, faded_slot_slot};
import package::engine::mesh::vertex::{pose_matrix, transform_vertex};
import package::engine::shading::PositionInput;
import package::engine::slots::slot_position;
import package::engine::view::{camera, detail_across, view_clip};
import package::generated::kinds::KIND_PLACED;

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
    output.object = vec4<u32>(draw.x, slot, 0u, KIND_PLACED);
    return output;
}

fn placed_extended_vertex(input: GeometryVertex, extra: GeometryExtra, slot: u32, fade: u32,
        draw: vec4<u32>) -> PlacedExtendedVertex {
    let vertex = placed_vertex(input, slot, fade, draw);
    return PlacedExtendedVertex(vertex.clip_position, vertex.normal_u, vertex.world_v,
        vertex.material, vertex.previous, extra.color, extra.uv1, vertex.object);
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
