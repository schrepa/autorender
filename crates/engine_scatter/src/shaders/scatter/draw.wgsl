import package::engine::mesh::declarations::AffineVertexInput;
import package::face::dither::stored;
import package::face::instances::{
    slot_debug, slot_fragment, slot_fragment_extended, slot_masked_depth, slot_pixel,
    slot_pixel_extended, slot_shade, slot_vertex, slot_vertex_extended
};
import package::generated::targets::{ViewTargets, view_targets};
import package::place::debug::DebugTargets;
import package::place::instances::{
    AffinePose, FADE_SLOT, FADE_WORD_ON, SlotColor, SlotExtras, SlotVertex, SlotVertexExtended
};

@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;
@group(2) @binding(2) var<storage, read> grounds: array<vec4<u32>>;

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

fn placed(vertex: SlotVertex, draw: vec4<u32>, slot: u32) -> PlacedVertex {
    return PlacedVertex(vertex.clip_position, vertex.normal_u, vertex.world_v, vertex.material,
        vertex.previous, vec4<u32>(draw.x, slot, 0u, 0u));
}

fn placed_extended(vertex: SlotVertexExtended, draw: vec4<u32>, slot: u32) -> PlacedExtendedVertex {
    return PlacedExtendedVertex(vertex.clip_position, vertex.normal_u, vertex.world_v,
        vertex.material, vertex.previous, vertex.color, vertex.uv1,
        vec4<u32>(draw.x, slot, 0u, 0u));
}

fn slot_of(vertex: PlacedVertex) -> SlotVertex {
    return SlotVertex(vertex.clip_position, vertex.normal_u, vertex.world_v, vertex.material,
        vertex.previous);
}

fn extended_of(vertex: PlacedExtendedVertex) -> SlotVertexExtended {
    return SlotVertexExtended(vertex.clip_position, vertex.normal_u, vertex.world_v,
        vertex.material, vertex.previous, vertex.color, vertex.uv1);
}

fn placed_vertex(input: AffineVertexInput, slot: u32, fade: u32, draw: vec4<u32>) -> PlacedVertex {
    let pose = poses[slot];
    return placed(slot_vertex(input, pose, pose, colors[slot], grounds[slot], fade), draw, slot);
}

fn placed_extended_vertex(input: AffineVertexInput, extras: SlotExtras, slot: u32, fade: u32,
        draw: vec4<u32>) -> PlacedExtendedVertex {
    let pose = poses[slot];
    return placed_extended(slot_vertex_extended(input, extras, pose, pose, colors[slot],
        grounds[slot], fade), draw, slot);
}

fn lit(vertex: SlotVertex, front: bool, pretested: bool) -> ViewTargets {
    var kept = slot_shade(slot_fragment(vertex), front, pretested);
    kept.color = stored(kept.color, vertex.clip_position.xy);
    return view_targets(kept, slot_pixel(vertex));
}

fn lit_extended(vertex: SlotVertexExtended, front: bool, pretested: bool) -> ViewTargets {
    var kept = slot_shade(slot_fragment_extended(vertex), front, pretested);
    kept.color = stored(kept.color, vertex.clip_position.xy);
    return view_targets(kept, slot_pixel_extended(vertex));
}

@vertex
fn vs_object(input: AffineVertexInput, @location(14) draw: vec4<u32>) -> PlacedVertex {
    return placed_vertex(input, input.slot, 0u, draw);
}

@vertex
fn vs_object_fading(input: AffineVertexInput, @location(14) draw: vec4<u32>) -> PlacedVertex {
    let slot = input.slot & FADE_SLOT;
    return placed_vertex(input, slot, (input.slot >> 24u) | FADE_WORD_ON, draw);
}

@vertex
fn vs_object_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>, @location(14) draw: vec4<u32>) -> PlacedExtendedVertex {
    return placed_extended_vertex(input, SlotExtras(color, uv1), input.slot, 0u, draw);
}

@vertex
fn vs_object_extended_fading(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>, @location(14) draw: vec4<u32>) -> PlacedExtendedVertex {
    let slot = input.slot & FADE_SLOT;
    return placed_extended_vertex(input, SlotExtras(color, uv1), slot,
        (input.slot >> 24u) | FADE_WORD_ON, draw);
}

@fragment
fn fs_object(input: PlacedVertex, @builtin(front_facing) front: bool) -> ViewTargets {
    return lit(slot_of(input), front, false);
}

@fragment
fn fs_object_pretested(input: PlacedVertex, @builtin(front_facing) front: bool) -> ViewTargets {
    return lit(slot_of(input), front, true);
}

@fragment
fn fs_object_extended(input: PlacedExtendedVertex, @builtin(front_facing) front: bool)
        -> ViewTargets {
    return lit_extended(extended_of(input), front, false);
}

@fragment
fn fs_object_extended_pretested(input: PlacedExtendedVertex,
        @builtin(front_facing) front: bool) -> ViewTargets {
    return lit_extended(extended_of(input), front, true);
}

@fragment
fn fs_masked_depth(input: PlacedVertex) {
    slot_masked_depth(slot_fragment(slot_of(input)));
}

@fragment
fn fs_masked_depth_extended(input: PlacedExtendedVertex) {
    slot_masked_depth(slot_fragment_extended(extended_of(input)));
}

@fragment
fn fs_debug_object(input: PlacedVertex, @builtin(front_facing) front: bool) -> DebugTargets {
    return slot_debug(slot_fragment(slot_of(input)), front, input.object.x, input.object.y,
        input.object.z);
}

@fragment
fn fs_debug_object_extended(input: PlacedExtendedVertex, @builtin(front_facing) front: bool)
        -> DebugTargets {
    return slot_debug(slot_fragment_extended(extended_of(input)), front, input.object.x,
        input.object.y, input.object.z);
}
