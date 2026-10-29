import package::frame::view::stable_normalize;
import package::meshes::declarations::{AffineVertexInput, VertexOutput, slot_index};
import package::meshes::surface::octahedral_coordinates;
import package::meshes::vertex::vertex_affine;

const DEBUG_KIND_TERRAIN: u32 = 6u;
const DEBUG_KIND_COVER: u32 = 7u;
const DEBUG_KIND_WATER: u32 = 8u;
const DEBUG_KIND_FALLS: u32 = 10u;
const DEBUG_KIND_SKY: u32 = 11u;
const DEBUG_LEVEL_SHIFT: u32 = 4u;
const DEBUG_FADING: u32 = 0x100u;
const DEBUG_SOURCE_SHIFT: u32 = 16u;
const DEBUG_SUN_NONE: u32 = 0u;
const DEBUG_SUN_LEVEL: u32 = 1u;
const DEBUG_SUN_HORIZON: u32 = 21u;
const DEBUG_SUN_BEYOND: u32 = 22u;
const DEBUG_SHADOWED: u32 = 0x80u;
const DEBUG_TILE_EDGE: u32 = 0x800u;
const DEBUG_PATCH_EDGE: u32 = 0x1000u;
const DEBUG_PATCH_SHIFT: u32 = 24u;
const DEBUG_NO_MATERIAL: u32 = 0xffu;
const DEBUG_SPAN_STEPS: f32 = 254.0;

struct DebugTargets {
    @location(0) ids: vec4<u32>,
    @location(1) count: f32,
    @location(2) terrain: vec2<u32>,
}

fn debug_targets(record: u32, object: u32, detail: u32, normal: vec3<f32>, sun: u32)
        -> DebugTargets {
    var targets: DebugTargets;
    targets.ids = vec4<u32>(record | (sun << DEBUG_SOURCE_SHIFT), object, detail,
        pack2x16snorm(octahedral_coordinates(stable_normalize(normal))));
    targets.count = 1.0;
    targets.terrain = vec2<u32>(0u);
    return targets;
}

struct DebugOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) @interpolate(flat) object: vec4<u32>,
}

fn debug_output(vertex: VertexOutput, color: vec4<f32>, uv1: vec2<f32>, object: vec4<u32>)
        -> DebugOutput {
    var output: DebugOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.color = color;
    output.uv1 = uv1;
    output.object = object;
    return output;
}

@vertex
fn vs_debug_affine(input: AffineVertexInput, @location(14) draw: vec4<u32>) -> DebugOutput {
    return debug_output(vertex_affine(input), vec4<f32>(1.0), input.uv,
        vec4<u32>(draw.x, slot_index(input.slot), 0u, 0u));
}

@vertex
fn vs_debug_affine_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>, @location(14) draw: vec4<u32>) -> DebugOutput {
    return debug_output(vertex_affine(input), color, uv1,
        vec4<u32>(draw.x, slot_index(input.slot), 0u, 0u));
}
