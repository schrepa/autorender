import package::engine::mesh::declarations::AffineVertexInput;
import package::engine::shading::{Shaded, ViewPixel};
import package::place::debug::DebugTargets;

struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}

struct SlotColor {
    color: vec2<u32>,
    surface: u32,
    flags: u32,
}

struct SlotExtras {
    color: vec4<f32>,
    uv1: vec2<f32>,
}

struct SlotVertex {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(8) @interpolate(flat) ground: vec4<u32>,
    @location(9) @interpolate(flat) origin: vec3<f32>,
}

struct SlotVertexExtended {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    @location(8) @interpolate(flat) ground: vec4<u32>,
    @location(9) @interpolate(flat) origin: vec3<f32>,
}

struct SlotFragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    surface: u32,
    pixel: vec2<f32>,
    ground: vec4<u32>,
    origin: vec3<f32>,
}

struct CardVertex {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frames_ab: vec4<f32>,
    @location(1) frame_c: vec2<f32>,
    @location(2) world: vec3<f32>,
    @location(3) @interpolate(flat) normal_x: vec4<f32>,
    @location(4) @interpolate(flat) normal_y: vec4<f32>,
    @location(5) @interpolate(flat) normal_z: vec3<f32>,
    @location(6) @interpolate(flat) depth_axis: vec4<f32>,
    @location(7) @interpolate(flat) material: vec4<u32>,
    @location(8) @interpolate(flat) batch: u32,
    @location(9) previous: vec3<f32>,
    @location(10) crossings: vec3<f32>,
    @location(11) @interpolate(flat) slopes_ab: vec4<f32>,
    @location(12) @interpolate(flat) slope_c: vec2<f32>,
}

struct CardShaded {
    shaded: Shaded,
    depth: f32,
}

struct CardDebugTargets {
    @location(0) ids: vec4<u32>,
    @location(1) count: f32,
    @location(2) terrain: vec2<u32>,
    @builtin(frag_depth) depth: f32,
}

fn instance_pose(slot: u32) -> AffinePose {
    var pose: AffinePose;
    return pose;
}

fn instance_still_frames(slot: u32) -> u32 {
    return 0u;
}

fn slot_vertex(input: AffineVertexInput, pose: AffinePose, previous: AffinePose,
        color: SlotColor, ground: vec4<u32>, fade: u32) -> SlotVertex {
    var vertex: SlotVertex;
    return vertex;
}

fn slot_vertex_extended(input: AffineVertexInput, extras: SlotExtras, pose: AffinePose,
        previous: AffinePose, color: SlotColor, ground: vec4<u32>, fade: u32)
        -> SlotVertexExtended {
    var vertex: SlotVertexExtended;
    return vertex;
}

fn slot_fragment(input: SlotVertex) -> SlotFragment {
    var fragment: SlotFragment;
    return fragment;
}

fn slot_fragment_extended(input: SlotVertexExtended) -> SlotFragment {
    var fragment: SlotFragment;
    return fragment;
}

fn slot_shade(fragment: SlotFragment, front: bool, pretested: bool) -> Shaded {
    var shaded: Shaded;
    return shaded;
}

fn slot_masked_depth(fragment: SlotFragment) {
}

fn slot_pixel(input: SlotVertex) -> ViewPixel {
    var pixel: ViewPixel;
    return pixel;
}

fn slot_pixel_extended(input: SlotVertexExtended) -> ViewPixel {
    var pixel: ViewPixel;
    return pixel;
}

fn slot_debug(fragment: SlotFragment, front: bool, record: u32, object: u32, detail: u32)
        -> DebugTargets {
    var targets: DebugTargets;
    return targets;
}

fn placed_card_vertex(vertex: u32, batch: u32, pose: AffinePose, previous: AffinePose,
        color: SlotColor) -> CardVertex {
    var card: CardVertex;
    return card;
}

fn placed_card_shade(input: CardVertex) -> CardShaded {
    var shaded: CardShaded;
    return shaded;
}

fn placed_card_pixel(input: CardVertex) -> ViewPixel {
    var pixel: ViewPixel;
    return pixel;
}

fn placed_card_debug(input: CardVertex, record: u32, object: u32) -> CardDebugTargets {
    var targets: CardDebugTargets;
    return targets;
}
