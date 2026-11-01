import package::engine::mesh::declarations::AffineVertexInput;
import package::engine::shading::{Shaded, ViewPixel};

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

struct SlotVertex {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
}

struct SlotFragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    surface: u32,
    pixel: vec2<f32>,
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

fn slot_fragment(input: SlotVertex) -> SlotFragment {
    var fragment: SlotFragment;
    return fragment;
}

fn slot_shade(fragment: SlotFragment, front: bool) -> Shaded {
    var shaded: Shaded;
    return shaded;
}

fn slot_masked_depth(fragment: SlotFragment) {
}

fn slot_pixel(input: SlotVertex) -> ViewPixel {
    var pixel: ViewPixel;
    return pixel;
}
