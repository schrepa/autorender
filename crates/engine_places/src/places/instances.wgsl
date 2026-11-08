import package::place::impostors::ImpostorRecord;
import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};

struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}

struct SlotColor {
    color: vec2<u32>,
    surface: u32,
}

struct SlotFragment {
    color: vec4<f32>,
    normal: vec3<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
    uv1: vec2<f32>,
    material: vec4<u32>,
    pixel: vec2<f32>,
}

struct CardVertex {
    @invariant @builtin(position) clip_position: vec4<f32>,
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

struct SlotShaded {
    point: SurfacePoint,
    shaded: Shaded,
}

struct CardShaded {
    point: SurfacePoint,
    shaded: Shaded,
    depth: f32,
}

struct CardDebugTargets {
    @location(0) ids: vec4<u32>,
    @location(1) count: f32,
    @location(2) words: vec2<u32>,
    @builtin(frag_depth) depth: f32,
}

fn instance_pose(slot: u32) -> AffinePose {
    var pose: AffinePose;
    return pose;
}

fn placed_card_vertex(vertex: u32, batch: u32, pose: AffinePose, previous: AffinePose,
        color: SlotColor, impostor: ImpostorRecord) -> CardVertex {
    var card: CardVertex;
    return card;
}

fn placed_card_shade(input: CardVertex, impostor: ImpostorRecord) -> CardShaded {
    var shaded: CardShaded;
    return shaded;
}

fn placed_card_depth(input: CardVertex, impostor: ImpostorRecord) -> f32 {
    return 0.0;
}

fn placed_card_pixel(input: CardVertex) -> ViewPixel {
    var pixel: ViewPixel;
    return pixel;
}

fn placed_card_debug(input: CardVertex, record: u32, object: u32, impostor: ImpostorRecord)
        -> CardDebugTargets {
    var targets: CardDebugTargets;
    return targets;
}
