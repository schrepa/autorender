import package::engine::shading::{Shaded, SurfacePoint, ViewPixel};
import package::place::instances::{AffinePose, SlotColor};

struct ImpostorRecord {
    sphere: vec4<f32>,
    extent: vec3<f32>,
    layer: u32,
    emissive: vec3<f32>,
    flags: u32,
}

struct ImpostorBasis {
    right: vec3<f32>,
    up: vec3<f32>,
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

fn impostor_frames(flags: u32) -> u32 {
    return 2u;
}

fn impostor_flags_texels(flags: u32) -> u32 {
    return 0u;
}

fn impostor_flags_hemisphere(flags: u32) -> bool {
    return false;
}

fn impostor_flags_thin(flags: u32) -> bool {
    return false;
}

fn impostor_cutoff() -> f32 {
    return 1.0;
}

fn impostor_grid(d: vec3<f32>, hemisphere: bool) -> vec2<f32> {
    return vec2<f32>(0.0);
}

fn impostor_frame_direction(frame: u32, frames: u32, hemisphere: bool) -> vec3<f32> {
    return vec3<f32>(0.0, 1.0, 0.0);
}

fn impostor_basis(d: vec3<f32>) -> ImpostorBasis {
    var basis: ImpostorBasis;
    return basis;
}

fn impostor_facing(slope: vec2<f32>) -> f32 {
    return 1.0;
}

fn impostor_met(sampled: f32, covered: f32, crossing: f32, facing: f32) -> f32 {
    return 0.0;
}

fn impostor_albedo(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn impostor_emissive(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn impostor_geometry(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn impostor_surface(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn impostor_size() -> u32 {
    return 1u;
}

fn impostor_levels() -> u32 {
    return 1u;
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
