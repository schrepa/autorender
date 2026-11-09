import package::engine::shading::{Shaded, SurfacePoint};

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

struct SlotShaded {
    point: SurfacePoint,
    shaded: Shaded,
}

fn instance_pose(slot: u32) -> AffinePose {
    var pose: AffinePose;
    return pose;
}

fn instance_previous_pose(slot: u32) -> AffinePose {
    var pose: AffinePose;
    return pose;
}

fn instance_color(slot: u32) -> SlotColor {
    var color: SlotColor;
    return color;
}

fn last_count() -> u32 {
    return 0u;
}

fn last_entry(index: u32) -> vec2<u32> {
    return vec2<u32>(0u);
}

fn last_late_count() -> u32 {
    return 0u;
}

fn last_late_entry(index: u32) -> vec2<u32> {
    return vec2<u32>(0u);
}

fn last_fading_count() -> u32 {
    return 0u;
}

fn last_fading_entry(index: u32) -> vec2<u32> {
    return vec2<u32>(0u);
}
