struct AffinePose {
    row_0: vec4<f32>,
    row_1: vec4<f32>,
    row_2: vec4<f32>,
}

struct SlotColor {
    color: vec2<u32>,
    surface: u32,
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
