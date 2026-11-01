import package::engine::mesh::impostor_frames::ImpostorRecord;

struct Record {
    center: vec3<f32>,
    radius: f32,
    group: u32,
    scale: f32,
    terrain: u32,
    sink: f32,
    state: u32,
    pad: array<u32, 3>,
}

struct Pose {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

fn placed_shown(record: Record) -> bool {
    return false;
}

fn placed_count() -> u32 {
    return 0u;
}

fn placed_record(slot: u32) -> Record {
    var record: Record;
    return record;
}

fn placed_pose(slot: u32) -> Pose {
    var pose: Pose;
    return pose;
}

fn placed_impostor(group: u32) -> ImpostorRecord {
    var impostor: ImpostorRecord;
    return impostor;
}

fn placed_color(group: u32) -> vec2<u32> {
    return vec2<u32>(0u);
}

fn placed_frame(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return vec4<f32>(0.0);
}

fn placed_frame_layers() -> u32 {
    return 1u;
}
