import package::engine::impostors::{IMPOSTOR_KINDS, impostor_atlas};
import package::engine::mesh::impostor_frames::ImpostorRecord;
import package::generated::scatter::{DEAD, HIDDEN, PlacedGroup};
import package::place::placed_tiles::{Pose, Record};

@group(0) @binding(0) var<storage, read> placed_records: array<Record>;
@group(0) @binding(0) var<storage, read> placed_poses: array<Pose>;
@group(0) @binding(0) var<storage, read> placed_groups: array<PlacedGroup>;
@group(0) @binding(0) var placed_frames_sampler: sampler;

fn placed_shown(record: Record) -> bool {
    return record.group != DEAD && (record.terrain & HIDDEN) == 0u;
}

fn placed_count() -> u32 {
    return arrayLength(&placed_records);
}

fn placed_record(slot: u32) -> Record {
    return placed_records[slot];
}

fn placed_pose(slot: u32) -> Pose {
    return placed_poses[slot];
}

fn placed_impostor(group: u32) -> ImpostorRecord {
    let held = placed_groups[group];
    return ImpostorRecord(held.sphere, held.extent, held.layer, held.emissive, held.flags);
}

fn placed_color(group: u32) -> vec2<u32> {
    return placed_groups[group].color;
}

fn placed_frame(at: vec2<f32>, layer: u32, level: f32) -> vec4<f32> {
    return textureSampleLevel(impostor_atlas, placed_frames_sampler, at, layer, level);
}

fn placed_frame_layers() -> u32 {
    return textureNumLayers(impostor_atlas) / IMPOSTOR_KINDS;
}
