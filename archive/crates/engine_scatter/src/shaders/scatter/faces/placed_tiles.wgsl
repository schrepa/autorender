import package::face::impostors::ImpostorRecord;
import package::generated::scatter::{DEAD, PlacedGroup, scatter_terrain_hidden};
import package::place::placed_tiles::{Pose, Record};

@group(0) @binding(0) var<storage, read> placed_records: array<Record>;
@group(0) @binding(0) var<storage, read> placed_poses: array<Pose>;
@group(0) @binding(0) var<storage, read> placed_groups: array<PlacedGroup>;

fn placed_shown(record: Record) -> bool {
    return record.group != DEAD && !scatter_terrain_hidden(record.terrain);
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
