import package::place::placed_tiles::{DEAD, HIDDEN, Pose, Record};

@group(0) @binding(0) var<storage, read> placed_records: array<Record>;
@group(0) @binding(0) var<storage, read> placed_poses: array<Pose>;

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
