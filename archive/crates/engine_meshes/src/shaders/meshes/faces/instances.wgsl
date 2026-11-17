import package::meshes::slots::{instance_colors, slot_pose, slot_previous_pose};
import package::place::instances::{AffinePose, SlotColor};

@group(0) @binding(0) var<storage, read> last_ids: array<u32>;
@group(0) @binding(0) var<storage, read> last_late_ids: array<u32>;
@group(0) @binding(0) var<storage, read> last_fading_ids: array<u32>;

fn instance_pose(slot: u32) -> AffinePose {
    return slot_pose(slot);
}

fn instance_previous_pose(slot: u32) -> AffinePose {
    return slot_previous_pose(slot);
}

fn instance_color(slot: u32) -> SlotColor {
    return instance_colors[slot];
}

fn last_count() -> u32 {
    return last_ids[0];
}

fn last_entry(index: u32) -> vec2<u32> {
    return vec2<u32>(last_ids[4u + 2u * index], last_ids[5u + 2u * index]);
}

fn last_late_count() -> u32 {
    return last_late_ids[0];
}

fn last_late_entry(index: u32) -> vec2<u32> {
    return vec2<u32>(last_late_ids[4u + 2u * index], last_late_ids[5u + 2u * index]);
}

fn last_fading_count() -> u32 {
    return last_fading_ids[0];
}

fn last_fading_entry(index: u32) -> vec2<u32> {
    return vec2<u32>(last_fading_ids[4u + 2u * index], last_fading_ids[5u + 2u * index]);
}
