import package::place::instances::{AffinePose, SlotColor};

@group(0) @binding(0) var<storage, read> instance_poses: array<AffinePose>;
@group(0) @binding(0) var<storage, read> instance_previous_poses: array<AffinePose>;
@group(0) @binding(0) var<storage, read> instance_colors: array<SlotColor>;
@group(0) @binding(0) var<storage, read> instance_grounds: array<vec4<u32>>;
@group(0) @binding(0) var<storage, read> instance_still: array<vec2<u32>>;

fn instance_pose(slot: u32) -> AffinePose {
    return instance_poses[slot];
}

fn instance_still_frames(slot: u32) -> u32 {
    return instance_still[slot].y;
}
