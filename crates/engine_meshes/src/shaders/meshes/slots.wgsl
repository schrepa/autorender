import package::engine::drawn::{drawn_poses, drawn_previous_poses};
import package::place::instances::{AffinePose, SlotColor};

@group(0) @binding(0) var<storage, read> instance_colors: array<SlotColor>;

fn slot_pose(slot: u32) -> AffinePose {
    let pose = drawn_poses[slot];
    return AffinePose(pose.row_0, pose.row_1, pose.row_2);
}

fn slot_previous_pose(slot: u32) -> AffinePose {
    let pose = drawn_previous_poses[slot];
    return AffinePose(pose.row_0, pose.row_1, pose.row_2);
}
