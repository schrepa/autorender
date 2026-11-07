import package::place::instances::{AffinePose, SlotColor};

@group(0) @binding(0) var<storage, read> instance_poses: array<AffinePose>;
@group(0) @binding(0) var<storage, read> instance_previous_poses: array<AffinePose>;
@group(0) @binding(0) var<storage, read> instance_colors: array<SlotColor>;
