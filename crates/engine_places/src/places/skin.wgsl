import package::engine::mesh::declarations::AffinePose;

struct SkinPose {
    now: AffinePose,
    was: AffinePose,
    posed: bool,
}

fn skin_pose(joints: vec4<u32>, weights: vec4<f32>, slot: u32) -> SkinPose {
    var pose: SkinPose;
    return pose;
}
