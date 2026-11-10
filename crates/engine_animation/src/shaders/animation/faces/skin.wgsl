import package::engine::geometry::declarations::AffinePose;
import package::generated::animation::{JointRows, SkinSlot};
import package::place::skin::SkinPose;

@group(0) @binding(0) var<storage, read> skin_rows: array<JointRows>;
@group(0) @binding(0) var<storage, read> skin_slots: array<SkinSlot>;

fn blended(start: u32, joints: vec4<u32>, weights: vec4<f32>) -> AffinePose {
    var mixed = AffinePose(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    for (var at = 0u; at < 4u; at++) {
        let held = skin_rows[start + joints[at]];
        mixed.row_0 += held.r0 * weights[at];
        mixed.row_1 += held.r1 * weights[at];
        mixed.row_2 += held.r2 * weights[at];
    }
    return mixed;
}

fn skin_pose(joints: vec4<u32>, weights: vec4<f32>, slot: u32) -> SkinPose {
    var pose: SkinPose;
    if slot >= arrayLength(&skin_slots) {
        return pose;
    }
    let held = skin_slots[slot];
    if held.joints == 0u {
        return pose;
    }
    pose.now = blended(held.now, joints, weights);
    pose.was = blended(held.was, joints, weights);
    pose.posed = true;
    return pose;
}
