import package::engine::shading::ViewPixel;
import package::engine::view::view_pixel;
import package::generated::writes::{Outputs, blended_outputs, outputs};
import package::meshes::shade::{extended_fragment, extended_motion_output, fragment, masked_depth,
    shade, motion_output};
import package::meshes::vertex::{ExtendedMotionOutput, MotionOutput};
import package::place::instances::SlotShaded;

fn drawn_pixel(clip_position: vec4<f32>, previous: vec3<f32>) -> ViewPixel {
    return view_pixel(clip_position, vec4<f32>(previous.xy, 0.0, previous.z));
}

fn mesh_targets(shaded: SlotShaded, clip_position: vec4<f32>, previous: vec3<f32>)
        -> Outputs {
    return outputs(shaded.point, shaded.shaded, drawn_pixel(clip_position, previous));
}

@fragment
fn fs_masked_depth(input: MotionOutput) {
    masked_depth(fragment(motion_output(input)));
}

@fragment
fn fs_masked_depth_extended(input: ExtendedMotionOutput) {
    masked_depth(extended_fragment(extended_motion_output(input)));
}

@fragment
fn fs_mesh(input: MotionOutput, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(fragment(motion_output(input)), front);
    return mesh_targets(shaded, input.clip_position, input.previous);
}

@fragment
fn fs_mesh_extended(input: ExtendedMotionOutput, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(extended_fragment(extended_motion_output(input)), front);
    return mesh_targets(shaded, input.clip_position, input.previous);
}

@fragment
fn fs_mesh_blended(input: MotionOutput, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(fragment(motion_output(input)), front);
    return blended_outputs(shaded.point, shaded.shaded,
        drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_mesh_extended_blended(input: ExtendedMotionOutput,
        @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(extended_fragment(extended_motion_output(input)), front);
    return blended_outputs(shaded.point, shaded.shaded,
        drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_view(input: MotionOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(fragment(motion_output(input)), front).shaded.linear, 1.0);
}

@fragment
fn fs_view_extended(input: ExtendedMotionOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    let shaded = shade(extended_fragment(extended_motion_output(input)), front);
    return vec4<f32>(shaded.shaded.linear, 1.0);
}
