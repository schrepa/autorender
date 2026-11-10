import package::engine::kinds::{ShownExtraVaryings, ShownVaryings, drawn_pixel, masked_depth,
    object_targets, shade, shown_extra_object, shown_object};
import package::face::debug::DebugTargets;
import package::generated::writes::{Outputs, blended_outputs, outputs};

@fragment
fn fs_fading_depth(input: ShownVaryings) {
    masked_depth(shown_object(input));
}

@fragment
fn fs_fading_depth_extended(input: ShownExtraVaryings) {
    masked_depth(shown_extra_object(input));
}

@fragment
fn fs_fading_lit(input: ShownVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(shown_object(input), front);
    return outputs(shaded.point, shaded.shaded, drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_fading_lit_extended(input: ShownExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(shown_extra_object(input), front);
    return outputs(shaded.point, shaded.shaded, drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_fading_blended(input: ShownVaryings, @builtin(front_facing) front: bool) -> Outputs {
    let shaded = shade(shown_object(input), front);
    return blended_outputs(shaded.point, shaded.shaded,
        drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_fading_blended_extended(input: ShownExtraVaryings, @builtin(front_facing) front: bool)
        -> Outputs {
    let shaded = shade(shown_extra_object(input), front);
    return blended_outputs(shaded.point, shaded.shaded,
        drawn_pixel(input.clip_position, input.previous));
}

@fragment
fn fs_fading_shown(input: ShownVaryings, @builtin(front_facing) front: bool) -> DebugTargets {
    return object_targets(shown_object(input), input.object, front);
}

@fragment
fn fs_fading_shown_extended(input: ShownExtraVaryings, @builtin(front_facing) front: bool)
        -> DebugTargets {
    return object_targets(shown_extra_object(input), input.object, front);
}
