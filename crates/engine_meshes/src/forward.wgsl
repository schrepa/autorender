import package::face::dither::stored;
import package::meshes::declarations::{SceneTargets, TemporalOutput, TemporalTargets, VertexOutput};
import package::meshes::shade::{extended_fragment, extended_temporal_output, fragment,
    masked_depth, shade, temporal_output};
import package::meshes::surface::{scene_targets, temporal_targets};
import package::meshes::vertex::{ExtendedOutput, ExtendedTemporalOutput};

@fragment
fn fs_masked_depth(input: VertexOutput) {
    masked_depth(fragment(input));
}

@fragment
fn fs_masked_depth_extended(input: ExtendedOutput) {
    masked_depth(extended_fragment(input));
}

@fragment
fn fs_masked_depth_temporal_extended(input: ExtendedTemporalOutput) {
    masked_depth(extended_fragment(extended_temporal_output(input)));
}

@fragment
fn fs_post(input: VertexOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    let shaded = shade(fragment(input), front);
    return scene_targets(shaded, stored(shaded.color, input.clip_position.xy));
}

@fragment
fn fs_temporal(input: TemporalOutput, @builtin(front_facing) front: bool) -> TemporalTargets {
    let shaded = shade(fragment(temporal_output(input)), front);
    return temporal_targets(shaded, stored(shaded.color, input.clip_position.xy),
        input.clip_position, input.previous);
}

@fragment
fn fs_probe(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(fragment(input), front).linear, 1.0);
}

@fragment
fn fs_probe_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(shade(extended_fragment(input), front).linear, 1.0);
}

@fragment
fn fs_post_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> SceneTargets {
    let shaded = shade(extended_fragment(input), front);
    return scene_targets(shaded, stored(shaded.color, input.clip_position.xy));
}

@fragment
fn fs_temporal_extended(input: ExtendedTemporalOutput, @builtin(front_facing) front: bool)
        -> TemporalTargets {
    let values = extended_fragment(extended_temporal_output(input));
    let shaded = shade(values, front);
    return temporal_targets(shaded, stored(shaded.color, input.clip_position.xy),
        input.clip_position, input.previous);
}
