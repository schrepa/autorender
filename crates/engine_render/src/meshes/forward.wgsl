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
fn fs_main(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade(fragment(input), front).color, input.clip_position.xy), 1.0);
}

@fragment
fn fs_main_lit(input: VertexOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_lit(fragment(input), front).color, input.clip_position.xy),
        1.0);
}

@fragment
fn fs_post(input: VertexOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    return scene_targets(shade(fragment(input), front), input.clip_position.xy);
}

@fragment
fn fs_post_lit(input: VertexOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    return scene_targets(shade_lit(fragment(input), front), input.clip_position.xy);
}

@fragment
fn fs_temporal(input: TemporalOutput, @builtin(front_facing) front: bool) -> TemporalTargets {
    return temporal_targets(shade(fragment(temporal_output(input)), front),
        input.clip_position, input.previous);
}

@fragment
fn fs_temporal_lit(input: TemporalOutput, @builtin(front_facing) front: bool)
        -> TemporalTargets {
    return temporal_targets(shade_lit(fragment(temporal_output(input)), front),
        input.clip_position, input.previous);
}

@fragment
fn fs_probe(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(fragment(input), front).linear, 1.0);
}

@fragment
fn fs_probe_lit(input: VertexOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(shade_lit(fragment(input), front).linear, 1.0);
}

@fragment
fn fs_main_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade(extended_fragment(input), front).color,
        input.clip_position.xy), 1.0);
}

@fragment
fn fs_main_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_lit(extended_fragment(input), front).color,
        input.clip_position.xy), 1.0);
}

@fragment
fn fs_probe_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(shade(extended_fragment(input), front).linear, 1.0);
}

@fragment
fn fs_probe_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(shade_lit(extended_fragment(input), front).linear, 1.0);
}

@fragment
fn fs_post_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> SceneTargets {
    return scene_targets(shade(extended_fragment(input), front), input.clip_position.xy);
}

@fragment
fn fs_post_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> SceneTargets {
    return scene_targets(shade_lit(extended_fragment(input), front), input.clip_position.xy);
}

@fragment
fn fs_temporal_extended(input: ExtendedTemporalOutput, @builtin(front_facing) front: bool)
        -> TemporalTargets {
    let values = extended_fragment(extended_temporal_output(input));
    return temporal_targets(shade(values, front), input.clip_position, input.previous);
}

@fragment
fn fs_temporal_lit_extended(input: ExtendedTemporalOutput,
        @builtin(front_facing) front: bool) -> TemporalTargets {
    let values = extended_fragment(extended_temporal_output(input));
    return temporal_targets(shade_lit(values, front), input.clip_position, input.previous);
}
