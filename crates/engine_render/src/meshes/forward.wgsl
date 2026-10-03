// Forward fragment stages of meshes: opaque passes into the canvas, the
// screen effects' targets and temporal targets, transparent accumulation
// and reflection probe captures, unlit and lit, for meshes with and without
// a second vertex stream (decision 034).

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade(fragment(input), front).color), 1.0);
}

@fragment
fn fs_main_lit(input: VertexOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_lit(fragment(input), front).color), 1.0);
}

// The scene target is sRGB: it stores linear color with output encoding.
@fragment
fn fs_post(input: VertexOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    return scene_targets(shade(fragment(input), front));
}

@fragment
fn fs_post_lit(input: VertexOutput, @builtin(front_facing) front: bool) -> SceneTargets {
    return scene_targets(shade_lit(fragment(input), front));
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
fn fs_transparent(input: VertexOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = fragment(input);
    let point = transparent_point(values, front);
    return transparent_targets(finish(point, vec3<f32>(0.0)), point, values);
}

@fragment
fn fs_transparent_lit(input: VertexOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = fragment(input);
    let point = transparent_point(values, front);
    return transparent_targets(finish(point, local_light(point, values.pixel)), point, values);
}

// Reflection probe captures (decision 043): the linear light before the
// grade, which the environment maps hold.
@fragment
fn fs_probe(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(fragment(input), front).linear, 1.0);
}

@fragment
fn fs_probe_lit(input: VertexOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(shade_lit(fragment(input), front).linear, 1.0);
}

// The same passes for meshes with a second vertex stream (decision 034).
@fragment
fn fs_main_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade(extended_fragment(input), front).color), 1.0);
}

@fragment
fn fs_main_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    return vec4<f32>(encoded(shade_lit(extended_fragment(input), front).color), 1.0);
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
    return scene_targets(shade(extended_fragment(input), front));
}

@fragment
fn fs_post_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> SceneTargets {
    return scene_targets(shade_lit(extended_fragment(input), front));
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

@fragment
fn fs_transparent_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = extended_fragment(input);
    let point = transparent_point(values, front);
    return transparent_targets(finish(point, vec3<f32>(0.0)), point, values);
}

@fragment
fn fs_transparent_lit_extended(input: ExtendedOutput, @builtin(front_facing) front: bool)
        -> TransparentTargets {
    let values = extended_fragment(input);
    let point = transparent_point(values, front);
    return transparent_targets(finish(point, local_light(point, values.pixel)), point, values);
}
