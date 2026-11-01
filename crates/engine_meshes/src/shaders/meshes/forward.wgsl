import package::engine::mesh::declarations::TemporalOutput;
import package::engine::shading::{Shaded, ViewPixel};
import package::generated::targets::{ViewTargets, view_targets};
import package::meshes::shade::{extended_fragment, extended_temporal_output, fragment,
    masked_depth, shade, temporal_output};
import package::meshes::vertex::ExtendedTemporalOutput;

fn drawn_pixel(clip_position: vec4<f32>, previous: vec3<f32>) -> ViewPixel {
    return view_pixel(clip_position, vec4<f32>(previous.xy, 0.0, previous.z));
}

fn mesh_targets(shaded: Shaded, clip_position: vec4<f32>, previous: vec3<f32>) -> ViewTargets {
    return view_targets(shaded, drawn_pixel(clip_position, previous));
}

@fragment
fn fs_masked_depth(input: TemporalOutput) {
    masked_depth(fragment(temporal_output(input)));
}

@fragment
fn fs_masked_depth_extended(input: ExtendedTemporalOutput) {
    masked_depth(extended_fragment(extended_temporal_output(input)));
}

@fragment
fn fs_mesh(input: TemporalOutput, @builtin(front_facing) front: bool) -> ViewTargets {
    let shaded = shade(fragment(temporal_output(input)), front, false);
    return mesh_targets(shaded, input.clip_position, input.previous);
}

@fragment
fn fs_mesh_pretested(input: TemporalOutput, @builtin(front_facing) front: bool) -> ViewTargets {
    let shaded = shade(fragment(temporal_output(input)), front, true);
    return mesh_targets(shaded, input.clip_position, input.previous);
}

@fragment
fn fs_mesh_extended(input: ExtendedTemporalOutput, @builtin(front_facing) front: bool)
        -> ViewTargets {
    let shaded = shade(extended_fragment(extended_temporal_output(input)), front, false);
    return mesh_targets(shaded, input.clip_position, input.previous);
}

@fragment
fn fs_mesh_extended_pretested(input: ExtendedTemporalOutput,
        @builtin(front_facing) front: bool) -> ViewTargets {
    let shaded = shade(extended_fragment(extended_temporal_output(input)), front, true);
    return mesh_targets(shaded, input.clip_position, input.previous);
}

@fragment
fn fs_probe(input: TemporalOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(fragment(temporal_output(input)), front, false).linear, 1.0);
}

@fragment
fn fs_probe_extended(input: ExtendedTemporalOutput, @builtin(front_facing) front: bool)
        -> @location(0) vec4<f32> {
    let shaded = shade(extended_fragment(extended_temporal_output(input)), front, false);
    return vec4<f32>(shaded.linear, 1.0);
}
