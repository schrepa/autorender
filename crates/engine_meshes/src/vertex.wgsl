import package::meshes::declarations::{AffineVertexInput, TemporalOutput, VertexOutput};
import package::meshes::vertex::{vertex_affine, vertex_affine_temporal};

@vertex
fn vs_affine(input: AffineVertexInput) -> VertexOutput {
    return vertex_affine(input);
}

@vertex
fn vs_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    return vertex_affine_temporal(input);
}

struct ExtendedOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
}

struct ExtendedTemporalOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
}

fn extended_output(vertex: VertexOutput, color: vec4<f32>, uv1: vec2<f32>) -> ExtendedOutput {
    var output: ExtendedOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.color = color;
    output.uv1 = uv1;
    return output;
}

@vertex
fn vs_affine_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>) -> ExtendedOutput {
    return extended_output(vertex_affine(input), color, uv1);
}

@vertex
fn vs_affine_temporal_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>) -> ExtendedTemporalOutput {
    let vertex = vertex_affine_temporal(input);
    var output: ExtendedTemporalOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = vertex.previous;
    output.color = color;
    output.uv1 = uv1;
    return output;
}
