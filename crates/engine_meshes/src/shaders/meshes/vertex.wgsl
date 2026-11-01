import package::engine::materials::GROUNDING;
import package::engine::mesh::declarations::{AffineVertexInput, TemporalOutput, slot_fade,
    slot_index, slot_moved};
import package::engine::mesh::vertex::{origin_key, placed_material, pose_matrix,
    transform_vertex};
import package::engine::view::{camera, detail_across};
import package::face::wind::{slot_sway, wind_sway};
import package::meshes::faces::instances::{instance_colors, instance_poses,
    instance_previous_poses};

fn vertex_wind(model: mat4x4<f32>, position: vec3<f32>, world: vec3<f32>, sway: f32,
        time: f32) -> vec3<f32> {
    let root = model[3].xyz;
    return wind_sway(vec4<f32>(camera.wind.xyz, time), 1.0 / max(detail_across(root), 1.0e-6),
        root, world.y - root.y, position, sway);
}

fn vertex_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    let slot = slot_index(input.slot);
    let model = pose_matrix(instance_poses[slot]);
    let material = instance_colors[slot];
    var vertex = transform_vertex(input.position, input.normal, material.color,
        material.surface, input.uv, model);
    vertex.material.w = slot_fade(input.slot) | origin_key(model[3].xyz);
    let point = vec4<f32>(input.position, 1.0);
    var world = model * point;
    if slot_moved(input.slot) {
        let pose = instance_previous_poses[slot];
        world = vec4<f32>(dot(pose.row_0, point), dot(pose.row_1, point),
            dot(pose.row_2, point), 1.0);
    }
    let sway = slot_sway(material.flags);
    if sway > 0.0 {
        let current = vertex.world_v.xyz
            + vertex_wind(model, input.position, vertex.world_v.xyz, sway, camera.position.w);
        vertex.world_v = vec4<f32>(current, vertex.world_v.w);
        vertex.clip_position = camera.view_projection * vec4<f32>(current, 1.0);
        world = vec4<f32>(world.xyz
            + vertex_wind(model, input.position, world.xyz, sway, camera.wind.w), 1.0);
    }
    if GROUNDING {
        vertex.material = placed_material(vertex.material, slot);
    }
    let previous = camera.previous_view_projection * world;
    var output: TemporalOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = previous.xyw;
    return output;
}

@vertex
fn vs_mesh(input: AffineVertexInput) -> TemporalOutput {
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

@vertex
fn vs_mesh_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
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
