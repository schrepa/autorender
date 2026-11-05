import package::engine::mesh::declarations::{AffinePose as PoseRows, AffineVertexInput,
    TemporalOutput, slot_fade, slot_index, slot_moved};
import package::engine::mesh::vertex::{first_person, first_person_clip, first_person_previous,
    origin_key, output_color, pose_matrix, transform_vertex};
import package::engine::view::{camera, detail_across};
import package::face::wind::{slot_sway, wind_sway};
import package::meshes::slots::{instance_colors, instance_poses, instance_previous_poses};
import package::place::instances::{AffinePose, SlotColor};

fn posed(pose: AffinePose) -> mat4x4<f32> {
    return pose_matrix(PoseRows(pose.row_0, pose.row_1, pose.row_2));
}

fn pose_origin(pose: AffinePose) -> vec3<f32> {
    return vec3<f32>(pose.row_0.w, pose.row_1.w, pose.row_2.w);
}

fn vertex_wind(model: mat4x4<f32>, position: vec3<f32>, world: vec3<f32>, sway: f32,
        time: f32) -> vec3<f32> {
    let root = model[3].xyz;
    return wind_sway(vec4<f32>(camera.wind.xyz, time), 1.0 / max(detail_across(root), 1.0e-6),
        root, world.y - root.y, position, sway);
}

fn slot_wind(model: mat4x4<f32>, world: vec3<f32>, sway: f32, time: f32) -> vec3<f32> {
    return vertex_wind(model, vec3<f32>(0.0), world, sway, time);
}

fn affine_temporal(input: AffineVertexInput, pose: AffinePose, previous: AffinePose,
        moved: bool, material: SlotColor, fade: u32) -> TemporalOutput {
    let model = posed(pose);
    var vertex = transform_vertex(input.position, input.normal, material.color,
        material.surface, input.uv, model);
    vertex.material.w = fade | origin_key(model[3].xyz);
    let point = vec4<f32>(input.position, 1.0);
    var world = model * point;
    if moved {
        world = vec4<f32>(dot(previous.row_0, point), dot(previous.row_1, point),
            dot(previous.row_2, point), 1.0);
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
    let shown = camera.previous_view_projection * world;
    var output: TemporalOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = shown.xyw;
    if first_person(material.flags) {
        output.clip_position = first_person_clip(vertex.clip_position, material.flags);
        output.previous = first_person_previous(shown, material.flags);
    }
    return output;
}

fn vertex_affine_temporal(input: AffineVertexInput) -> TemporalOutput {
    let slot = slot_index(input.slot);
    let moved = slot_moved(input.slot);
    var previous: AffinePose;
    if moved {
        previous = instance_previous_poses[slot];
    }
    return affine_temporal(input, instance_poses[slot], previous, moved, instance_colors[slot],
        slot_fade(input.slot));
}

fn slot_temporal(input: AffineVertexInput, pose: AffinePose, previous: AffinePose,
        color: SlotColor, ground: vec4<u32>, fade: u32) -> TemporalOutput {
    let moved = any(previous.row_0 != pose.row_0) || any(previous.row_1 != pose.row_1)
        || any(previous.row_2 != pose.row_2);
    var vertex = affine_temporal(input, pose, previous, moved, color, fade);
    if ground.z != 0u {
        vertex.material.x = pack4x8unorm(output_color(vertex.material));
    }
    return vertex;
}

struct MeshOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

fn drawn_object(input: AffineVertexInput, draw: vec4<u32>) -> vec4<u32> {
    return vec4<u32>(draw.x, slot_index(input.slot), 0u, 0u);
}

@vertex
fn vs_mesh(input: AffineVertexInput, @location(14) draw: vec4<u32>) -> MeshOutput {
    let vertex = vertex_affine_temporal(input);
    var output: MeshOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = vertex.previous;
    output.object = drawn_object(input, draw);
    return output;
}

@vertex
fn vs_view(input: AffineVertexInput) -> TemporalOutput {
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

struct ExtendedMeshOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(4) color: vec4<f32>,
    @location(5) uv1: vec2<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

fn extended_temporal(input: AffineVertexInput, color: vec4<f32>, uv1: vec2<f32>)
        -> ExtendedTemporalOutput {
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

@vertex
fn vs_mesh_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>, @location(14) draw: vec4<u32>) -> ExtendedMeshOutput {
    let vertex = extended_temporal(input, color, uv1);
    var output: ExtendedMeshOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = vertex.previous;
    output.color = vertex.color;
    output.uv1 = vertex.uv1;
    output.object = drawn_object(input, draw);
    return output;
}

@vertex
fn vs_view_extended(input: AffineVertexInput, @location(9) color: vec4<f32>,
        @location(10) uv1: vec2<f32>) -> ExtendedTemporalOutput {
    return extended_temporal(input, color, uv1);
}
