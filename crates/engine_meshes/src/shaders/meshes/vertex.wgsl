import package::engine::mesh::declarations::{AffinePose as PoseRows, MotionOutput,
    drawn_slot_moved, drawn_slot_slot};
import package::engine::drawn::{GeometryExtra, GeometrySkin, GeometryVertex, INPUT_DRAW,
    INPUT_SLOT, drawn_clip, drawn_flags_own_projection, drawn_records};
import package::engine::mesh::vertex::{pose_matrix, transform_vertex};
import package::engine::shading::PositionInput;
import package::engine::slots::slot_position;
import package::engine::view::{camera, detail_across, view_clip};
import package::face::skin::skin_pose;
import package::meshes::slots::{instance_colors, instance_poses, instance_previous_poses};
import package::place::instances::{AffinePose, SlotColor};

fn rows(pose: AffinePose) -> PoseRows {
    return PoseRows(pose.row_0, pose.row_1, pose.row_2);
}

fn posed(pose: AffinePose) -> mat4x4<f32> {
    return pose_matrix(rows(pose));
}

fn placed_position(origin: vec3<f32>, position: vec3<f32>, world: vec3<f32>, time: f32)
        -> vec3<f32> {
    var input: PositionInput;
    input.local = position;
    input.world = world;
    input.origin = origin;
    input.time = time;
    input.pixels = 1.0 / max(detail_across(origin), 1.0e-6);
    return slot_position(input);
}

fn affine_motion(input: GeometryVertex, model: mat4x4<f32>, previous: PoseRows, moved: bool,
        origin: vec3<f32>, material: SlotColor) -> MotionOutput {
    var vertex = transform_vertex(input.position, input.normal, material.color,
        material.surface, input.uv0, model);
    let point = vec4<f32>(input.position, 1.0);
    var world = model * point;
    if moved {
        world = vec4<f32>(dot(previous.row_0, point), dot(previous.row_1, point),
            dot(previous.row_2, point), 1.0);
    }
    let current = placed_position(origin, input.position, vertex.world_v.xyz, camera.position.w);
    vertex.world_v = vec4<f32>(current, vertex.world_v.w);
    vertex.clip_position = view_clip(current);
    world = vec4<f32>(placed_position(origin, input.position, world.xyz, camera.previous.w),
        1.0);
    let shown = camera.previous_view_projection * world;
    var output: MotionOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = shown.xyw;
    return output;
}

fn own_projection(output: MotionOutput, slot: u32) -> MotionOutput {
    let record = drawn_records[slot];
    if !drawn_flags_own_projection(record.flags) {
        return output;
    }
    let lens = camera.view_projection;
    let rise = vec3<f32>(lens[0].y, lens[1].y, lens[2].y)
        - camera.jitter.y * vec3<f32>(lens[0].w, lens[1].w, lens[2].w);
    let before = camera.previous_view_projection;
    let risen = vec3<f32>(before[0].y, before[1].y, before[2].y);
    let shown = vec4<f32>(output.previous.xy, 0.0, output.previous.z);
    var projected = output;
    projected.clip_position = drawn_clip(record, output.clip_position, length(rise),
        camera.jitter.xy * output.clip_position.w);
    projected.previous = drawn_clip(record, shown, length(risen), vec2<f32>(0.0)).xyw;
    return projected;
}

fn mesh_motion(input: GeometryVertex, drawn: u32) -> MotionOutput {
    let slot = drawn_slot_slot(drawn);
    let moved = drawn_slot_moved(drawn);
    var previous: AffinePose;
    if moved {
        previous = instance_previous_poses[slot];
    }
    let model = posed(instance_poses[slot]);
    return own_projection(affine_motion(input, model, rows(previous), moved, model[3].xyz,
        instance_colors[slot]), slot);
}

fn skin_motion(input: GeometryVertex, skin: GeometrySkin, drawn: u32) -> MotionOutput {
    let slot = drawn_slot_slot(drawn);
    let pose = skin_pose(skin.joints, skin.weights, slot);
    if !pose.posed {
        var hidden: MotionOutput;
        return hidden;
    }
    let node = instance_poses[slot];
    let origin = vec3<f32>(node.row_0.w, node.row_1.w, node.row_2.w);
    return own_projection(affine_motion(input, pose_matrix(pose.now), pose.was, true, origin,
        instance_colors[slot]), slot);
}

struct MeshOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) previous: vec3<f32>,
    @location(6) @interpolate(flat) object: vec4<u32>,
}

fn drawn_object(drawn: u32, draw: vec4<u32>) -> vec4<u32> {
    return vec4<u32>(draw.x, drawn_slot_slot(drawn), 0u, 0u);
}

@vertex
fn vs_mesh(input: GeometryVertex, @location(INPUT_SLOT) slot: u32,
        @location(INPUT_DRAW) draw: vec4<u32>) -> MeshOutput {
    let vertex = mesh_motion(input, slot);
    var output: MeshOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = vertex.previous;
    output.object = drawn_object(slot, draw);
    return output;
}

@vertex
fn vs_view(input: GeometryVertex, @location(INPUT_SLOT) slot: u32) -> MotionOutput {
    return mesh_motion(input, slot);
}

struct ExtendedOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) world_v: vec4<f32>,
    @location(2) @interpolate(flat) material: vec4<u32>,
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
}

struct ExtendedMotionOutput {
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

fn extended_motion(vertex: MotionOutput, extra: GeometryExtra) -> ExtendedMotionOutput {
    var output: ExtendedMotionOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = vertex.previous;
    output.color = extra.color;
    output.uv1 = extra.uv1;
    return output;
}

fn extended_mesh(vertex: MotionOutput, extra: GeometryExtra, object: vec4<u32>)
        -> ExtendedMeshOutput {
    var output: ExtendedMeshOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.world_v = vertex.world_v;
    output.material = vertex.material;
    output.previous = vertex.previous;
    output.color = extra.color;
    output.uv1 = extra.uv1;
    output.object = object;
    return output;
}

@vertex
fn vs_mesh_extended(input: GeometryVertex, extra: GeometryExtra,
        @location(INPUT_SLOT) slot: u32, @location(INPUT_DRAW) draw: vec4<u32>)
        -> ExtendedMeshOutput {
    return extended_mesh(mesh_motion(input, slot), extra, drawn_object(slot, draw));
}

@vertex
fn vs_view_extended(input: GeometryVertex, extra: GeometryExtra,
        @location(INPUT_SLOT) slot: u32) -> ExtendedMotionOutput {
    return extended_motion(mesh_motion(input, slot), extra);
}

@vertex
fn vs_skin(input: GeometryVertex, extra: GeometryExtra, skin: GeometrySkin,
        @location(INPUT_SLOT) slot: u32, @location(INPUT_DRAW) draw: vec4<u32>)
        -> ExtendedMeshOutput {
    return extended_mesh(skin_motion(input, skin, slot), extra, drawn_object(slot, draw));
}

@vertex
fn vs_skin_view(input: GeometryVertex, extra: GeometryExtra, skin: GeometrySkin,
        @location(INPUT_SLOT) slot: u32) -> ExtendedMotionOutput {
    return extended_motion(skin_motion(input, skin, slot), extra);
}
