import package::engine::mesh::declarations::{AffinePose as PoseRows, AffineVertexInput,
    MotionOutput, drawn_slot_moved, drawn_slot_slot};
import package::engine::drawn::{DRAWN_OWN_PROJECTION, drawn_clip, drawn_records};
import package::engine::mesh::vertex::{pose_matrix, transform_vertex};
import package::engine::shading::PositionInput;
import package::engine::slots::slot_position;
import package::engine::view::{camera, detail_across, view_clip};
import package::meshes::slots::{instance_colors, instance_poses, instance_previous_poses};
import package::place::instances::{AffinePose, SlotColor};

fn posed(pose: AffinePose) -> mat4x4<f32> {
    return pose_matrix(PoseRows(pose.row_0, pose.row_1, pose.row_2));
}

fn placed_position(model: mat4x4<f32>, position: vec3<f32>, world: vec3<f32>, time: f32)
        -> vec3<f32> {
    let origin = model[3].xyz;
    var input: PositionInput;
    input.local = position;
    input.world = world;
    input.origin = origin;
    input.time = time;
    input.pixels = 1.0 / max(detail_across(origin), 1.0e-6);
    return slot_position(input);
}

fn affine_motion(input: AffineVertexInput, pose: AffinePose, previous: AffinePose,
        moved: bool, material: SlotColor, fade: u32) -> MotionOutput {
    let model = posed(pose);
    var vertex = transform_vertex(input.position, input.normal, material.color,
        material.surface, input.uv, model);
    vertex.material.w = fade;
    let point = vec4<f32>(input.position, 1.0);
    var world = model * point;
    if moved {
        world = vec4<f32>(dot(previous.row_0, point), dot(previous.row_1, point),
            dot(previous.row_2, point), 1.0);
    }
    let current = placed_position(model, input.position, vertex.world_v.xyz, camera.position.w);
    vertex.world_v = vec4<f32>(current, vertex.world_v.w);
    vertex.clip_position = view_clip(current);
    world = vec4<f32>(placed_position(model, input.position, world.xyz, camera.previous.w),
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
    if (record.flags & DRAWN_OWN_PROJECTION) == 0u {
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

fn vertex_affine_motion(input: AffineVertexInput) -> MotionOutput {
    let slot = drawn_slot_slot(input.slot);
    let moved = drawn_slot_moved(input.slot);
    var previous: AffinePose;
    if moved {
        previous = instance_previous_poses[slot];
    }
    return own_projection(affine_motion(input, instance_poses[slot], previous, moved,
        instance_colors[slot], 0u), slot);
}

fn slot_motion(input: AffineVertexInput, pose: AffinePose, previous: AffinePose,
        color: SlotColor, fade: u32) -> MotionOutput {
    let moved = any(previous.row_0 != pose.row_0) || any(previous.row_1 != pose.row_1)
        || any(previous.row_2 != pose.row_2);
    return affine_motion(input, pose, previous, moved, color, fade);
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
    return vec4<u32>(draw.x, drawn_slot_slot(input.slot), 0u, 0u);
}

@vertex
fn vs_mesh(input: AffineVertexInput, @location(14) draw: vec4<u32>) -> MeshOutput {
    let vertex = vertex_affine_motion(input);
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
fn vs_view(input: AffineVertexInput) -> MotionOutput {
    return vertex_affine_motion(input);
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

fn extended_motion(input: AffineVertexInput, color: vec4<f32>, uv1: vec2<f32>)
        -> ExtendedMotionOutput {
    let vertex = vertex_affine_motion(input);
    var output: ExtendedMotionOutput;
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
    let vertex = extended_motion(input, color, uv1);
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
        @location(10) uv1: vec2<f32>) -> ExtendedMotionOutput {
    return extended_motion(input, color, uv1);
}
