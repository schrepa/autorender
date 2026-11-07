import package::engine::mesh::declarations::slot_index;
import package::engine::shading::SurfaceInput;
import package::engine::surface::opacity_cut;
import package::engine::view::camera;
import package::face::instance_motion::{instance_moves, instance_offset};
import package::meshes::slots::{instance_colors, instance_poses};

fn flattened(clip: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(clip.xy, max(clip.z, 0.0), clip.w);
}

fn project(world: vec3<f32>) -> vec4<f32> {
    return flattened(camera.view_projection * vec4<f32>(world, 1.0));
}

fn slot_world(slot: u32, position: vec3<f32>) -> vec3<f32> {
    let pose = instance_poses[slot];
    let point = vec4<f32>(position, 1.0);
    let world = vec3<f32>(dot(pose.row_0, point), dot(pose.row_1, point), dot(pose.row_2, point));
    let flags = instance_colors[slot].flags;
    if !instance_moves(flags) {
        return world;
    }
    let root = vec3<f32>(pose.row_0.w, pose.row_1.w, pose.row_2.w);
    let pixels = camera.position.w / max(length(root - camera.position.xyz), 1.0e-6);
    return world + instance_offset(camera.wind, pixels, root, world.y - root.y, position, flags);
}

@vertex
fn vs_caster(@location(0) position: vec3<f32>, @location(2) drawn: u32)
        -> @builtin(position) vec4<f32> {
    return project(slot_world(slot_index(drawn), position));
}

struct MaskedOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv_alpha: vec3<f32>,
}

@vertex
fn vs_caster_masked(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(2) drawn: u32) -> MaskedOutput {
    let slot = slot_index(drawn);
    var output: MaskedOutput;
    output.clip = project(slot_world(slot, position));
    output.uv_alpha = vec3<f32>(uv, unpack2x16unorm(instance_colors[slot].color.y).y);
    return output;
}

@fragment
fn fs_caster_masked(input: MaskedOutput) {
    var surface: SurfaceInput;
    surface.color = vec4<f32>(1.0, 1.0, 1.0, input.uv_alpha.z);
    surface.uv = input.uv_alpha.xy;
    surface.uv1 = input.uv_alpha.xy;
    surface.front = true;
    if opacity_cut(surface) {
        discard;
    }
}
