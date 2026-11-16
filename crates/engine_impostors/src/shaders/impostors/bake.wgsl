import package::engine::drawn::{GeometryExtra, GeometryVertex};
import package::engine::materials::alpha_test;
import package::engine::geometry::vertex::octahedral_normal;
import package::engine::shading::SurfaceInput;
import package::engine::mapped::{base_texel, emissive_texel};
import package::engine::surface::{masked_out, surface_point};
import package::engine::view::stable_normalize;
import package::face::lighting::SURFACE_THIN;
import package::generated::impostors::ImpostorBake;
import package::impostors::faces::impostors::{impostor_basis, impostor_flags_hemisphere,
    impostor_frame_direction, impostor_frames};

@group(2) @binding(0) var<uniform> impostor_bake: ImpostorBake;
@group(2) @binding(1) var<storage, read_write> bake_kinds: array<atomic<u32>>;

fn bake_hemisphere() -> bool {
    return impostor_flags_hemisphere(impostor_bake.flags.x);
}

struct BakeOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) local_v: vec4<f32>,
    @location(2) depth: f32,
}

struct BakeExtendedOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal_u: vec4<f32>,
    @location(1) local_v: vec4<f32>,
    @location(2) depth: f32,
    @location(3) color: vec4<f32>,
    @location(4) uv1: vec2<f32>,
}

fn bake_vertex(position: vec3<f32>, encoded_normal: vec2<f32>, uv: vec2<f32>, instance: u32)
        -> BakeOutput {
    let frames = impostor_frames(impostor_bake.flags.x);
    let tile = impostor_bake.tile;
    let cell = vec2<u32>(instance % tile.z, instance / tile.z);
    let frame = (tile.y + cell.y) * frames + tile.x + cell.x;
    let d = impostor_frame_direction(frame, frames, bake_hemisphere());
    let basis = impostor_basis(d);
    let local = (position - impostor_bake.sphere.xyz) / impostor_bake.sphere.w;
    let image = vec2<f32>(dot(local, basis.right), -dot(local, basis.up));
    let depth = clamp(dot(local, d), -1.0, 1.0);
    let drawn = (vec2<f32>(cell) + 0.5 + image * 0.5) / f32(tile.w);
    var output: BakeOutput;
    output.clip_position = vec4<f32>(drawn.x * 2.0 - 1.0, 1.0 - drawn.y * 2.0,
        0.5 + 0.5 * depth, 1.0);
    output.normal_u = vec4<f32>(octahedral_normal(encoded_normal), uv.x);
    output.local_v = vec4<f32>(position, uv.y);
    output.depth = depth;
    return output;
}

@vertex
fn vs_impostor_bake(input: GeometryVertex, @builtin(instance_index) instance: u32)
        -> BakeOutput {
    return bake_vertex(input.position, input.normal, input.uv0, instance);
}

@vertex
fn vs_impostor_bake_extended(input: GeometryVertex, extra: GeometryExtra,
        @builtin(instance_index) instance: u32) -> BakeExtendedOutput {
    let vertex = bake_vertex(input.position, input.normal, input.uv0, instance);
    var output: BakeExtendedOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.local_v = vertex.local_v;
    output.depth = vertex.depth;
    output.color = extra.color;
    output.uv1 = extra.uv1;
    return output;
}

struct BakeTargets {
    @location(0) albedo: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) geometry: vec4<f32>,
    @location(3) surface: vec4<f32>,
}

fn bake_input(input: BakeOutput, color: vec4<f32>, uv1: vec2<f32>, front: bool)
        -> SurfaceInput {
    var surface: SurfaceInput;
    surface.color = color;
    surface.normal = input.normal_u.xyz;
    surface.world = input.local_v.xyz;
    surface.world_dx = dpdx(surface.world);
    surface.world_dy = dpdy(surface.world);
    surface.uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    surface.uv1 = uv1;
    surface.surface = vec4<f32>(1.0, 1.0, 0.0, 0.0);
    surface.front = front;
    surface.eye = surface.world + input.normal_u.xyz;
    return surface;
}

fn bake_targets(input: BakeOutput, color: vec4<f32>, uv1: vec2<f32>, front: bool)
        -> BakeTargets {
    let surface = bake_input(input, color, uv1, front);
    let point = surface_point(surface);
    let base = base_texel(surface);
    if alpha_test() && masked_out(surface, point) {
        discard;
    }
    if point.kind == SURFACE_THIN {
        atomicOr(&bake_kinds[0], 1u);
    }
    var targets: BakeTargets;
    targets.albedo = vec4<f32>(point.base_color, 1.0);
    targets.emissive = vec4<f32>(emissive_texel(surface), base.a);
    targets.geometry = vec4<f32>(stable_normalize(point.normal) * 0.5 + 0.5,
        0.5 + 0.5 * input.depth);
    targets.surface = vec4<f32>(point.roughness, point.metalness, point.occlusion, 0.0);
    return targets;
}

@fragment
fn fs_impostor_bake(input: BakeOutput, @builtin(front_facing) front: bool) -> BakeTargets {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    return bake_targets(input, vec4<f32>(1.0), uv, front);
}

@fragment
fn fs_impostor_bake_extended(input: BakeExtendedOutput, @builtin(front_facing) front: bool)
        -> BakeTargets {
    var vertex: BakeOutput;
    vertex.clip_position = input.clip_position;
    vertex.normal_u = input.normal_u;
    vertex.local_v = input.local_v;
    vertex.depth = input.depth;
    return bake_targets(vertex, input.color, input.uv1, front);
}
