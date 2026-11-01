import package::engine::materials::{
    ALPHA_MASK, MATERIAL_MAPS, MaterialTexels, NORMAL_MAP, TOP_LAYER, VARIANT_MAPS,
    VARIANT_NORMAL_MAP, VARIANT_TOP_LAYER, geometric_normal, mapped_normal, material_flag,
    material_maps, material_texels, top_layer_amount
};
import package::engine::mesh::impostor_frames::{
    IMPOSTOR_HEMISPHERE, ImpostorBake, impostor_basis, impostor_frame_direction, impostor_frames
};
import package::engine::mesh::vertex::octahedral_normal;
import package::engine::view::stable_normalize;

@group(2) @binding(15) var<uniform> impostor_bake: ImpostorBake;
@group(2) @binding(16) var impostor_shade: texture_depth_2d;

fn bake_hemisphere() -> bool {
    return (impostor_bake.flags.x & IMPOSTOR_HEMISPHERE) != 0u;
}

fn impostor_self_visibility(position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let size = textureDimensions(impostor_shade).x;
    let frames = impostor_frames(impostor_bake.flags.x);
    let texel = 2.0 * f32(frames) / f32(size);
    let p = (position - impostor_bake.sphere.xyz) / impostor_bake.sphere.w
        + normal * (1.5 * texel);
    let hemisphere = bake_hemisphere();
    var seen = 0.0;
    var total = 0.0;
    let stride = (frames + 7u) / 8u;
    for (var frame = 0u; frame < frames * frames; frame++) {
        if (frame % frames) % stride != 0u || (frame / frames) % stride != 0u {
            continue;
        }
        let d = impostor_frame_direction(frame, frames, hemisphere);
        let facing = dot(normal, d);
        if d.y < 0.0 || facing <= 0.0 {
            continue;
        }
        let basis = impostor_basis(d);
        let image = vec2<f32>(dot(p, basis.right), -dot(p, basis.up));
        let cell = vec2<f32>(f32(frame % frames), f32(frame / frames));
        let atlas = (cell + 0.5 + image * 0.5) / f32(frames);
        let at = min(vec2<u32>(max(atlas, vec2<f32>(0.0)) * f32(size)), vec2<u32>(size - 1u));
        let nearest = textureLoad(impostor_shade, at, 0);
        total += facing;
        if 0.5 + 0.5 * dot(p, d) >= nearest {
            seen += facing;
        }
    }
    if total <= 0.0 {
        return 1.0;
    }
    return seen / total;
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
fn vs_impostor_bake(@location(0) position: vec3<f32>, @location(1) normal: vec2<f32>,
        @location(7) uv: vec2<f32>, @builtin(instance_index) instance: u32) -> BakeOutput {
    return bake_vertex(position, normal, uv, instance);
}

@vertex
fn vs_impostor_bake_extended(@location(0) position: vec3<f32>, @location(1) normal: vec2<f32>,
        @location(7) uv: vec2<f32>, @location(9) color: vec4<f32>, @location(10) uv1: vec2<f32>,
        @builtin(instance_index) instance: u32) -> BakeExtendedOutput {
    let vertex = bake_vertex(position, normal, uv, instance);
    var output: BakeExtendedOutput;
    output.clip_position = vertex.clip_position;
    output.normal_u = vertex.normal_u;
    output.local_v = vertex.local_v;
    output.depth = vertex.depth;
    output.color = color;
    output.uv1 = uv1;
    return output;
}

struct BakeTargets {
    @location(0) albedo: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) geometry: vec4<f32>,
    @location(3) surface: vec4<f32>,
}

fn bake_mask(texels: MaterialTexels, color: vec4<f32>) {
    if ALPHA_MASK && texels.base.a * color.a * texels.coverage < material_maps.params.z {
        discard;
    }
}

fn bake_targets(input: BakeOutput, color: vec4<f32>, uv1: vec2<f32>, front: bool)
        -> BakeTargets {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    let texels = material_texels(uv, uv1, 0.0);
    let local = input.local_v.xyz;
    let local_dx = dpdx(local);
    let local_dy = dpdy(local);
    let geometric = geometric_normal(input.normal_u.xyz, front);
    var normal = geometric;
    if material_flag(VARIANT_NORMAL_MAP, NORMAL_MAP) {
        normal = mapped_normal(geometric, local_dx, local_dy, texels.normal_dx,
            texels.normal_dy, texels.normal, material_maps.params.x);
    }
    var albedo = texels.base.rgb * color.rgb;
    if material_flag(VARIANT_TOP_LAYER, TOP_LAYER) {
        let up = stable_normalize(normal).y;
        let amount = (top_layer_amount(up, 0.25, texels) + top_layer_amount(up, 0.5, texels)
            + top_layer_amount(up, 0.75, texels)) / 3.0;
        albedo = mix(albedo, material_maps.top.rgb, amount);
        normal = mix(normal, geometric, 0.5 * amount);
    }
    var emissive = vec3<f32>(0.0);
    var occlusion = 1.0;
    if material_flag(VARIANT_MAPS, MATERIAL_MAPS) {
        emissive = texels.emissive;
        occlusion = 1.0 + material_maps.params.y * (texels.occlusion - 1.0);
    }
    bake_mask(texels, color);
    var targets: BakeTargets;
    targets.albedo = vec4<f32>(albedo, 1.0);
    targets.emissive = vec4<f32>(emissive, texels.base.a);
    targets.geometry = vec4<f32>(stable_normalize(normal) * 0.5 + 0.5, 0.5 + 0.5 * input.depth);
    targets.surface = vec4<f32>(texels.factors, occlusion,
        impostor_self_visibility(local, geometric));
    return targets;
}

@fragment
fn fs_impostor_shade(input: BakeOutput) {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    bake_mask(material_texels(uv, uv, 0.0), vec4<f32>(1.0));
}

@fragment
fn fs_impostor_shade_extended(input: BakeExtendedOutput) {
    let uv = vec2<f32>(input.normal_u.w, input.local_v.w);
    bake_mask(material_texels(uv, input.uv1, 0.0), input.color);
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
