import package::engine::materials::{base_color_sampler, base_color_texture, material_maps};
import package::engine::view::camera;
import package::face::wind::{slot_sway, wind_sway};
import package::place::instances::{AffinePose, SlotColor};

@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;

fn flattened(clip: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(clip.xy, max(clip.z, 0.0), clip.w);
}

fn project(world: vec3<f32>) -> vec4<f32> {
    return flattened(camera.view_projection * vec4<f32>(world, 1.0));
}

fn slot_world(slot: u32, position: vec3<f32>) -> vec3<f32> {
    let pose = poses[slot];
    let point = vec4<f32>(position, 1.0);
    let world = vec3<f32>(dot(pose.row_0, point), dot(pose.row_1, point), dot(pose.row_2, point));
    let sway = slot_sway(colors[slot].flags);
    if sway == 0.0 {
        return world;
    }
    let root = vec3<f32>(pose.row_0.w, pose.row_1.w, pose.row_2.w);
    let pixels = camera.position.w / max(length(root - camera.position.xyz), 1.0e-6);
    return world + wind_sway(camera.wind, pixels, root, world.y - root.y, position, sway);
}

@vertex
fn vs_caster(@location(0) position: vec3<f32>, @location(2) slot: u32) -> @builtin(position) vec4<f32> {
    return project(slot_world(slot, position));
}

struct MaskedOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv_alpha: vec3<f32>,
}

@vertex
fn vs_caster_masked(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(2) slot: u32) -> MaskedOutput {
    var output: MaskedOutput;
    output.clip = project(slot_world(slot, position));
    output.uv_alpha = vec3<f32>(uv, unpack2x16unorm(colors[slot].color.y).y);
    return output;
}

fn masked_cut(uv_alpha: vec3<f32>) -> bool {
    let point = vec3<f32>(uv_alpha.xy, 1.0);
    let uv = vec2<f32>(dot(material_maps.transforms[0].xyz, point),
        dot(material_maps.transforms[1].xyz, point));
    let texel = textureSample(base_color_texture, base_color_sampler, uv);
    let size = vec2<f32>(textureDimensions(base_color_texture));
    let dx = dpdx(uv) * size;
    let dy = dpdy(uv) * size;
    let coverage = 1.0 + 0.25 * max(0.5 * log2(max(dot(dx, dx), dot(dy, dy))), 0.0);
    return texel.a * uv_alpha.z * coverage < material_maps.params.z;
}

@fragment
fn fs_caster_masked(input: MaskedOutput) {
    if masked_cut(input.uv_alpha) {
        discard;
    }
}
