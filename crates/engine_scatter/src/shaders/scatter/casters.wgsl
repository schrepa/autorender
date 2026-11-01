import package::engine::impostors::{impostor_albedo, impostor_sampler};
import package::engine::materials::{base_color_sampler, base_color_texture, material_maps};
import package::engine::mesh::impostor_frames::{
    IMPOSTOR_HEMISPHERE, impostor_basis, impostor_frame_direction, impostor_frames, impostor_grid,
    impostor_index
};
import package::engine::view::camera;
import package::face::instances::{AffinePose, SlotColor};
import package::face::wind::{slot_sway, wind_sway};
import package::scatter::faces::placed_tiles::placed_impostor;

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

struct CardOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) frame: vec4<u32>,
}

const CARD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_impostor_card(@builtin(vertex_index) vertex: u32, @location(2) slot: u32,
        @location(3) index: u32) -> CardOutput {
    let pose = poses[slot];
    let record = placed_impostor(impostor_index(index));
    let clip = camera.view_projection;
    let toward = -vec3<f32>(clip[0].z, clip[1].z, clip[2].z);
    let x = vec3<f32>(pose.row_0.x, pose.row_1.x, pose.row_2.x);
    let y = vec3<f32>(pose.row_0.y, pose.row_1.y, pose.row_2.y);
    let z = vec3<f32>(pose.row_0.z, pose.row_1.z, pose.row_2.z);
    let yz = cross(y, z);
    let orientation = sign(dot(x, yz));
    let sun = normalize(vec3<f32>(dot(yz, toward), dot(cross(z, x), toward),
        dot(cross(x, y), toward)) * orientation);
    let hemisphere = (record.flags & IMPOSTOR_HEMISPHERE) != 0u;
    let frames = impostor_frames(record.flags);
    let grid = (impostor_grid(sun, hemisphere) * 0.5 + 0.5) * f32(frames - 1u);
    let cell = vec2<u32>(min(round(grid), vec2<f32>(f32(frames - 1u))));
    let basis = impostor_basis(impostor_frame_direction(cell.y * frames + cell.x, frames,
        hemisphere));
    let corner = CARD[vertex];
    let local = vec4<f32>(record.sphere.xyz
        + (basis.right * corner.x + basis.up * corner.y) * record.sphere.w, 1.0);
    var output: CardOutput;
    output.clip = project(vec3<f32>(dot(pose.row_0, local), dot(pose.row_1, local),
        dot(pose.row_2, local)));
    output.uv = (vec2<f32>(cell) + 0.5 + vec2<f32>(corner.x, -corner.y) * 0.5) / f32(frames);
    output.frame = vec4<u32>(record.layer, cell, frames);
    return output;
}

@fragment
fn fs_impostor_card(input: CardOutput) {
    let size = f32(textureDimensions(impostor_albedo).x);
    let dx = dpdx(input.uv) * size;
    let dy = dpdy(input.uv) * size;
    let coarsest = f32(textureNumLevels(impostor_albedo) - 1u);
    let squared = max(max(dot(dx, dx), dot(dy, dy)), 1.0e-12);
    let level = clamp(0.5 * log2(squared), 0.0, coarsest);
    let span = 1.0 / f32(input.frame.w);
    let cell = vec2<f32>(input.frame.yz) * span;
    let inset = 0.5 * exp2(ceil(level)) / size;
    let uv = clamp(input.uv, cell + inset, cell + span - inset);
    let coverage = textureSampleLevel(impostor_albedo, impostor_sampler, uv,
        input.frame.x, level).a;
    if coverage * (1.0 + 0.25 * level) < 0.5 {
        discard;
    }
}
