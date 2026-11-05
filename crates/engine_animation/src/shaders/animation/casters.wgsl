import package::engine::materials::{base_color_sampler, base_color_texture, material_maps};
import package::face::caster_routes::{outside_route, route};
import package::face::instances::SlotColor;

@group(2) @binding(0) var<storage, read> colors: array<SlotColor>;

struct CasterOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) tile: u32,
}

@vertex
fn vs_caster(@location(0) position: vec3<f32>, @location(2) tile: u32) -> CasterOutput {
    var output: CasterOutput;
    output.tile = tile;
    output.clip = route(position, tile);
    return output;
}

@fragment
fn fs_caster(input: CasterOutput) {
    if outside_route(input.clip.xy, input.tile) {
        discard;
    }
}

struct MaskedOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv_alpha: vec3<f32>,
    @location(1) @interpolate(flat) tile: u32,
}

@vertex
fn vs_caster_masked(@location(0) position: vec3<f32>, @location(7) uv: vec2<f32>,
        @location(3) slot: u32, @location(2) tile: u32) -> MaskedOutput {
    var output: MaskedOutput;
    output.tile = tile;
    output.clip = route(position, tile);
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
    let cut = masked_cut(input.uv_alpha);
    if cut || outside_route(input.clip.xy, input.tile) {
        discard;
    }
}
