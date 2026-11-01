import package::face::dither::stored;
import package::face::instances::{
    placed_card_debug, placed_card_pixel, placed_card_shade, placed_card_vertex
};
import package::generated::targets::{DepthViewTargets, depth_view_targets};
import package::place::instances::{AffinePose, CardDebugTargets, CardVertex, SlotColor};

@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;

struct PlacedCardVertex {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frames_ab: vec4<f32>,
    @location(1) frame_c: vec2<f32>,
    @location(2) world: vec3<f32>,
    @location(3) @interpolate(flat) normal_x: vec4<f32>,
    @location(4) @interpolate(flat) normal_y: vec4<f32>,
    @location(5) @interpolate(flat) normal_z: vec3<f32>,
    @location(6) @interpolate(flat) depth_axis: vec4<f32>,
    @location(7) @interpolate(flat) material: vec4<u32>,
    @location(8) @interpolate(flat) batch: u32,
    @location(9) previous: vec3<f32>,
    @location(10) crossings: vec3<f32>,
    @location(11) @interpolate(flat) slopes_ab: vec4<f32>,
    @location(12) @interpolate(flat) slope_c: vec2<f32>,
    @location(13) @interpolate(flat) object: vec2<u32>,
}

fn card_of(input: PlacedCardVertex) -> CardVertex {
    return CardVertex(input.clip_position, input.frames_ab, input.frame_c, input.world,
        input.normal_x, input.normal_y, input.normal_z, input.depth_axis, input.material,
        input.batch, input.previous, input.crossings, input.slopes_ab, input.slope_c);
}

@vertex
fn vs_placed_card(@builtin(vertex_index) vertex: u32, @location(2) slot: u32,
        @location(3) batch: u32, @location(14) draw: vec4<u32>) -> PlacedCardVertex {
    let pose = poses[slot];
    let card = placed_card_vertex(vertex, batch, pose, pose, colors[slot]);
    return PlacedCardVertex(card.clip_position, card.frames_ab, card.frame_c, card.world,
        card.normal_x, card.normal_y, card.normal_z, card.depth_axis, card.material, card.batch,
        card.previous, card.crossings, card.slopes_ab, card.slope_c, vec2<u32>(draw.x, slot));
}

@fragment
fn fs_placed_card(input: PlacedCardVertex) -> DepthViewTargets {
    let card = card_of(input);
    let shaded = placed_card_shade(card);
    var kept = shaded.shaded;
    kept.color = stored(kept.color, input.clip_position.xy);
    return depth_view_targets(kept, placed_card_pixel(card), shaded.depth);
}

@fragment
fn fs_debug_placed_card(input: PlacedCardVertex) -> CardDebugTargets {
    return placed_card_debug(card_of(input), input.object.x, input.object.y);
}
