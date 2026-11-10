import package::engine::drawn::{drawn_fade, drawn_records};
import package::engine::geometry::declarations::{drawn_slot_moved, drawn_slot_slot, fade_word_on};
import package::engine::view::view_pixel;
import package::face::debug::debug_record_with_kind;
import package::face::instances::{DEBUG_LEVEL_AT, LAST_LEVEL, instance_color, instance_pose,
    instance_previous_pose, last_entry, last_fading_entry, last_late_entry};
import package::generated::writes::{DepthOutputs, depth_outputs};
import package::impostors::faces::impostors::{card_debug, card_depth, card_lit, card_point,
    card_previous, posed_card};
import package::place::impostors::{CardDebugTargets, CardVertex, DEBUG_CARD, ImpostorRecord};

@group(1) @binding(0) var<storage, read> card_records: array<ImpostorRecord>;

const CARD_RECORD: u32 = DEBUG_CARD | (LAST_LEVEL << DEBUG_LEVEL_AT);

struct CardOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
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
    @location(13) @interpolate(flat) slot: u32,
}

fn card_output(vertex: u32, entry: vec2<u32>) -> CardOutput {
    let slot = drawn_slot_slot(entry.x);
    let color = instance_color(slot);
    let pose = instance_pose(slot);
    var previous = pose;
    if drawn_slot_moved(entry.x) {
        previous = instance_previous_pose(slot);
    }
    let made = posed_card(vertex, entry.y, pose, color, card_records[entry.y]);
    let card = made.output;
    return CardOutput(card.clip_position, card.frames_ab, card.frame_c, card.world,
        card.normal_x, card.normal_y, card.normal_z, card.depth_axis, card.material, card.batch,
        card_previous(made.local, previous, color), card.crossings, card.slopes_ab, card.slope_c,
        slot);
}

@vertex
fn vs_card(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> CardOutput {
    return card_output(vertex, last_entry(instance));
}

@vertex
fn vs_late_card(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> CardOutput {
    return card_output(vertex, last_late_entry(instance));
}

@vertex
fn vs_fading_card(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> CardOutput {
    return card_output(vertex, last_fading_entry(instance));
}

fn card_vertex(input: CardOutput) -> CardVertex {
    return CardVertex(input.clip_position, input.frames_ab, input.frame_c, input.world,
        input.normal_x, input.normal_y, input.normal_z, input.depth_axis, input.material,
        input.batch, input.previous, input.crossings, input.slopes_ab, input.slope_c);
}

@fragment
fn fs_card(input: CardOutput) -> DepthOutputs {
    let card = card_vertex(input);
    let fade = drawn_fade(input.slot);
    let shaded = card_lit(card, card_point(card, card_records[input.batch], fade),
        fade_word_on(fade));
    return depth_outputs(shaded.point, shaded.shaded,
        view_pixel(input.clip_position, vec4<f32>(input.previous.xy, 0.0, input.previous.z)),
        shaded.depth);
}

@fragment
fn fs_card_depth(input: CardOutput) -> @builtin(frag_depth) f32 {
    let card = card_vertex(input);
    let fade = drawn_fade(input.slot);
    return card_depth(card, card_point(card, card_records[input.batch], fade), fade_word_on(fade));
}

@fragment
fn fs_debug_card(input: CardOutput) -> CardDebugTargets {
    let card = card_vertex(input);
    let fade = drawn_fade(input.slot);
    let record = debug_record_with_kind(CARD_RECORD, drawn_records[input.slot].kind);
    return card_debug(card, card_point(card, card_records[input.batch], fade),
        vec2<u32>(record, input.slot), input.batch, fade, fade_word_on(fade));
}
