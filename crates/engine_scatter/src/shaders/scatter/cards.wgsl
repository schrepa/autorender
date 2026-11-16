import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::slots::{derived, set_direct_softness, set_metalness, set_roughness,
    set_transmission};
import package::face::impostors::{CardDebugTargets, CardVertex, ImpostorRecord,
    placed_card_debug, placed_card_depth, placed_card_lit, placed_card_pixel, placed_card_point,
    placed_card_vertex};
import package::face::instances::{AffinePose, SlotColor};
import package::face::lighting::SURFACE_THIN;
import package::engine::drawn::{INPUT_BATCH, INPUT_DRAW, INPUT_SLOT};
import package::engine::geometry::declarations::faded_slot_slot;
import package::face::debug::debug_record_with_kind;
import package::generated::kinds::KIND_PLACED;
import package::generated::writes::{DepthOutputs, depth_outputs};
import package::scatter::faces::placed_tiles::placed_impostor;
import package::scatter::grounding::{grounded_color, grounded_metalness, grounded_normal,
    grounded_roughness};
import package::scatter::leaves::{carried_look, light_through, varied};
import package::scatter::values::placed_values_of;

@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;

struct PlacedCardVertex {
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
    @location(13) @interpolate(flat) object: vec2<u32>,
}

fn card_of(input: PlacedCardVertex) -> CardVertex {
    return CardVertex(input.clip_position, input.frames_ab, input.frame_c, input.world,
        input.normal_x, input.normal_y, input.normal_z, input.depth_axis, input.material,
        input.batch, input.previous, input.crossings, input.slopes_ab, input.slope_c);
}

fn plant_point(card: CardVertex, impostor: ImpostorRecord, slot: u32) -> SurfacePoint {
    let baked = placed_card_point(card, impostor);
    var input: SurfaceInput;
    input.world = baked.world;
    input.kind = KIND_PLACED;
    input.object = slot;
    var point = baked;
    var color = baked.base_color;
    if baked.kind == SURFACE_THIN {
        let values = placed_values_of(KIND_PLACED, slot);
        let look = carried_look(values.look);
        color = varied(color, values.tint, look);
        point = set_transmission(point, light_through(look, baked.occlusion));
        point = set_direct_softness(point, look.softness);
    }
    point.base_color = grounded_color(input, color);
    point.normal = grounded_normal(input, baked.normal);
    point = set_roughness(point, grounded_roughness(input, baked.roughness));
    point = set_metalness(point, grounded_metalness(input, baked.metalness));
    return derived(point);
}

@vertex
fn vs_placed_card(@builtin(vertex_index) vertex: u32, @location(INPUT_SLOT) slot: u32,
        @location(INPUT_BATCH) batch: u32, @location(INPUT_DRAW) draw: vec4<u32>)
        -> PlacedCardVertex {
    let pose = poses[slot];
    let card = placed_card_vertex(vertex, batch, pose, pose, colors[slot],
        placed_impostor(faded_slot_slot(batch)));
    return PlacedCardVertex(card.clip_position, card.frames_ab, card.frame_c, card.world,
        card.normal_x, card.normal_y, card.normal_z, card.depth_axis, card.material, card.batch,
        card.previous, card.crossings, card.slopes_ab, card.slope_c, vec2<u32>(draw.x, slot));
}

@fragment
fn fs_placed_card(input: PlacedCardVertex) -> DepthOutputs {
    let card = card_of(input);
    let impostor = placed_impostor(faded_slot_slot(card.batch));
    let shaded = placed_card_lit(card, impostor, plant_point(card, impostor, input.object.y));
    return depth_outputs(shaded.point, shaded.shaded, placed_card_pixel(card),
        shaded.depth);
}

@fragment
fn fs_placed_card_depth(input: PlacedCardVertex) -> @builtin(frag_depth) f32 {
    let card = card_of(input);
    return placed_card_depth(card, placed_impostor(faded_slot_slot(card.batch)));
}

@fragment
fn fs_debug_placed_card(input: PlacedCardVertex) -> CardDebugTargets {
    let card = card_of(input);
    let impostor = placed_impostor(faded_slot_slot(card.batch));
    return placed_card_debug(card, plant_point(card, impostor, input.object.y),
        debug_record_with_kind(input.object.x, KIND_PLACED), input.object.y);
}
