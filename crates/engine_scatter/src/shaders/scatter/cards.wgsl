import package::engine::shading::{SurfaceInput, SurfacePoint};
import package::engine::slots::{derived, set_direct_softness, set_metalness, set_roughness,
    set_transmission};
import package::face::impostors::{CardDebugTargets, CardShaded, CardVertex, ImpostorRecord,
    placed_card_debug, placed_card_depth, placed_card_level, placed_card_lit, placed_card_pixel,
    placed_card_point, placed_card_reach, placed_card_vertex};
import package::face::instances::{AffinePose, SlotColor};
import package::face::lighting::SURFACE_THIN;
import package::engine::drawn::{INPUT_BATCH, INPUT_DRAW, INPUT_SLOT};
import package::engine::geometry::declarations::faded_slot_slot;
import package::face::debug::debug_record_with_kind;
import package::generated::kinds::KIND_PLACED;
import package::generated::writes::{DepthOutputs, Outputs, depth_outputs, outputs};
import package::scatter::faces::placed_tiles::placed_impostor;
import package::scatter::grounding::{grounded_color, grounded_metalness, grounded_normal,
    grounded_roughness};
import package::scatter::leaves::{carried_look, light_through, varied};
import package::scatter::values::placed_values_of;
import package::engine::presentation::ROUNDING_MARGIN;
import package::engine::view::camera;

@group(2) @binding(0) var<storage, read> poses: array<AffinePose>;
@group(2) @binding(1) var<storage, read> colors: array<SlotColor>;
@group(2) @binding(2) var laid_depth: texture_2d<f32>;

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

fn plant_point(baked: SurfacePoint, slot: u32) -> SurfacePoint {
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

fn placed_card_output(vertex: u32, slot: u32, batch: u32, draw: vec4<u32>) -> PlacedCardVertex {
    let pose = poses[slot];
    let card = placed_card_vertex(vertex, batch, pose, pose, colors[slot],
        placed_impostor(faded_slot_slot(batch)));
    return PlacedCardVertex(card.clip_position, card.frames_ab, card.frame_c, card.world,
        card.normal_x, card.normal_y, card.normal_z, card.depth_axis, card.material, card.batch,
        card.previous, card.crossings, card.slopes_ab, card.slope_c, vec2<u32>(draw.x, slot));
}

fn card_reach(slot: u32, batch: u32, margin: f32) -> f32 {
    return placed_card_reach(poses[slot], placed_impostor(faded_slot_slot(batch)), margin);
}

const UNDRAWN: vec4<f32> = vec4<f32>(2.0, 2.0, 2.0, 1.0);

@vertex
fn vs_placed_card(@builtin(vertex_index) vertex: u32, @location(INPUT_SLOT) slot: u32,
        @location(INPUT_BATCH) batch: u32, @location(INPUT_DRAW) draw: vec4<u32>)
        -> PlacedCardVertex {
    return placed_card_output(vertex, slot, batch, draw);
}

@vertex
fn vs_placed_card_tested(@builtin(vertex_index) vertex: u32, @location(INPUT_SLOT) slot: u32,
        @location(INPUT_BATCH) batch: u32, @location(INPUT_DRAW) draw: vec4<u32>)
        -> PlacedCardVertex {
    var output = placed_card_output(vertex, slot, batch, draw);
    let nearest = card_reach(slot, batch, ROUNDING_MARGIN);
    if nearest < 0.0 {
        output.clip_position = UNDRAWN;
        return output;
    }
    output.clip_position.z = nearest * output.clip_position.w;
    return output;
}

@vertex
fn vs_placed_card_crossing(@builtin(vertex_index) vertex: u32, @location(INPUT_SLOT) slot: u32,
        @location(INPUT_BATCH) batch: u32, @location(INPUT_DRAW) draw: vec4<u32>)
        -> PlacedCardVertex {
    if card_reach(slot, batch, 2.0 * ROUNDING_MARGIN) >= 0.0 {
        var output: PlacedCardVertex;
        output.clip_position = UNDRAWN;
        return output;
    }
    return placed_card_output(vertex, slot, batch, draw);
}

@fragment
fn fs_placed_card(input: PlacedCardVertex) -> DepthOutputs {
    let card = card_of(input);
    let impostor = placed_impostor(faded_slot_slot(card.batch));
    let level = placed_card_level(card, impostor);
    let shaded = placed_card_lit(card, impostor,
        plant_point(placed_card_point(card, impostor, level), input.object.y), level);
    return depth_outputs(shaded.point, shaded.shaded, placed_card_pixel(card),
        shaded.depth);
}

fn nearest_depth(input: PlacedCardVertex) -> f32 {
    let projection = camera.view_projection;
    let axis = input.depth_axis;
    let middle = projection * vec4<f32>(input.world - axis.xyz * axis.w, 1.0);
    let along = projection * vec4<f32>(axis.xyz, 0.0);
    let reach = vec4<f32>(abs(input.world) + abs(axis.xyz) * (abs(axis.w) + 1.0), 1.0);
    let rows = transpose(projection);
    let span = vec2<f32>(dot(abs(rows[2]), reach), dot(abs(rows[3]), reach)) * ROUNDING_MARGIN;
    let low = vec2<f32>(middle.z - along.z, middle.w - along.w) + vec2<f32>(span.x, -span.y);
    let high = vec2<f32>(middle.z + along.z, middle.w + along.w) + vec2<f32>(span.x, -span.y);
    if !(low.y > 0.0 && high.y > 0.0) {
        return 1.0;
    }
    return max(max(low.x, 0.0) / low.y, max(high.x, 0.0) / high.y) * (1.0 + ROUNDING_MARGIN);
}

fn quad_any(here: bool) -> bool {
    let own = select(0.0, 1.0, here);
    let across = dpdxFine(own);
    let down = dpdyFine(own);
    let diagonal = dpdyFine(across);
    return here | (across != 0.0) | (down != 0.0) | (diagonal != 0.0);
}

struct LaidCard {
    card: CardVertex,
    impostor: ImpostorRecord,
    baked: SurfacePoint,
    depth: f32,
    level: f32,
    seen: bool,
}

fn laid_card(input: PlacedCardVertex) -> LaidCard {
    let laid = textureLoad(laid_depth, vec2<i32>(input.clip_position.xy), 0).r;
    if !quad_any(!(nearest_depth(input) < laid)) {
        discard;
    }
    let card = card_of(input);
    let impostor = placed_impostor(faded_slot_slot(card.batch));
    let level = placed_card_level(card, impostor);
    let baked = placed_card_point(card, impostor, level);
    let clip = camera.view_projection * vec4<f32>(baked.world, 1.0);
    let depth = saturate(clip.z / clip.w);
    let seen = depth >= laid;
    if !seen {
        discard;
    }
    return LaidCard(card, impostor, baked, depth, placed_card_level(card, impostor), seen);
}

fn laid_shaded(input: PlacedCardVertex, laid: LaidCard) -> CardShaded {
    return placed_card_lit(laid.card, laid.impostor, plant_point(laid.baked, input.object.y),
        laid.level);
}

@fragment
fn fs_placed_card_laid(input: PlacedCardVertex) -> DepthOutputs {
    let laid = laid_card(input);
    var written: DepthOutputs;
    if laid.seen {
        let shaded = laid_shaded(input, laid);
        written = depth_outputs(shaded.point, shaded.shaded, placed_card_pixel(laid.card),
            laid.depth);
    }
    return written;
}

@fragment
fn fs_placed_card_tested(input: PlacedCardVertex) -> Outputs {
    let laid = laid_card(input);
    var written: Outputs;
    if laid.seen {
        let shaded = laid_shaded(input, laid);
        written = outputs(shaded.point, shaded.shaded, placed_card_pixel(laid.card));
    }
    return written;
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
    let level = placed_card_level(card, impostor);
    let point = plant_point(placed_card_point(card, impostor, level), input.object.y);
    return placed_card_debug(card, point, debug_record_with_kind(input.object.x, KIND_PLACED),
        input.object.y);
}
