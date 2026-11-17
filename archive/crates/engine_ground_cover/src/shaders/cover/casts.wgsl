import package::cover::faces::cover::cover_cell;
import package::cover::ground::{ordered_height, ordered_to_height};
import package::cover::laws::{BLADE_SIZE, CARD_CELL, ROUNDING};
import package::engine::drawn::{drawn_view_sees, drawn_views};
import package::generated::cover::{CoverBlade, CoverCard, ShapeRows};

@group(1) @binding(8) var<storage, read_write> cover_casts: array<atomic<u32>>;

const PLACE_ROUNDING: f32 = 1.0e-5;
const LEAN_LEAST: f32 = 1.0e-4;
const BLADE_KINDS: u32 = 2u;
const CAST_BOX: u32 = 2u * BLADE_KINDS;
const CELL_CASTS: u32 = CAST_BOX + 6u;

var<workgroup> cell_casts: array<atomic<u32>, CELL_CASTS>;

struct Bounds {
    low: vec3<f32>,
    high: vec3<f32>,
}

fn bounds_around(root: vec3<f32>, low: vec3<f32>, high: vec3<f32>) -> Bounds {
    let margin = PLACE_ROUNDING * max(max(abs(root.x), abs(root.y)), abs(root.z))
        + ROUNDING * max(high.x - low.x, high.y - low.y);
    return Bounds(root + low - vec3<f32>(margin), root + high + vec3<f32>(margin));
}

fn blade_bounds(blade: CoverBlade, shape: ShapeRows) -> Bounds {
    let size = unpack2x16float(blade.size & BLADE_SIZE);
    let most = max(shape.bends.x, shape.bends.z);
    let out = size.x * shape.arc.z * most + 0.5 * size.y;
    let dip = min(size.x * (1.0 - shape.arc.y * most * most), 0.0);
    return bounds_around(blade.root, vec3<f32>(-out, dip, -out), vec3<f32>(out, size.x, out));
}

fn card_bounds(card: CoverCard, shape: ShapeRows) -> Bounds {
    let size = unpack2x16float(card.size);
    let span = cover_cell(card.cell & CARD_CELL).span;
    let bend = abs(size.y) * max(shape.bends.x, LEAN_LEAST);
    let tip = max(1.0, abs(1.0 - shape.arc.y * bend * bend)) + bend * shape.arc.z;
    let out = size.x * (max(max(shape.arc.x, tip), abs(span.z))
        + max(abs(span.x), abs(span.y)));
    return bounds_around(card.root, vec3<f32>(-out), vec3<f32>(out));
}

fn mark_casts(room: vec4<u32>, bounds: Bounds, kind: u32, first: u32, last: u32) {
    let views = min(room.w, arrayLength(&drawn_views));
    let kinds = BLADE_KINDS + room.z;
    for (var view = 0u; view < views; view++) {
        if drawn_views[view].owned == 0u || !drawn_view_sees(view, bounds.low, bounds.high) {
            continue;
        }
        let at = 2u * (view * kinds + kind);
        atomicMax(&cover_casts[at], ~first);
        atomicMax(&cover_casts[at + 1u], last + 1u);
    }
}

fn hold_casts(bounds: Bounds, kind: u32, first: u32, last: u32) {
    atomicMax(&cell_casts[2u * kind], ~first);
    atomicMax(&cell_casts[2u * kind + 1u], last + 1u);
    atomicMax(&cell_casts[CAST_BOX], ~ordered_height(bounds.low.x));
    atomicMax(&cell_casts[CAST_BOX + 1u], ~ordered_height(bounds.low.y));
    atomicMax(&cell_casts[CAST_BOX + 2u], ~ordered_height(bounds.low.z));
    atomicMax(&cell_casts[CAST_BOX + 3u], ordered_height(bounds.high.x));
    atomicMax(&cell_casts[CAST_BOX + 4u], ordered_height(bounds.high.y));
    atomicMax(&cell_casts[CAST_BOX + 5u], ordered_height(bounds.high.z));
}

fn cast_cell(room: vec4<u32>, lane: u32) {
    let views = min(room.w, arrayLength(&drawn_views));
    let kinds = BLADE_KINDS + room.z;
    let low = vec3<f32>(ordered_to_height(~atomicLoad(&cell_casts[CAST_BOX])),
        ordered_to_height(~atomicLoad(&cell_casts[CAST_BOX + 1u])),
        ordered_to_height(~atomicLoad(&cell_casts[CAST_BOX + 2u])));
    let high = vec3<f32>(ordered_to_height(atomicLoad(&cell_casts[CAST_BOX + 3u])),
        ordered_to_height(atomicLoad(&cell_casts[CAST_BOX + 4u])),
        ordered_to_height(atomicLoad(&cell_casts[CAST_BOX + 5u])));
    for (var slot = lane; slot < views * BLADE_KINDS; slot += 64u) {
        let kind = slot % BLADE_KINDS;
        let view = slot / BLADE_KINDS;
        let end = atomicLoad(&cell_casts[2u * kind + 1u]);
        if end == 0u || drawn_views[view].owned == 0u || !drawn_view_sees(view, low, high) {
            continue;
        }
        let at = 2u * (view * kinds + kind);
        atomicMax(&cover_casts[at], atomicLoad(&cell_casts[2u * kind]));
        atomicMax(&cover_casts[at + 1u], end);
    }
}
