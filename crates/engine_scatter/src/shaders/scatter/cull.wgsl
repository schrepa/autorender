import package::engine::detail::{ViewDetail, fits, handover};
import package::engine::drawn::{drawn_in_view, drawn_view_sees};
import package::engine::geometry::declarations::{FADE_LEVEL_MAX, faded_slot, faded_slot_slot};
import package::engine::view::seen_share;
import package::generated::scatter::{
    CullFrame, CullGroup, CullView, DEAD, NO_LAYER, SCATTER_GROUP, scatter_terrain_hidden,
    scatter_terrain_layer, scatter_terrain_rank
};
import package::place::placed_tiles::Record;

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

fn placed_seen(radius: f32, rank: f32, away: f32) -> bool {
    return rank < seen_share(2.0 * radius * frame.eye.w / max(away, 1.0e-3));
}

struct CardArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

@group(1) @binding(0) var<uniform> frame: CullFrame;
@group(1) @binding(1) var<storage, read> records: array<Record>;
@group(1) @binding(2) var<storage, read> groups: array<CullGroup>;
@group(1) @binding(3) var<storage, read_write> ids: array<u32>;
@group(1) @binding(4) var<storage, read_write> draws: array<DrawArgs>;
@group(1) @binding(5) var<storage, read> views: array<CullView>;
@group(1) @binding(6) var<storage, read_write> cards: array<vec2<u32>>;
@group(1) @binding(7) var<storage, read_write> card_args: array<CardArgs>;

fn append(view: u32, index: u32, drawn: CullGroup, level: u32, entry: u32) {
    if level >= drawn.levels {
        return;
    }
    let start = view * frame.strides.x + drawn.ids + level * drawn.count;
    if start + drawn.count > arrayLength(&ids) {
        left_out();
        return;
    }
    let draw = view * frame.strides.y + index * frame.counts.z + level;
    let slot = atomicAdd(&draws[draw].instance_count, 1u);
    ids[start + slot] = entry;
}

fn list(view: u32, record: Record, group: CullGroup, level: u32, entry: u32) {
    append(view, record.group, group, level, entry);
    for (var part = 1u; part <= group.parts; part++) {
        append(view, record.group + part, groups[record.group + part], level, entry);
    }
}

fn measure(view: CullView) -> ViewDetail {
    let step = 1.0 / view.eye.w;
    if view.shape.x != 0u {
        return ViewDetail(view.eye.xyz, step, 0.0, 1.0);
    }
    return ViewDetail(view.eye.xyz, 0.0, step, 1.0);
}

fn seen_in(view: CullView, radius: f32, away: f32) -> bool {
    if view.eye.w <= 0.0 {
        return true;
    }
    return radius * view.eye.w >= select(1.0, away, view.shape.x != 0u);
}

fn card(view: u32, id: u32, word: u32) {
    let start = view * frame.strides.w;
    if start + frame.strides.w > arrayLength(&cards) {
        left_out();
        return;
    }
    let index = atomicAdd(&card_args[view].instance_count, 1u);
    cards[start + index] = vec2<u32>(id, word);
}

fn left_out() {
    atomicAdd(&card_args[frame.counts.w].instance_count, 1u);
}

fn chosen(view: u32, fading: u32, id: u32, record: Record, group: CullGroup, wanted: CullView,
        away: f32) {
    if wanted.eye.w <= 0.0 {
        list(view, record, group, 0u, id);
        return;
    }
    let detail = measure(wanted);
    let near = mix(record.center, wanted.eye.xyz, min(record.radius / max(away, 1.0e-3), 1.0));
    var level = 0u;
    for (var l = 1u; l < group.camera; l++) {
        if fits(detail, group.errors[l - 1u] * record.scale, near) {
            level = l;
        }
    }
    var share = 0.0;
    if wanted.shape.y != 0u && frame.strides.z != 0u && group.impostor > 0.0 {
        share = handover(detail, group.impostor * record.scale, near);
    }
    let step = u32(share * f32(FADE_LEVEL_MAX));
    if step > 0u {
        card(view, id, faded_slot(group.record, step, false));
    }
    if step >= FADE_LEVEL_MAX {
        return;
    }
    if step == 0u || fading == view || faded_slot_slot(id) != id {
        list(view, record, group, level, id);
        return;
    }
    list(fading, record, group, level, faded_slot(id, step, true));
}

@compute @workgroup_size(SCATTER_GROUP)
fn cull_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.counts.x {
        return;
    }
    let record = records[id.x];
    if record.group == DEAD {
        return;
    }
    let group = groups[record.group];
    let retires = scatter_terrain_layer(record.terrain) != NO_LAYER;
    let rank = f32(scatter_terrain_rank(record.terrain)) / 1023.0;
    let hidden = scatter_terrain_hidden(record.terrain);
    let retired = hidden
        || (retires && !placed_seen(record.radius, rank, distance(record.center, frame.eye.xyz)));
    let reach = vec3<f32>(record.radius);
    for (var view = 0u; view < frame.counts.w; view++) {
        let wanted = views[view];
        if !drawn_in_view(wanted.shape.w, frame.drawn.x)
                || !drawn_view_sees(wanted.shape.w, record.center - reach, record.center + reach) {
            continue;
        }
        let away = distance(record.center, wanted.eye.xyz);
        if view == 0u {
            if !retired {
                chosen(view, frame.counts.w, id.x, record, group, wanted, away);
            }
            continue;
        }
        let casts = wanted.shape.z != 0u;
        if (casts && hidden) || (!casts && retired) || !seen_in(wanted, record.radius, away) {
            continue;
        }
        chosen(view, view, id.x, record, group, wanted, away);
    }
}
