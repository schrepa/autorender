import package::engine::drawn::{drawn_in_view, drawn_view_sees};
import package::engine::geometry::declarations::{FADE_LEVEL_MAX, faded_slot, faded_slot_slot};
import package::engine::view::seen_share;
import package::generated::scatter::{
    CullFrame, CullGroup, CullView, DEAD, FADED, NO_LAYER, SCATTER_GROUP, SHOWN_CARD, SHOWN_NONE,
    scatter_fade, scatter_fade_adopt, scatter_fade_previous, scatter_fade_progress,
    scatter_fade_shown, scatter_terrain_hidden, scatter_terrain_layer, scatter_terrain_rank
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
    return rank < seen_share(2.0 * radius * frame.fade.w / max(away, 1.0e-3));
}

struct CardArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

@group(1) @binding(0) var<uniform> frame: CullFrame;
@group(1) @binding(1) var<storage, read_write> records: array<Record>;
@group(1) @binding(2) var<storage, read> groups: array<CullGroup>;
@group(1) @binding(3) var<storage, read_write> ids: array<u32>;
@group(1) @binding(4) var<storage, read_write> draws: array<DrawArgs>;
@group(1) @binding(5) var<storage, read> views: array<CullView>;
@group(1) @binding(6) var<storage, read_write> cards: array<vec2<u32>>;
@group(1) @binding(7) var<storage, read_write> card_args: array<CardArgs>;

fn shown_at(record: Record, group: CullGroup, retires: bool, rank: f32, away: f32) -> u32 {
    if retires && !placed_seen(record.radius, rank, away) {
        return SHOWN_NONE;
    }
    if frame.eye.w <= 0.0 {
        return 0u;
    }
    if frame.strides.z != 0u && group.impostor > 0.0 && record.radius * frame.fade.z <= away {
        return SHOWN_CARD;
    }
    var level = 0u;
    for (var l = 1u; l < group.camera; l++) {
        if group.errors[l - 1u] * record.scale * frame.eye.w + record.radius <= away {
            level = l;
        }
    }
    return level;
}

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

fn append_caster(view: u32, index: u32, drawn: CullGroup, level: u32, entry: u32) {
    if drawn.casters == 0u {
        return;
    }
    append(view, index, drawn, min(level, drawn.casters - 1u), entry);
}

fn list_casters(view: u32, record: Record, group: CullGroup, level: u32, entry: u32) {
    append_caster(view, record.group, group, level, entry);
    for (var part = 1u; part <= group.parts; part++) {
        append_caster(view, record.group + part, groups[record.group + part], level, entry);
    }
}

fn fits(view: CullView, error: f32, radius: f32, away: f32) -> bool {
    let reach = error * view.eye.w;
    if view.shape.x != 0u {
        return reach + radius <= away;
    }
    return reach <= 1.0;
}

fn seen_in(view: CullView, radius: f32, away: f32) -> bool {
    if view.eye.w <= 0.0 {
        return true;
    }
    return radius * view.eye.w >= select(1.0, away, view.shape.x != 0u);
}

fn view_level(view: CullView, record: Record, group: CullGroup, away: f32) -> u32 {
    var level = 0u;
    if view.eye.w <= 0.0 {
        return level;
    }
    for (var l = 1u; l < group.camera; l++) {
        if fits(view, group.errors[l - 1u] * record.scale, record.radius, away) {
            level = l;
        }
    }
    return level;
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

fn show(id: u32, record: Record, group: CullGroup, way: u32) {
    if way == SHOWN_NONE {
        return;
    }
    if way == SHOWN_CARD {
        card(0u, id, faded_slot(group.record, FADE_LEVEL_MAX, false));
        return;
    }
    list(0u, record, group, way, id);
}

fn show_fading(id: u32, record: Record, group: CullGroup, way: u32, fade: f32, out: bool) {
    if way == SHOWN_NONE {
        return;
    }
    if way == SHOWN_CARD {
        card(0u, id, faded_slot(group.record, u32(fade * f32(FADE_LEVEL_MAX)), out));
        return;
    }
    if faded_slot_slot(id) != id {
        if !out {
            list(0u, record, group, way, id);
        }
        return;
    }
    list(frame.counts.w, record, group, way,
        faded_slot(id, u32(fade * f32(FADE_LEVEL_MAX)), out));
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
    let away = distance(record.center, frame.eye.xyz);
    let layer = scatter_terrain_layer(record.terrain);
    let retires = layer != NO_LAYER;
    let rank = f32(scatter_terrain_rank(record.terrain)) / 1023.0;
    let hidden = scatter_terrain_hidden(record.terrain);
    var shown = SHOWN_NONE;
    var previous = SHOWN_NONE;
    var progress = FADED;
    if frame.eye.w <= 0.0 {
        if !hidden {
            shown = shown_at(record, group, retires, rank, away);
        }
        previous = shown;
    } else {
        var nearer = SHOWN_NONE;
        var farther = SHOWN_NONE;
        if !hidden {
            let margin = frame.fade.y;
            nearer = shown_at(record, group, retires, rank, away * (1.0 - margin));
            farther = shown_at(record, group, retires, rank, away * (1.0 + margin));
        }
        if scatter_fade_adopt(record.state) {
            if !hidden {
                shown = shown_at(record, group, retires, rank, away);
            }
            previous = shown;
        } else {
            shown = scatter_fade_shown(record.state);
            previous = scatter_fade_previous(record.state);
            progress = scatter_fade_progress(record.state);
            var wanted = max(nearer, min(shown, farther));
            if shown == SHOWN_NONE && previous == SHOWN_NONE && !hidden {
                wanted = shown_at(record, group, retires, rank, away);
            }
            if progress >= FADED {
                if wanted != shown {
                    previous = shown;
                    shown = wanted;
                    progress = 0u;
                }
            } else if wanted == previous {
                previous = shown;
                shown = wanted;
                progress = FADED - progress;
            }
            progress = min(progress + u32(frame.fade.x * f32(FADED)), FADED);
        }
        records[id.x].state = scatter_fade(shown, previous, progress, false);
    }
    let fade = f32(progress) / f32(FADED);
    let kept = select(shown, previous, shown == SHOWN_NONE && progress < FADED);
    let reach = vec3<f32>(record.radius);
    for (var view = 0u; view < frame.counts.w; view++) {
        let wanted = views[view];
        if !drawn_in_view(wanted.shape.w, frame.drawn.x)
                || !drawn_view_sees(wanted.shape.w, record.center - reach, record.center + reach) {
            continue;
        }
        if view == 0u {
            if progress < FADED {
                show_fading(id.x, record, group, shown, fade, false);
                show_fading(id.x, record, group, previous, fade, true);
            } else {
                show(id.x, record, group, shown);
            }
            continue;
        }
        let near = distance(record.center, wanted.eye.xyz);
        if kept == SHOWN_NONE || !seen_in(wanted, record.radius, near) {
            continue;
        }
        if wanted.shape.y != 0u && frame.strides.z != 0u && group.impostor > 0.0
                && fits(wanted, group.impostor * record.scale, record.radius, near) {
            card(view, id.x, faded_slot(group.record, FADE_LEVEL_MAX, false));
            continue;
        }
        let level = view_level(wanted, record, group, near);
        if wanted.shape.z != 0u {
            list_casters(view, record, group, level, id.x);
        } else {
            list(view, record, group, level, id.x);
        }
    }
}
