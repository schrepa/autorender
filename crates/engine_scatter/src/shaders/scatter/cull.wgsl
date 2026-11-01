import package::engine::view::seen_share;
import package::generated::scatter::{
    CullFrame, CullGroup, CullView, DEAD, FADED, HIDDEN, LAYER_SHIFT, NO_LAYER, RANK_SHIFT,
    SCATTER_GROUP, SHOWN_CARD, SHOWN_NONE, STATE_ADOPT
};
import package::place::instances::FADE_SLOT;
import package::place::placed_tiles::Record;

const FADE_OUT: u32 = 0x80000000u;

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

const FADING: u32 = 0x80000000u;
const FADING_OUT: u32 = 0x40000000u;

@group(1) @binding(0) var<uniform> frame: CullFrame;
@group(1) @binding(1) var<storage, read_write> records: array<Record>;
@group(1) @binding(2) var<storage, read> groups: array<CullGroup>;
@group(1) @binding(3) var<storage, read_write> ids: array<u32>;
@group(1) @binding(4) var<storage, read_write> draws: array<DrawArgs>;
@group(1) @binding(5) var<storage, read> views: array<CullView>;
@group(1) @binding(6) var<storage, read_write> cards: array<vec2<u32>>;
@group(1) @binding(7) var<storage, read_write> card_args: array<CardArgs>;

fn inside(view: u32, center: vec3<f32>, radius: f32) -> bool {
    for (var plane = 0u; plane < 6u; plane++) {
        let bound = views[view].planes[plane];
        if dot(bound.xyz, center) + bound.w < -radius {
            return false;
        }
    }
    return true;
}

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

fn caster_level(record: Record, group: CullGroup, half_texel: f32) -> u32 {
    var level = 0u;
    for (var l = 1u; l < group.casters; l++) {
        if group.errors[l - 1u] * record.scale <= half_texel {
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
        card(0u, id, group.record);
        return;
    }
    list(0u, record, group, way, id);
}

fn show_fading(id: u32, record: Record, group: CullGroup, way: u32, fade: f32, out: bool) {
    if way == SHOWN_NONE {
        return;
    }
    if way == SHOWN_CARD {
        if group.record >= 0x8000u {
            if !out {
                card(0u, id, group.record);
            }
            return;
        }
        card(0u, id, group.record | FADING | select(0u, FADING_OUT, out)
            | (u32(fade * 16383.0) << 16u));
        return;
    }
    if id > FADE_SLOT {
        if !out {
            list(0u, record, group, way, id);
        }
        return;
    }
    list(frame.counts.w, record, group, way,
        id | (u32(fade * 127.0) << 24u) | select(0u, FADE_OUT, out));
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
    let layer = (record.terrain >> LAYER_SHIFT) & NO_LAYER;
    let retires = layer != NO_LAYER;
    let rank = f32((record.terrain >> RANK_SHIFT) & 0x3ffu) / 1023.0;
    let hidden = (record.terrain & HIDDEN) != 0u;
    var shown = SHOWN_NONE;
    var previous = SHOWN_NONE;
    var progress = FADED;
    if frame.eye.w <= 0.0 || frame.fade.x < 0.0 {
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
        if record.state == STATE_ADOPT {
            if !hidden {
                shown = shown_at(record, group, retires, rank, away);
            }
            previous = shown;
        } else {
            shown = record.state & 7u;
            previous = (record.state >> 3u) & 7u;
            progress = record.state >> 6u;
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
        records[id.x].state = shown | (previous << 3u) | (progress << 6u);
    }
    let fade = f32(progress) / f32(FADED);
    for (var view = 0u; view < frame.counts.w; view++) {
        if view == 0u {
            if !inside(0u, record.center, record.radius) {
                continue;
            }
            if progress < FADED {
                show_fading(id.x, record, group, shown, fade, false);
                show_fading(id.x, record, group, previous, fade, true);
            } else {
                show(id.x, record, group, shown);
            }
            continue;
        }
        let casts = select(shown, previous, shown == SHOWN_NONE && progress < FADED);
        if casts == SHOWN_NONE || group.casters == 0u {
            continue;
        }
        let limits = views[view].limits;
        if record.radius < limits.x || !inside(view, record.center, record.radius) {
            continue;
        }
        let impostor = group.impostor * record.scale;
        if frame.strides.z != 0u && impostor > 0.0 && impostor <= limits.y {
            card(view, id.x, group.record);
            continue;
        }
        append(view, record.group, group, caster_level(record, group, limits.x), id.x);
    }
}
