// Placed objects: each frame, every object's bounding sphere
// against each view's frustum (the camera's, then those of the shadow maps
// drawn this frame), and the level of detail it draws at, chosen from the
// camera in every view so shadows match what casts them, by the same rule
// as GPU presentation's (`presentation.wgsl`): the coarsest level whose
// geometric error, projected at the object's distance, stays within the
// pixel error. Visible objects append their pose index to their draw
// group's list for that view and level and count in its indirect draw.
// Where the camera sees an object's impostor within the pixel error (on
// frames that draw impostors), it appends its pose and
// impostor to the camera's cards instead, drawn facing it; the near
// cascades cast it as a card facing the sun, and the moving
// map leaves it out. The kept shadow map, redrawn only where objects come
// and go, holds every object as seen from afar: its impostor, cast as a
// card facing the sun, or the coarsest level its texels hide.
// The camera switches how it shows an object (a level, its card or
// nothing, past the density its layer keeps) a margin past each switch's
// distance, and fades every switch over frames: while one fades, the
// object shows both ways, each on its share of the pixels.

struct Frame {
    // The camera's eye (world) and pixels per world unit at unit distance
    // over the pixel error; 0 keeps every object at its finest level.
    eye: vec4<f32>,
    // Objects, draw groups, levels each group lists, and views.
    counts: vec4<u32>,
    // The IDs and draws each view's lists take, whether the camera's view
    // draws impostors, and the cards each view's list takes.
    strides: vec4<u32>,
    // Fades: the share of a switch this frame fades (1
    // switches at once; negative on exact frames, which show objects as
    // they stand and advance nothing) and the share of its distance past a
    // switch the eye goes before the camera switches back. The camera's
    // fading objects draw from the lists of the view after the last.
    fade: vec4<f32>,
}

// A view: its frustum planes (world, inside positive; a shadow map's depth
// planes pass everything, but a kept level's far one), the radius objects
// need to draw in it (x, half its texel, which also bounds the error a
// caster coarser than the camera's may add), the geometric error its texels
// hide (y, world units; 0 for the camera), whether it is a kept map's or
// kept level's (z, 1) and whether it is a near cascade's (w, 1).
struct View {
    planes: array<vec4<f32>, 6>,
    limits: vec4<f32>,
}

// One object: its bounding sphere's center (world) and radius, its draw
// group and scale, the terrain it stands on and how far its base sinks
// (`scatter_snap.wgsl`), and how the camera shows it.
struct Record {
    center: vec3<f32>,
    radius: f32,
    group: u32,
    scale: f32,
    terrain: u32,
    sink: f32,
    state: u32,
    pad_0: u32,
    pad_1: u32,
    pad_2: u32,
}

// How the camera shows an object: one of its levels (0 to
// 4), its card, or nothing; coarser ways number higher.
const SHOWN_CARD: u32 = 5u;
const SHOWN_NONE: u32 = 6u;
// An object's state: how the camera shows it (bits 0 to 2), how it showed
// it before (3 to 5) and how far that switch has faded, of `FADED` (from
// bit 6). The renderer writes objects new to the view showing nothing, so
// they fade in, and those it rewrites `STATE_ADOPT`: shown at once as the
// camera would show them (`scatter.rs`).
const STATE_ADOPT: u32 = 0xffffffffu;
const FADED: u32 = 65535u;
// A fading-list entry's slot bits and its flag of fading out
// (`meshes/declarations.wgsl`).
const FADE_SLOT: u32 = 0x00ffffffu;
const FADE_OUT: u32 = 0x80000000u;

// A draw group: where its objects' lists start, how many objects it has,
// its levels of detail, how many of them the camera draws (coarser ones
// serve shadow maps alone), their errors (levels 1 to 4, mesh units), and
// its impostor's error (mesh units; 0 without one) and record.
struct Group {
    ids: u32,
    count: u32,
    levels: u32,
    camera: u32,
    errors: vec4<f32>,
    impostor: f32,
    record: u32,
    pad: vec2<u32>,
}

// An indexed indirect draw, counting its instances.
struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// How each terrain's placement layers thin with distance,
// `THIN_LAYERS` entries a terrain: the share standing far away, the meters
// the layer's own tiles and its far tiles reach (0 where they stay resident
// everywhere), and 1 where the layer thins at all.
const THIN_LAYERS: u32 = 64u;
// Where an object's layer and its rank under the density lie in its
// terrain word (`scatter.rs`).
const TERRAIN_INDEX: u32 = 0x1ffu;
const LAYER_SHIFT: u32 = 9u;
const NO_LAYER: u32 = 0x7fu;
const RANK_SHIFT: u32 = 21u;

// The share of a layer's density `away` meters from the eye:
// all of it within seven tenths of its tiles' reach, falling to its
// far share across most of the rest, then to none across the same part of
// its far tiles' reach unless those stay resident everywhere: objects
// ranked past it hide before their tiles leave, so density never steps
// where tiles end.
fn thinned(thin: vec4<f32>, away: f32) -> f32 {
    if thin.w == 0.0 {
        return 1.0;
    }
    var keep = 1.0;
    if thin.y > 0.0 {
        keep = mix(1.0, thin.x, smoothstep(0.7 * thin.y, 0.95 * thin.y, away));
    }
    if thin.z > 0.0 {
        keep *= 1.0 - smoothstep(0.7 * thin.z, 0.95 * thin.z, away);
    }
    return keep;
}

// A non-indexed indirect draw, counting its instances.
struct CardArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

// The card word's flags of a fading card and of one fading out
// (`impostor_frames.wgsl`).
const FADING: u32 = 0x80000000u;
const FADING_OUT: u32 = 0x40000000u;

// The draw group of slots no object holds, and the terrain word's flag of
// objects standing far away hidden where tiles holding them instead are
// resident (`scatter.rs`).
const DEAD: u32 = 0xffffffffu;
const HIDDEN: u32 = 0x80000000u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read_write> records: array<Record>;
@group(0) @binding(2) var<storage, read> groups: array<Group>;
@group(0) @binding(3) var<storage, read_write> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> draws: array<DrawArgs>;
@group(0) @binding(5) var<storage, read> views: array<View>;
// Each view's impostor cards: each one's pose and impostor record, and
// their draws.
@group(0) @binding(6) var<storage, read_write> cards: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read_write> card_args: array<CardArgs>;
@group(0) @binding(8) var<storage, read> thins: array<vec4<f32>>;

// Whether a sphere at `center` with `radius` reaches inside every plane of
// view `view`.
fn inside(view: u32, center: vec3<f32>, radius: f32) -> bool {
    for (var plane = 0u; plane < 6u; plane++) {
        let bound = views[view].planes[plane];
        if dot(bound.xyz, center) + bound.w < -radius {
            return false;
        }
    }
    return true;
}

// How the camera shows `record` (of draw group `group`) from `away` meters:
// nothing past the density its layer keeps there (`thin` and `rank`, where
// `thinning`), its card where its impostor's texels show within the pixel
// error on frames drawing impostors, else the coarsest level it draws
// whose error shows within it; exact frames show the finest level.
fn shown_at(record: Record, group: Group, thinning: bool, thin: vec4<f32>, rank: f32,
        away: f32) -> u32 {
    if thinning && rank >= thinned(thin, away) {
        return SHOWN_NONE;
    }
    if frame.eye.w <= 0.0 {
        return 0u;
    }
    let impostor = group.impostor * record.scale;
    if frame.strides.z != 0u && impostor > 0.0 && impostor * frame.eye.w + record.radius <= away {
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

// Appends `entry` to view `view`'s list of level `level` of `record`'s draw
// group `group`, counting it in the list's draw.
fn list(view: u32, record: Record, group: Group, level: u32, entry: u32) {
    let draw = view * frame.strides.y + record.group * frame.counts.z + level;
    let slot = atomicAdd(&draws[draw].instance_count, 1u);
    ids[view * frame.strides.x + group.ids + level * group.count + slot] = entry;
}

// Appends object `id`'s card with batch word `word` to view `view`'s cards.
fn card(view: u32, id: u32, word: u32) {
    let index = atomicAdd(&card_args[view].instance_count, 1u);
    cards[view * frame.strides.w + index] = vec2<u32>(id, word);
}

// Shows object `id` to the camera as `way`, steadily.
fn show(id: u32, record: Record, group: Group, way: u32) {
    if way == SHOWN_NONE {
        return;
    }
    if way == SHOWN_CARD {
        card(0u, id, group.record);
        return;
    }
    list(0u, record, group, way, id);
}

// Shows object `id` to the camera as `way` fading in, or out with `out`,
// `fade` of its switch done: its card's word carries the fade, its levels
// go to the camera's fading lists with it. Objects whose words cannot
// carry the fade show steadily fading in and not at all fading out.
fn show_fading(id: u32, record: Record, group: Group, way: u32, fade: f32, out: bool) {
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

// One object (x) in every view: its record and draw group read once, dead
// objects leave before any view, and how the camera shows it advances.
@compute @workgroup_size(64)
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
    // Objects ranked past the density their distance allows show nothing,
    // nor do those standing far away where tiles holding
    // them instead are resident.
    let layer = (record.terrain >> LAYER_SHIFT) & NO_LAYER;
    let thin_index = (record.terrain & TERRAIN_INDEX) * THIN_LAYERS + layer;
    let thinning = layer != NO_LAYER && thin_index < arrayLength(&thins);
    var thin = vec4<f32>(0.0);
    if thinning {
        thin = thins[thin_index];
    }
    let rank = f32((record.terrain >> RANK_SHIFT) & 0x3ffu) / 1023.0;
    let hidden = (record.terrain & HIDDEN) != 0u;
    // How the camera shows it. Exact frames show it as it
    // stands, leaving its state; other frames switch a margin past each
    // switch's distance, toward the finer way coming near and the coarser
    // going away, and fade each switch.
    var shown = SHOWN_NONE;
    var previous = SHOWN_NONE;
    var progress = FADED;
    if frame.eye.w <= 0.0 || frame.fade.x < 0.0 {
        if !hidden {
            shown = shown_at(record, group, thinning, thin, rank, away);
        }
        previous = shown;
    } else {
        var nearer = SHOWN_NONE;
        var farther = SHOWN_NONE;
        if !hidden {
            let margin = frame.fade.y;
            nearer = shown_at(record, group, thinning, thin, rank, away * (1.0 - margin));
            farther = shown_at(record, group, thinning, thin, rank, away * (1.0 + margin));
        }
        if record.state == STATE_ADOPT {
            if !hidden {
                shown = shown_at(record, group, thinning, thin, rank, away);
            }
            previous = shown;
        } else {
            shown = record.state & 7u;
            previous = (record.state >> 3u) & 7u;
            progress = record.state >> 6u;
            let wanted = max(nearer, min(shown, farther));
            if progress >= FADED {
                if wanted != shown {
                    previous = shown;
                    shown = wanted;
                    progress = 0u;
                }
            } else if wanted == previous {
                // Switching back before the fade ends turns it around.
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
            // The camera: how it shows the object, and while that fades
            // how it showed it, each on its share of the pixels.
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
        // Shadow maps cast what the camera shows, objects fading out of it
        // until they are gone and those fading in from the start.
        let casts = select(shown, previous, shown == SHOWN_NONE && progress < FADED);
        if casts == SHOWN_NONE {
            continue;
        }
        // Cards near cascades cast too and the moving map
        // leaves out; kept maps cast one wherever their texels are no finer
        // than the impostor's.
        let limits = views[view].limits;
        let kept = limits.z > 0.0;
        let impostor = group.impostor * record.scale;
        let card_shown = frame.strides.z != 0u && impostor > 0.0
            && select(casts == SHOWN_CARD, impostor <= limits.y, kept);
        let leaves_cards = !kept && limits.w == 0.0;
        if record.radius < limits.x || (card_shown && leaves_cards)
            || !inside(view, record.center, record.radius) {
            continue;
        }
        if card_shown {
            card(view, id.x, group.record);
            continue;
        }
        // Shadow maps cast from the level the camera draws, or a coarser one
        // only where half the map's texel hides its error: a coarser caster
        // stands out of the surface the camera draws in every hollow, by
        // more than lookups allow, and shadows it in blotches.
        var level = 0u;
        if kept {
            // The coarsest level half the kept map's texel hides.
            for (var l = 1u; l < group.levels; l++) {
                if group.errors[l - 1u] * record.scale <= limits.x {
                    level = l;
                }
            }
        } else {
            level = select(casts, group.camera - 1u, casts == SHOWN_CARD);
            for (var l = level + 1u; l < group.levels; l++) {
                if group.errors[l - 1u] * record.scale <= limits.x {
                    level = l;
                }
            }
        }
        list(view, record, group, level, id.x);
    }
}
