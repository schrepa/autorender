import package::engine::detail::{ViewDetail, fits, handover};
import package::engine::drawn::{drawn_in_view, drawn_prepass_pyramid, drawn_view_sees,
    drawn_views};
import package::engine::geometry::declarations::{FADE_LEVEL_MAX, faded_slot, faded_slot_slot};
import package::engine::view::seen_share;
import package::generated::scatter::{
    CullFrame, CullGroup, CullView, DEAD, NO_LAYER, SCATTER_GROUP, chosen_entry_card,
    chosen_entry_fading, chosen_entry_level, chosen_entry_meshes, chosen_entry_step,
    chosen_entry_with_card, chosen_entry_with_fading, chosen_entry_with_level,
    chosen_entry_with_meshes, chosen_entry_with_step, scatter_terrain_hidden,
    scatter_terrain_layer, scatter_terrain_rank
};
import package::place::placed_tiles::{Pose, Record};
import package::scatter::faces::placed_tiles::{placed_impostor, placed_pose};

const BOX_SLACK: f32 = 1.0e-4;

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
@group(1) @binding(8) var<storage, read_write> picks: array<u32>;

fn append(view: u32, index: u32, drawn: CullGroup, level: u32, entry: u32, owns: bool) {
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
    if owns {
        ids[start + slot] = entry;
    }
}

fn list(view: u32, record: Record, group: CullGroup, level: u32, entry: u32) {
    append(view, record.group, group, level, entry, true);
    for (var part = 1u; part <= group.parts; part++) {
        append(view, record.group + part, groups[record.group + part], level, entry, false);
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
    atomicAdd(&card_args[frame.counts.w + 1u].instance_count, 1u);
}

fn box_seen(view: u32, pose: Pose, group: CullGroup) -> bool {
    let reach = group.middle.w;
    if !(reach >= 0.0) || view >= arrayLength(&drawn_views) || drawn_views[view].frustum == 0u {
        return true;
    }
    let middle = vec4<f32>(group.middle.xyz, 1.0);
    let center = vec3<f32>(dot(pose.r0, middle), dot(pose.r1, middle), dot(pose.r2, middle));
    for (var plane = 0u; plane < 6u; plane++) {
        let facing = drawn_views[view].planes[plane];
        let along = facing.x * pose.r0.xyz + facing.y * pose.r1.xyz + facing.z * pose.r2.xyz;
        let ahead = dot(facing.xyz, center);
        let spread = dot(abs(along), group.half.xyz) + reach * length(facing.xyz);
        let slack = BOX_SLACK * (abs(ahead) + abs(facing.w) + spread);
        if ahead + facing.w + spread + slack < 0.0 {
            return false;
        }
    }
    return true;
}

fn chosen(view: u32, fading: u32, id: u32, record: Record, group: CullGroup, wanted: CullView,
        away: f32, meshes: bool) -> u32 {
    if wanted.eye.w <= 0.0 {
        if !meshes {
            return 0u;
        }
        list(view, record, group, 0u, id);
        return chosen_entry_with_meshes(0u, true);
    }
    let detail = measure(wanted);
    let near = mix(record.center, wanted.eye.xyz, min(record.radius / max(away, 1.0e-3), 1.0));
    var level = 0u;
    for (var l = 1u; l < group.camera; l++) {
        if fits(detail, groups[record.group].errors[l - 1u] * record.scale, near) {
            level = l;
        }
    }
    var share = 0.0;
    if wanted.shape.y != 0u && frame.strides.z != 0u && group.impostor > 0.0 {
        share = handover(detail, group.impostor * record.scale, near);
    }
    let step = u32(share * f32(FADE_LEVEL_MAX));
    var word = chosen_entry_with_step(0u, step);
    if step > 0u {
        card(view, id, faded_slot(group.record, step, false));
        word = chosen_entry_with_card(word, true);
    }
    if step >= FADE_LEVEL_MAX || !meshes {
        return word;
    }
    word = chosen_entry_with_level(chosen_entry_with_meshes(word, true), level);
    if step == 0u || fading == view || faded_slot_slot(id) != id {
        list(view, record, group, level, id);
        return word;
    }
    list(fading, record, group, level, faded_slot(id, step, true));
    return chosen_entry_with_fading(word, true);
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
    var pose: Pose;
    var posed = false;
    var picked = 0u;
    for (var view = 0u; view < frame.counts.w; view++) {
        let wanted = views[view];
        if !drawn_in_view(wanted.shape.w, frame.drawn.x)
                || !drawn_view_sees(wanted.shape.w, record.center - reach, record.center + reach) {
            continue;
        }
        let away = distance(record.center, wanted.eye.xyz);
        let casts = wanted.shape.z != 0u;
        let skipped = (casts && hidden) || (!casts && retired)
            || !seen_in(wanted, record.radius, away);
        if (view == 0u && retired) || (view != 0u && skipped) {
            continue;
        }
        if !posed {
            pose = placed_pose(id.x);
            posed = true;
        }
        let meshes = box_seen(wanted.shape.w, pose, group);
        let fading = select(view, frame.counts.w, view == 0u);
        let word = chosen(view, fading, id.x, record, group, wanted, away, meshes);
        if view == 0u {
            picked = word;
        }
    }
    picks[id.x] = picked;
}

const DEPTH_SHARE: f32 = 2.44140625e-4; // optics: 2^-12, far above either side's rounding
const DEPTH_FLOOR: f32 = 9.5367431640625e-7; // optics: 2^-20, rounding next to the far plane

fn pyramid_texel(at: vec2<u32>, level: u32) -> f32 {
    return textureLoad(drawn_prepass_pyramid, at, level).r;
}

fn pyramid_hidden(center: vec3<f32>, extent: vec3<f32>) -> bool {
    let levels = textureNumLevels(drawn_prepass_pyramid);
    let render = frame.drawn.yz;
    let size = vec2<f32>(render);
    let drawn = views[0].shape.w;
    if any(render == vec2<u32>(0u)) || drawn >= arrayLength(&drawn_views)
            || any(textureDimensions(drawn_prepass_pyramid, 0u) * 2u < render) {
        return false;
    }
    let clip_from = drawn_views[drawn].rasterized;
    var low = vec2<f32>(3.0e38);
    var high = vec2<f32>(-3.0e38);
    var nearest = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let side = vec3<f32>(select(-1.0, 1.0, (corner & 1u) != 0u),
            select(-1.0, 1.0, (corner & 2u) != 0u), select(-1.0, 1.0, (corner & 4u) != 0u));
        let clip = clip_from * vec4<f32>(center + extent * side, 1.0);
        if !(clip.w > 1.0e-6) || !(clip.z <= clip.w) {
            return false;
        }
        let ndc = clip.xyz / clip.w;
        low = min(low, ndc.xy);
        high = max(high, ndc.xy);
        nearest = max(nearest, ndc.z);
    }
    let first = clamp(floor(vec2<f32>(low.x * 0.5 + 0.5, 0.5 - high.y * 0.5) * size) - 1.0,
        vec2<f32>(0.0), size - 1.0);
    let last = clamp(floor(vec2<f32>(high.x * 0.5 + 0.5, 0.5 - low.y * 0.5) * size) + 1.0,
        vec2<f32>(0.0), size - 1.0);
    var a = vec2<u32>(first) / 2u;
    var b = vec2<u32>(last) / 2u;
    var level = 0u;
    loop {
        if (b.x - a.x <= 1u && b.y - a.y <= 1u) || level + 1u >= levels {
            break;
        }
        a = a / 2u;
        b = b / 2u;
        level += 1u;
    }
    if b.x - a.x > 1u || b.y - a.y > 1u {
        return false;
    }
    let farthest = min(min(pyramid_texel(a, level), pyramid_texel(vec2<u32>(b.x, a.y), level)),
        min(pyramid_texel(vec2<u32>(a.x, b.y), level), pyramid_texel(b, level)));
    return nearest * (1.0 + DEPTH_SHARE) + DEPTH_FLOOR < farthest;
}

fn posed_middle(pose: Pose, local: vec3<f32>) -> vec3<f32> {
    let point = vec4<f32>(local, 1.0);
    return vec3<f32>(dot(pose.r0, point), dot(pose.r1, point), dot(pose.r2, point));
}

fn meshes_hidden(pose: Pose, group: CullGroup) -> bool {
    let reach = group.middle.w;
    if !(reach >= 0.0) {
        return false;
    }
    let half = group.half.xyz;
    let extent = vec3<f32>(dot(abs(pose.r0.xyz), half), dot(abs(pose.r1.xyz), half),
        dot(abs(pose.r2.xyz), half)) + reach;
    return pyramid_hidden(posed_middle(pose, group.middle.xyz), extent);
}

fn card_hidden(pose: Pose, group: CullGroup) -> bool {
    let impostor = placed_impostor(group.record);
    let squared = dot(impostor.extent, impostor.extent);
    let deepest = max(impostor.sphere.w * impostor.sphere.w, squared);
    let radius = sqrt(2.0 * squared + deepest);
    if !(radius > 0.0) {
        return false;
    }
    let extent = radius * vec3<f32>(length(pose.r0.xyz), length(pose.r1.xyz),
        length(pose.r2.xyz));
    return pyramid_hidden(posed_middle(pose, impostor.sphere.xyz), extent);
}

@compute @workgroup_size(SCATTER_GROUP)
fn cull_colour(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.counts.x {
        return;
    }
    let word = picks[id.x];
    if word == 0u {
        return;
    }
    let record = records[id.x];
    if record.group == DEAD {
        return;
    }
    let group = groups[record.group];
    let pose = placed_pose(id.x);
    let lists = frame.counts.w;
    let step = chosen_entry_step(word);
    if chosen_entry_card(word) && !card_hidden(pose, group) {
        card(lists, id.x, faded_slot(group.record, step, false));
    }
    if !chosen_entry_meshes(word) || meshes_hidden(pose, group) {
        return;
    }
    let level = chosen_entry_level(word);
    if chosen_entry_fading(word) {
        list(lists + 2u, record, group, level, faded_slot(id.x, step, true));
        return;
    }
    list(lists + 1u, record, group, level, id.x);
}
