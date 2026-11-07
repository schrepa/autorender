import package::engine::math::maximum_abs;
import package::engine::mesh::declarations::drawn_slot;
import package::engine::mesh::impostor_frames::{CARD_PHASE_FADING, CARD_PHASE_STEADY,
    FADING_CARD_FADE_MAX, FADING_CARD_INDEX_MAX, fading_card, steady_card};
import package::engine::presentation::{IMPOSTOR_VERTICES, MAX_COEFFICIENT, MAX_PRODUCT_BOUND,
    NO_PARENT, POSE_FLOATS, Pose, ROUNDING_MARGIN, drawn_local, pose_admitted, pose_trs,
    pose_word_blend, pose_word_composed, pose_word_parent_composed, pose_word_valid};
import package::generated::meshes::{ARGS_GROUP, BatchData as Batch, CULL_GROUP,
    CandidateData as Candidate, FIRST_PERSON_BATCH, GROUP, LOD_FADED, LOD_HISTORY_HELD,
    LOD_LEVELS, NO_NODE, PARENT_GROUP, PresentationFrame as Frame, UNWATCHED,
    candidate_state_level, candidate_state_moving, candidate_state_occluded,
    candidate_state_phase_one, candidate_state_phase_two, candidate_state_subpixel,
    candidate_state_valid, candidate_state_view, candidate_state_view_level,
    candidate_state_visible, candidate_state_with_level, candidate_state_with_moving,
    candidate_state_with_occluded, candidate_state_with_phase_one,
    candidate_state_with_phase_two, candidate_state_with_subpixel, candidate_state_with_valid,
    candidate_state_with_view, candidate_state_with_view_level, candidate_state_with_visible,
    lod_fade, lod_fade_history, lod_fade_previous, lod_fade_progress, lod_fade_shown,
    node_word_mobility};
import package::meshes::screen_cull::{F32_MAX, covers_no_pixel, hidden};
import package::meshes::slots::instance_poses;

struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

struct Composed {
    display: Rows,
    current: Rows,
    status: vec4<u32>,
}

struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct Level {
    start: u32,
    end: u32,
    pad0: u32,
    pad1: u32,
}

@group(1) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(1) var<storage, read> locals: array<f32>;
@group(1) @binding(2) var<storage, read> info: array<vec2<u32>>;
@group(1) @binding(3) var<storage, read> authority: array<Rows>;
@group(1) @binding(4) var<storage, read_write> composed: array<Composed>;
@group(1) @binding(5) var<storage, read> parents: array<u32>;
@group(1) @binding(6) var<storage, read> candidates: array<Candidate>;
@group(1) @binding(7) var<storage, read> batches: array<Batch>;
@group(1) @binding(8) var<storage, read_write> poses: array<Rows>;
@group(1) @binding(9) var<storage, read_write> scan: array<u32>;
@group(1) @binding(10) var<storage, read_write> batch_bases: array<u32>;
@group(1) @binding(11) var<storage, read_write> visible_ids: array<u32>;
@group(1) @binding(12) var<storage, read_write> draws: array<DrawArgs>;
@group(1) @binding(14) var<storage, read> slot_candidates: array<Candidate>;
@group(1) @binding(18) var depth_pyramid: texture_2d<f32>;
@group(1) @binding(19) var<storage, read_write> occlusion_counts: array<atomic<u32>>;
@group(1) @binding(23) var<storage, read> node_words: array<u32>;
@group(1) @binding(24) var<storage, read_write> impostor_ids: array<vec2<u32>>;
@group(1) @binding(25) var<storage, read_write> lod_states: array<u32>;
@group(1) @binding(26) var<storage, read_write> fade_cards: array<atomic<u32>>;
@group(1) @binding(27) var<storage, read_write> totals: array<u32, 14>;
@group(2) @binding(0) var<uniform> level: Level;

const IMPOSTOR: u32 = LOD_LEVELS;

var<workgroup> shared_values: array<u32, GROUP>;
var<workgroup> shared_counts: array<vec4<u32>, GROUP>;

fn rows_matrix(rows: Rows) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(rows.r0, rows.r1, rows.r2, vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

fn matrix_rows(m: mat4x4<f32>) -> Rows {
    let t = transpose(m);
    return Rows(t[0], t[1], t[2]);
}

fn slot_rows(slot: u32) -> Rows {
    let pose = instance_poses[slot];
    return Rows(pose.row_0, pose.row_1, pose.row_2);
}

fn finite(value: vec4<f32>) -> bool {
    return all(abs(value) <= vec4<f32>(F32_MAX));
}

fn node_pose(at: u32) -> Pose {
    return Pose(
        vec3<f32>(locals[at], locals[at + 1u], locals[at + 2u]),
        vec4<f32>(locals[at + 4u], locals[at + 5u], locals[at + 6u], locals[at + 7u]),
        vec3<f32>(locals[at + 3u], locals[at + 8u], locals[at + 9u]),
    );
}

fn compose(node: u32) -> Composed {
    let record = info[node];
    let parent = record.x;
    let word = record.y;
    var base_display = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    var base_current = base_display;
    var parent_fallback = false;
    if pose_word_parent_composed(word) {
        let above = composed[parent];
        base_display = rows_matrix(above.display);
        base_current = rows_matrix(above.current);
        parent_fallback = above.status.x != 0u;
    } else if parent != NO_PARENT {
        base_display = rows_matrix(authority[parent]);
        base_current = base_display;
    }
    let now = node_pose(node * POSE_FLOATS);
    let current = base_current * pose_trs(now);
    var result: Composed;
    result.current = matrix_rows(current);
    if !parent_fallback {
        let was = node_pose((frame.node_capacity + node) * POSE_FLOATS);
        let display = base_display * drawn_local(was, now, frame.alpha, pose_word_blend(word));
        if pose_admitted(display) {
            result.display = matrix_rows(display);
            result.status = vec4<u32>(0u);
            return result;
        }
    }
    result.display = result.current;
    result.status = vec4<u32>(1u, 0u, 0u, 0u);
    return result;
}

fn outside(m: mat4x4<f32>, bounds_min: vec3<f32>, bounds_max: vec3<f32>) -> bool {
    if frame.culling == 0u {
        return false;
    }
    let maximum = max(abs(bounds_min), abs(bounds_max));
    let absolute = mat4x4<f32>(abs(m[0]), abs(m[1]), abs(m[2]), abs(m[3]));
    if maximum_abs(maximum) > MAX_COEFFICIENT
        || max(maximum_abs(absolute[0].xyz), max(maximum_abs(absolute[1].xyz),
            max(maximum_abs(absolute[2].xyz), maximum_abs(absolute[3].xyz)))) > MAX_COEFFICIENT {
        return false;
    }
    let world_bound = absolute * vec4<f32>(maximum, 1.0);
    let clip_bound = frame.absolute_view_projection * world_bound;
    if !finite(world_bound) || !finite(clip_bound)
        || max(maximum_abs(world_bound.xyz), world_bound.w) > MAX_PRODUCT_BOUND
        || max(maximum_abs(clip_bound.xyz), abs(clip_bound.w)) > MAX_PRODUCT_BOUND {
        return false;
    }
    var plane_bounds = array<f32, 6>(
        clip_bound.x + clip_bound.w,
        clip_bound.x + clip_bound.w,
        clip_bound.y + clip_bound.w,
        clip_bound.y + clip_bound.w,
        clip_bound.z,
        clip_bound.z + clip_bound.w,
    );
    let t = transpose(m);
    for (var plane = 0u; plane < 6u; plane++) {
        let local = t * frame.planes[plane];
        let support = select(bounds_min, bounds_max, local.xyz >= vec3<f32>(0.0));
        let distance = dot(local.xyz, support) + local.w;
        if !finite(local) || !(abs(distance) <= F32_MAX) {
            return false;
        }
        if distance < -ROUNDING_MARGIN * max(plane_bounds[plane], 1.0) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(PARENT_GROUP)
fn compose_parents(@builtin(global_invocation_id) id: vec3<u32>) {
    let position = level.start + id.x;
    if position >= level.end {
        return;
    }
    let node = parents[position];
    composed[node] = compose(node);
}

@compute @workgroup_size(CULL_GROUP)
fn cull(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    let previous = frame.occlusion != 0u && candidate_state_occluded(scan[candidate.position]);
    var state = candidate_state_with_occluded(0u, previous);
    if candidate.node != NO_NODE && pose_word_valid(info[candidate.node].y) {
        let word = info[candidate.node].y;
        state = candidate_state_with_valid(state, true);
        var model: mat4x4<f32>;
        if pose_word_composed(word) {
            model = rows_matrix(compose(candidate.node).display);
        } else {
            model = rows_matrix(authority[candidate.node]);
        }
        poses[candidate.slot] = matrix_rows(model);
        let batch = batches[candidate.batch];
        let first_person = batch.bounds_max.w == FIRST_PERSON_BATCH;
        var visible = first_person || !outside(model, batch.bounds_min.xyz, batch.bounds_max.xyz);
        if visible && frame.subpixel != 0u && !first_person
                && covers_no_pixel(frame.view_projection * model, batch.bounds_min.xyz,
                    batch.bounds_max.xyz, frame.viewport.xy) {
            visible = false;
            state = candidate_state_with_subpixel(state, true);
        }
        if visible {
            let occludable = frame.occlusion != 0u && batch.bounds_max.w == 0.0;
            state = candidate_state_with_visible(state, true);
            state = candidate_state_with_moving(state,
                frame.motion != 0u && pose_word_composed(word));
            state = candidate_state_with_phase_one(state, !occludable || !previous);
            state = candidate_state_with_level(state, lod_level(batch, model));
        }
    }
    scan[candidate.position] = state;
}

fn lod_level(batch: Batch, model: mat4x4<f32>) -> u32 {
    return lod_level_at(batch, model, 1.0);
}

fn lod_level_at(batch: Batch, model: mat4x4<f32>, scale_distance: f32) -> u32 {
    let levels = u32(batch.lod_errors.z);
    let impostor = frame.eye.w != 0.0 && batch.lod_errors.w > 0.0;
    if (levels <= 1u && !impostor) || !(frame.lod_error > 0.0) {
        return 0u;
    }
    let center = (model * vec4<f32>((batch.bounds_min.xyz + batch.bounds_max.xyz) * 0.5, 1.0)).xyz;
    let scale = max(length(model[0].xyz), max(length(model[1].xyz), length(model[2].xyz)));
    let radius = length(batch.bounds_max.xyz - batch.bounds_min.xyz) * 0.5 * scale;
    let projection = frame.view_projection;
    let y_row = vec3<f32>(projection[0].y, projection[1].y, projection[2].y);
    let w_row = vec3<f32>(projection[0].w, projection[1].w, projection[2].w);
    let pixels = length(y_row) * frame.viewport.y * 0.5 / frame.lod_error;
    let perspective = length(w_row) > 0.5;
    let to_eye = center - frame.eye.xyz;
    let distance_squared = dot(to_eye, to_eye) * scale_distance * scale_distance;
    var level = 0u;
    for (var l = 1u; l < IMPOSTOR + 1u; l++) {
        var error = select(batch.lod_errors.y, batch.lod_errors.x, l == 1u);
        if l == IMPOSTOR {
            error = batch.lod_errors.w;
        }
        if (l < IMPOSTOR && l >= levels) || (l == IMPOSTOR && !impostor) {
            continue;
        }
        let projected = error * scale * pixels;
        var fits = projected <= 1.0;
        if perspective {
            let reach = projected + radius;
            fits = reach * reach <= distance_squared;
        }
        if fits {
            level = l;
            if l < IMPOSTOR && level_indices(batch, l) == 0u {
                return l;
            }
        }
    }
    return level;
}

fn level_indices(batch: Batch, level: u32) -> u32 {
    return select(batch.lod_ranges.w, batch.lod_ranges.y, level == 1u);
}

@compute @workgroup_size(CULL_GROUP)
fn fade_levels(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates || !(frame.lod_error > 0.0) {
        return;
    }
    let candidate = slot_candidates[id.x];
    let flags = scan[candidate.position];
    let batch = batches[candidate.batch];
    if (flags & frame.mask) == 0u || frame.fade.x >= 1.0 {
        lod_states[candidate.slot] = 0u;
        return;
    }
    let level = candidate_state_level(flags);
    let state = lod_states[candidate.slot];
    var shown = level;
    var previous = level;
    var progress = LOD_FADED;
    if lod_fade_history(state) == LOD_HISTORY_HELD {
        shown = lod_fade_shown(state);
        previous = lod_fade_previous(state);
        progress = lod_fade_progress(state);
        let model = rows_matrix(slot_rows(candidate.slot));
        let nearer = lod_level_at(batch, model, 1.0 - frame.fade.y);
        let farther = lod_level_at(batch, model, 1.0 + frame.fade.y);
        let wanted = max(nearer, min(shown, farther));
        if progress >= LOD_FADED {
            if wanted != shown {
                previous = shown;
                shown = wanted;
                progress = 0u;
            }
        } else if wanted == previous {
            previous = shown;
            shown = wanted;
            progress = LOD_FADED - progress;
        }
        let cards = frame.eye.w != 0.0 && batch.lod_errors.w > 0.0;
        let card_switch = (shown == IMPOSTOR) != (previous == IMPOSTOR);
        if !card_switch || !cards || candidate_state_moving(flags)
                || candidate.batch > FADING_CARD_INDEX_MAX {
            progress = LOD_FADED;
        }
        progress = min(progress + u32(frame.fade.x * f32(LOD_FADED)), LOD_FADED);
    }
    lod_states[candidate.slot] = lod_fade(shown, previous, progress, LOD_HISTORY_HELD);
    var steady = shown;
    if progress < LOD_FADED {
        steady = select(shown, previous, shown == IMPOSTOR);
        let fade = u32(f32(progress) / f32(LOD_FADED) * f32(FADING_CARD_FADE_MAX));
        let word = fading_card(candidate.batch, true, fade, shown != IMPOSTOR, CARD_PHASE_FADING);
        let index = atomicAdd(&fade_cards[1], 1u);
        atomicStore(&fade_cards[4u + 2u * index], drawn_slot(candidate.slot, false));
        atomicStore(&fade_cards[5u + 2u * index], word);
    }
    scan[candidate.position] = candidate_state_with_level(flags, steady);
}

@compute @workgroup_size(CULL_GROUP)
fn occlusion_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    var flags = scan[candidate.position];
    if !candidate_state_visible(flags) {
        return;
    }
    let batch = batches[candidate.batch];
    if batch.bounds_max.w != 0.0 {
        return;
    }
    let is_hidden = hidden(frame.view_projection * rows_matrix(slot_rows(candidate.slot)),
        batch.bounds_min.xyz, batch.bounds_max.xyz, depth_pyramid, frame.viewport);
    flags = candidate_state_with_occluded(flags, is_hidden);
    if !candidate_state_phase_one(flags) {
        if is_hidden {
            atomicAdd(&occlusion_counts[0], 1u);
        } else {
            flags = candidate_state_with_phase_two(flags, true);
        }
    }
    scan[candidate.position] = flags;
}

fn view_takes(node: u32) -> bool {
    return frame.bases.z == 0u || node_word_mobility(node_words[node]) + 1u == frame.bases.z;
}

@compute @workgroup_size(CULL_GROUP)
fn view_cull(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    var flags = candidate_state_with_view(scan[candidate.position], false);
    let batch = batches[candidate.batch];
    if candidate_state_valid(flags) && batch.bounds_max.w != FIRST_PERSON_BATCH
            && view_takes(candidate.node) {
        let model = rows_matrix(slot_rows(candidate.slot));
        if !outside(model, batch.bounds_min.xyz, batch.bounds_max.xyz)
                && !(frame.subpixel != 0u && covers_no_pixel(frame.view_projection * model,
                    batch.bounds_min.xyz, batch.bounds_max.xyz, frame.viewport.xy)) {
            flags = candidate_state_with_view(flags, true);
            flags = candidate_state_with_view_level(flags, lod_level(batch, model));
        }
    }
    scan[candidate.position] = flags;
}

fn bucket(flags: u32) -> u32 {
    if !(frame.lod_error > 0.0) {
        return 0u;
    }
    return select(candidate_state_level(flags), candidate_state_view_level(flags),
        candidate_state_view(frame.mask));
}

fn gathered(flags: u32) -> vec4<u32> {
    if (flags & frame.mask) == 0u {
        return vec4<u32>(0u);
    }
    var hot = vec4<u32>(0u);
    hot[bucket(flags)] = 1u;
    return hot;
}

fn load_sums(index: u32) -> vec4<u32> {
    let at = frame.candidates + 4u * index;
    return vec4<u32>(scan[at], scan[at + 1u], scan[at + 2u], scan[at + 3u]);
}

fn store_sums(index: u32, value: vec4<u32>) {
    let at = frame.candidates + 4u * index;
    scan[at] = value.x;
    scan[at + 1u] = value.y;
    scan[at + 2u] = value.z;
    scan[at + 3u] = value.w;
}

@compute @workgroup_size(GROUP)
fn sum_groups(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    var counts = vec4<u32>(0u);
    var subpixel = 0u;
    if id.x < frame.candidates {
        let flags = scan[id.x];
        counts = gathered(flags);
        if candidate_state_subpixel(flags) && !candidate_state_phase_two(frame.mask)
                && !candidate_state_view(frame.mask) {
            subpixel = 1u;
        }
    }
    shared_counts[lane] = counts;
    shared_values[lane] = subpixel;
    workgroupBarrier();
    for (var stride = GROUP / 2u; stride > 0u; stride = stride >> 1u) {
        if lane < stride {
            shared_counts[lane] += shared_counts[lane + stride];
            shared_values[lane] += shared_values[lane + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        store_sums(group.x, shared_counts[0]);
        if shared_values[0] != 0u {
            atomicAdd(&occlusion_counts[1], shared_values[0]);
        }
    }
}

@compute @workgroup_size(GROUP)
fn scan_groups(@builtin(local_invocation_index) lane: u32) {
    var carry = vec4<u32>(0u);
    for (var chunk = 0u; chunk < frame.groups; chunk += 2u * GROUP) {
        let first = chunk + lane * 2u;
        var left = vec4<u32>(0u);
        var right = vec4<u32>(0u);
        if first < frame.groups {
            left = load_sums(first);
        }
        if first + 1u < frame.groups {
            right = load_sums(first + 1u);
        }
        let pair = left + right;
        shared_counts[lane] = pair;
        workgroupBarrier();
        for (var offset = 1u; offset < GROUP; offset = offset << 1u) {
            var addend = vec4<u32>(0u);
            if lane >= offset {
                addend = shared_counts[lane - offset];
            }
            workgroupBarrier();
            shared_counts[lane] += addend;
            workgroupBarrier();
        }
        let exclusive = carry + shared_counts[lane] - pair;
        if first < frame.groups {
            store_sums(first, exclusive);
        }
        if first + 1u < frame.groups {
            store_sums(first + 1u, exclusive + left);
        }
        carry += shared_counts[GROUP - 1u];
        workgroupBarrier();
    }
    if lane == 0u {
        store_sums(frame.groups, carry);
        scan[frame.candidates + 4u * frame.groups + 4u] = carry.x + carry.y + carry.z + carry.w;
    }
}

fn saturating_mul(a: u32, b: u32) -> u32 {
    let a_high = a >> 16u;
    let b_high = b >> 16u;
    let cross = a_high * (b & 0xffffu) + (a & 0xffffu) * b_high;
    let low = (a & 0xffffu) * (b & 0xffffu);
    let high = cross << 16u;
    let overflow = ((a_high != 0u) & (b_high != 0u)) | (cross > 0xffffu) | (low > 0xffffffffu - high);
    return select(low + high, 0xffffffffu, overflow);
}

fn saturating_add(a: u32, b: u32) -> u32 {
    return select(a + b, 0xffffffffu, a > 0xffffffffu - b);
}

fn visible_before(position: u32) -> vec4<u32> {
    if position >= frame.candidates {
        return load_sums(frame.groups);
    }
    let group = position / GROUP;
    var value = load_sums(group);
    for (var index = group * GROUP; index < position; index++) {
        value += gathered(scan[index]);
    }
    return value;
}

@compute @workgroup_size(ARGS_GROUP)
fn batch_args(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.batches {
        return;
    }
    let batch = batches[id.x];
    let base = visible_before(batch.start);
    let visible = visible_before(batch.start + batch.count) - base;
    var instances = 0u;
    if base.x < frame.budget {
        instances = min(visible.x, frame.budget - base.x);
    }
    for (var level = 0u; level < 4u; level++) {
        batch_bases[4u * id.x + level] = base[level];
    }
    let at = frame.bases.y;
    let first = batch.first_instance + select(0u, frame.bases.x, batch.first_plane != 0u);
    draws[at + id.x] = DrawArgs(batch.index_count, instances, batch.first_index, batch.base_vertex,
        first);
    if !candidate_state_view(frame.mask) {
        let triangles = saturating_add(saturating_add(
            saturating_mul(batch.index_count / 3u, instances),
            saturating_mul(batch.lod_ranges.y / 3u, visible.y)),
            saturating_mul(batch.lod_ranges.w / 3u, visible.z));
        if triangles != 0u {
            atomicAdd(&occlusion_counts[2], triangles);
        }
    }
    if id.x == 0u {
        let impostors = load_sums(frame.groups)[IMPOSTOR];
        draws[at + LOD_LEVELS * frame.batches] = DrawArgs(IMPOSTOR_VERTICES, impostors, 0u, 0,
            0u);
        if impostors != 0u {
            atomicAdd(&occlusion_counts[2], saturating_mul(IMPOSTOR_VERTICES / 3u, impostors));
        }
    }
    let levels = u32(batch.lod_errors.z);
    if levels > 1u {
        draws[at + frame.batches + id.x] = DrawArgs(batch.lod_ranges.y, visible.y,
            batch.lod_ranges.x, batch.base_vertex, first + batch.first_plane);
    }
    if levels > 2u {
        draws[at + 2u * frame.batches + id.x] = DrawArgs(batch.lod_ranges.w, visible.z,
            batch.lod_ranges.z, batch.base_vertex, first + 2u * batch.first_plane);
    }
}

@compute @workgroup_size(GROUP)
fn scatter(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    var stored = 0u;
    if id.x < frame.candidates {
        stored = scan[id.x];
    }
    let hot = gathered(stored);
    shared_counts[lane] = hot;
    workgroupBarrier();
    for (var offset = 1u; offset < GROUP; offset = offset << 1u) {
        var addend = vec4<u32>(0u);
        if lane >= offset {
            addend = shared_counts[lane - offset];
        }
        workgroupBarrier();
        shared_counts[lane] += addend;
        workgroupBarrier();
    }
    if any(hot != vec4<u32>(0u)) {
        let level = bucket(stored);
        let global = (load_sums(group.x) + shared_counts[lane] - hot)[level];
        if level == IMPOSTOR {
            let candidate = candidates[id.x];
            impostor_ids[global] = vec2<u32>(
                drawn_slot(candidate.slot, candidate_state_moving(stored)),
                steady_card(candidate.batch, CARD_PHASE_STEADY));
        } else if level != 0u || global < frame.budget {
            let candidate = candidates[id.x];
            let batch = candidate.batch;
            visible_ids[frame.bases.x + level * frame.plane + batches[batch].start
                + (global - batch_bases[4u * batch + level])] =
                drawn_slot(candidate.slot, candidate_state_moving(stored));
        }
    }
}

fn totals_at() -> u32 {
    return frame.candidates + 4u * frame.groups;
}

@compute @workgroup_size(1)
fn gather_phase_one() {
    for (var word = 0u; word < 14u; word++) {
        totals[word] = 0u;
    }
    let at = totals_at();
    totals[0] = scan[at + 4u];
    for (var level = 0u; level < 4u; level++) {
        totals[4u + level] = scan[at + level];
    }
    totals[3] = atomicLoad(&occlusion_counts[1]);
    if frame.occlusion == 0u {
        totals[12] = atomicLoad(&occlusion_counts[2]);
        gather_watched();
    }
}

@compute @workgroup_size(1)
fn gather_phase_two() {
    let at = totals_at();
    totals[1] = scan[at + 4u];
    for (var level = 0u; level < 4u; level++) {
        totals[8u + level] = scan[at + level];
    }
    totals[2] = atomicLoad(&occlusion_counts[0]);
    totals[12] = atomicLoad(&occlusion_counts[2]);
    gather_watched();
}

fn gather_watched() {
    if frame.watched != UNWATCHED {
        totals[13] = lod_states[frame.watched];
    }
}
