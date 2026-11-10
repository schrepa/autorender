import package::engine::drawn::{drawn_in_view, drawn_mark_first, drawn_mark_moving,
    drawn_mark_subpixel, drawn_marks, drawn_model, drawn_occlusion, drawn_poses, drawn_views,
    occlusion_mark_hidden, occlusion_mark_late};
import package::engine::drawn_fades::set_drawn_fade;
import package::engine::geometry::declarations::{FADE_LEVEL_MAX, drawn_slot, fade_word};
import package::generated::meshes::{ARGS_GROUP, BatchData as Batch, CULL_GROUP,
    CandidateData as Candidate, GROUP, LOD_FADED, LOD_HISTORY_HELD,
    LOD_LEVELS, PresentationFrame as Frame, UNWATCHED,
    candidate_state_level, candidate_state_moving,
    candidate_state_phase_one, candidate_state_phase_two, candidate_state_subpixel,
    candidate_state_view, candidate_state_view_level,
    candidate_state_visible, candidate_state_with_level, candidate_state_with_moving,
    candidate_state_with_phase_one,
    candidate_state_with_phase_two, candidate_state_with_subpixel,
    candidate_state_with_view, candidate_state_with_view_level, candidate_state_with_visible,
    lod_fade, lod_fade_history, lod_fade_previous, lod_fade_progress, lod_fade_shown};

struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(1) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(6) var<storage, read> candidates: array<Candidate>;
@group(1) @binding(7) var<storage, read> batches: array<Batch>;
@group(1) @binding(9) var<storage, read_write> scan: array<u32>;
@group(1) @binding(10) var<storage, read_write> batch_bases: array<u32>;
@group(1) @binding(11) var<storage, read_write> visible_ids: array<u32>;
@group(1) @binding(12) var<storage, read_write> draws: array<DrawArgs>;
@group(1) @binding(14) var<storage, read> slot_candidates: array<Candidate>;
@group(1) @binding(19) var<storage, read_write> occlusion_counts: array<atomic<u32>>;
@group(1) @binding(24) var<storage, read_write> last_ids: array<u32>;
@group(1) @binding(25) var<storage, read_write> lod_states: array<u32>;
@group(1) @binding(26) var<storage, read_write> fading_ids: array<atomic<u32>>;
@group(1) @binding(27) var<storage, read_write> totals: array<u32, 26>;

const LAST: u32 = LOD_LEVELS;

var<workgroup> shared_values: array<u32, GROUP>;
var<workgroup> shared_counts: array<vec4<u32>, GROUP>;

@compute @workgroup_size(CULL_GROUP)
fn cull(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    let mark = drawn_marks[candidate.slot];
    var state = candidate_state_with_subpixel(0u, drawn_mark_subpixel(mark));
    if drawn_in_view(0u, candidate.slot) {
        state = candidate_state_with_visible(state, true);
        state = candidate_state_with_moving(state, drawn_mark_moving(mark));
        state = candidate_state_with_phase_one(state,
            frame.occlusion == 0u || drawn_mark_first(mark));
        state = candidate_state_with_level(state,
            lod_level(batches[candidate.batch], drawn_model(candidate.slot), 0u));
    }
    scan[candidate.position] = state;
}

fn lod_level(batch: Batch, model: mat4x4<f32>, view: u32) -> u32 {
    return lod_level_at(batch, model, 1.0, view);
}

fn lod_level_at(batch: Batch, model: mat4x4<f32>, scale_distance: f32, view: u32) -> u32 {
    let levels = u32(batch.lod_errors.z);
    let last = frame.last_level != 0u && batch.lod_errors.w > 0.0;
    if (levels <= 1u && !last) || !(frame.lod_error > 0.0) {
        return 0u;
    }
    let center = (model * vec4<f32>((batch.bounds_min.xyz + batch.bounds_max.xyz) * 0.5, 1.0)).xyz;
    let scale = max(length(model[0].xyz), max(length(model[1].xyz), length(model[2].xyz)));
    let radius = length(batch.bounds_max.xyz - batch.bounds_min.xyz) * 0.5 * scale;
    let projection = drawn_views[view].rasterized;
    let y_row = vec3<f32>(projection[0].y, projection[1].y, projection[2].y);
    let w_row = vec3<f32>(projection[0].w, projection[1].w, projection[2].w);
    let pixels = length(y_row) * drawn_views[view].viewport.y * 0.5 / frame.lod_error;
    let perspective = length(w_row) > 0.5;
    let to_eye = center - frame.eye.xyz;
    let distance_squared = dot(to_eye, to_eye) * scale_distance * scale_distance;
    var level = 0u;
    for (var l = 1u; l < LAST + 1u; l++) {
        var error = select(batch.lod_errors.y, batch.lod_errors.x, l == 1u);
        if l == LAST {
            error = batch.lod_errors.w;
        }
        if (l < LAST && l >= levels) || (l == LAST && !last) {
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
            if l < LAST && level_indices(batch, l) == 0u {
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
        let model = drawn_model(candidate.slot);
        let nearer = lod_level_at(batch, model, 1.0 - frame.fade.y, 0u);
        let farther = lod_level_at(batch, model, 1.0 + frame.fade.y, 0u);
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
        let last = frame.last_level != 0u && batch.lod_errors.w > 0.0;
        let last_switch = (shown == LAST) != (previous == LAST);
        if !last_switch || !last || candidate_state_moving(flags) {
            progress = LOD_FADED;
        }
        progress = min(progress + u32(frame.fade.x * f32(LOD_FADED)), LOD_FADED);
    }
    lod_states[candidate.slot] = lod_fade(shown, previous, progress, LOD_HISTORY_HELD);
    var steady = shown;
    if progress < LOD_FADED {
        steady = select(shown, previous, shown == LAST);
        let faded = u32(f32(progress) / f32(LOD_FADED) * f32(FADE_LEVEL_MAX));
        set_drawn_fade(candidate.slot, fade_word(faded, shown != LAST, true));
        let index = atomicAdd(&fading_ids[0], 1u);
        atomicStore(&fading_ids[4u + 2u * index], drawn_slot(candidate.slot, false));
        atomicStore(&fading_ids[5u + 2u * index], candidate.batch);
    }
    scan[candidate.position] = candidate_state_with_level(flags, steady);
}

@compute @workgroup_size(CULL_GROUP)
fn occluded_lists(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    var flags = scan[candidate.position];
    if !candidate_state_visible(flags) || candidate_state_phase_one(flags) {
        return;
    }
    let held = drawn_occlusion[candidate.slot];
    if occlusion_mark_late(held) {
        flags = candidate_state_with_phase_two(flags, true);
    } else if occlusion_mark_hidden(held) {
        atomicAdd(&occlusion_counts[0], 1u);
    }
    scan[candidate.position] = flags;
}

@compute @workgroup_size(CULL_GROUP)
fn view_cull(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    var flags = candidate_state_with_view(scan[candidate.position], false);
    if drawn_in_view(frame.bases.z, candidate.slot) {
        flags = candidate_state_with_view(flags, true);
        flags = candidate_state_with_view_level(flags,
            lod_level(batches[candidate.batch], drawn_model(candidate.slot), frame.bases.z));
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
        let triangles = saturating_add(saturating_add(saturating_add(
            saturating_mul(batch.index_count / 3u, instances),
            saturating_mul(batch.lod_ranges.y / 3u, visible.y)),
            saturating_mul(batch.lod_ranges.w / 3u, visible.z)),
            saturating_mul(batch.last_triangles, visible.w));
        if triangles != 0u {
            atomicAdd(&occlusion_counts[2], triangles);
        }
        if id.x == 0u {
            last_ids[0] = load_sums(frame.groups)[LAST];
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
        if level == LAST {
            let candidate = candidates[id.x];
            let entry = 4u + 2u * global;
            last_ids[entry] = drawn_slot(candidate.slot, candidate_state_moving(stored));
            last_ids[entry + 1u] = candidate.batch;
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
    for (var word = 0u; word < 26u; word++) {
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
        let pose = drawn_poses[frame.watched];
        for (var lane = 0u; lane < 4u; lane++) {
            totals[14u + lane] = bitcast<u32>(pose.row_0[lane]);
            totals[18u + lane] = bitcast<u32>(pose.row_1[lane]);
            totals[22u + lane] = bitcast<u32>(pose.row_2[lane]);
        }
    }
}
