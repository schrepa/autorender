import package::engine::math::maximum_abs;
import package::engine::mesh::impostor_frames::{IMPOSTOR_FADING as CARD_FADING,
    IMPOSTOR_FADING_OUT as CARD_FADING_OUT, IMPOSTOR_OVER as CARD_OVER};
import package::generated::meshes::{ARGS_GROUP, BatchData as Batch, CULL_GROUP,
    CandidateData as Candidate, GROUP, IMPOSTOR_VERTICES, LOD_LEVELS, PARENT_GROUP, PHASE1, PHASE2,
    PROBE, PresentationFrame as Frame, VISIBLE};
import package::meshes::screen_cull::{F32_MAX, covers_no_pixel, hidden};
import package::place::instances::{MOVED_SLOT, STILL_FRAMES};

const POSE_FLOATS: u32 = 10u;

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
@group(1) @binding(13) var<storage, read_write> locals_shift: array<f32>;
@group(1) @binding(14) var<storage, read> slot_candidates: array<Candidate>;
@group(1) @binding(18) var depth_pyramid: texture_2d<f32>;
@group(1) @binding(19) var<storage, read_write> occlusion_counts: array<atomic<u32>>;
@group(1) @binding(23) var<storage, read_write> still_frames: array<vec2<u32>>;
@group(1) @binding(24) var<storage, read_write> impostor_ids: array<vec2<u32>>;
@group(1) @binding(25) var<storage, read_write> lod_states: array<u32>;
@group(1) @binding(26) var<storage, read_write> fade_cards: array<atomic<u32>>;
@group(1) @binding(27) var<storage, read_write> totals: array<u32, 14>;
@group(1) @binding(28) var<storage, read> node_staging: array<f32>;
@group(1) @binding(29) var<storage, read> node_ranges: array<vec4<u32>>;
@group(2) @binding(0) var<uniform> level: Level;

const NO_PARENT: u32 = 0xffffffffu;
const NODE_VALID: u32 = 1u;
const NODE_BLEND: u32 = 2u;
const NODE_ACTIVE: u32 = 4u;
const NODE_PARENT_ACTIVE: u32 = 8u;
const MOVING: u32 = 2u;
const VALID: u32 = 4u;
const OCCLUDED: u32 = 32u;
const SUBPIXEL: u32 = 64u;
const LOD_SHIFT: u32 = 8u;
const IMPOSTOR: u32 = LOD_LEVELS;
const LOD_STATE: u32 = 0x80000000u;
const LOD_FADED: u32 = 65535u;
const LINEAR_ROTATION_DOT: f32 = 0.9995;
const ROUNDING_MARGIN: f32 = 0.000003814697265625f;
const MAX_COEFFICIENT: f32 = 1.0e15;
const MAX_PRODUCT_BOUND: f32 = 4.2535293e37f;
const MIN_LINEAR_SCALE: f32 = 0.0009765625f;
const COEFFICIENT_LIMIT: f32 = 1048576.0f;
const COMMON_DETERMINANT_MARGIN: f32 = 0.001953125f;

var<workgroup> shared_values: array<u32, GROUP>;
var<workgroup> shared_counts: array<vec4<u32>, GROUP>;

fn rows_matrix(rows: Rows) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(rows.r0, rows.r1, rows.r2, vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

fn matrix_rows(m: mat4x4<f32>) -> Rows {
    let t = transpose(m);
    return Rows(t[0], t[1], t[2]);
}

fn trs(translation: vec3<f32>, rotation: vec4<f32>, scale: vec3<f32>) -> mat4x4<f32> {
    let x = rotation.x;
    let y = rotation.y;
    let z = rotation.z;
    let w = rotation.w;
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
    return mat4x4<f32>(
        vec4<f32>(1.0 - (yy + zz), xy + wz, xz - wy, 0.0) * scale.x,
        vec4<f32>(xy - wz, 1.0 - (xx + zz), yz + wx, 0.0) * scale.y,
        vec4<f32>(xz + wy, yz - wx, 1.0 - (xx + yy), 0.0) * scale.z,
        vec4<f32>(translation, 1.0),
    );
}

fn blend_rotation(previous: vec4<f32>, current: vec4<f32>, alpha: f32) -> vec4<f32> {
    let p = normalize(previous);
    var c = normalize(current);
    let d = dot(p, c);
    if abs(d) >= LINEAR_ROTATION_DOT {
        if d < 0.0 {
            c = -c;
        }
        return normalize(p * (1.0 - alpha) + c * alpha);
    }
    var cosine = d;
    if cosine < 0.0 {
        c = -c;
        cosine = -cosine;
    }
    let theta = acos(cosine);
    let start = sin(theta * (1.0 - alpha));
    let end = sin(theta * alpha);
    return normalize((p * start + c * end) * (1.0 / sin(theta)));
}

fn finite(value: vec4<f32>) -> bool {
    return all(abs(value) <= vec4<f32>(F32_MAX));
}

fn admitted(m: mat4x4<f32>) -> bool {
    if !(finite(m[0]) && finite(m[1]) && finite(m[2]) && finite(m[3])) {
        return false;
    }
    let x = m[0].xyz;
    let y = m[1].xyz;
    let z = m[2].xyz;
    let scale = max(maximum_abs(x), max(maximum_abs(y), maximum_abs(z)));
    let determinant = dot(x, cross(y, z));
    return scale >= MIN_LINEAR_SCALE && scale <= COEFFICIENT_LIMIT
        && maximum_abs(m[3].xyz) <= COEFFICIENT_LIMIT
        && abs(determinant) >= COMMON_DETERMINANT_MARGIN * ((scale * scale) * scale);
}

fn compose(node: u32) -> Composed {
    let record = info[node];
    let parent = record.x;
    let flags = record.y;
    var base_display = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    var base_current = base_display;
    var parent_fallback = false;
    if (flags & NODE_PARENT_ACTIVE) != 0u {
        let above = composed[parent];
        base_display = rows_matrix(above.display);
        base_current = rows_matrix(above.current);
        parent_fallback = above.status.x != 0u;
    } else if parent != NO_PARENT {
        base_display = rows_matrix(authority[parent]);
        base_current = base_display;
    }
    let now = node * POSE_FLOATS;
    let now_translation = vec3<f32>(locals[now], locals[now + 1u], locals[now + 2u]);
    let now_rotation = vec4<f32>(locals[now + 4u], locals[now + 5u], locals[now + 6u], locals[now + 7u]);
    let now_scale = vec3<f32>(locals[now + 3u], locals[now + 8u], locals[now + 9u]);
    let current = base_current * trs(now_translation, now_rotation, now_scale);
    var result: Composed;
    result.current = matrix_rows(current);
    if !parent_fallback {
        var local = trs(now_translation, now_rotation, now_scale);
        let alpha = frame.alpha;
        if (flags & NODE_BLEND) != 0u && frame.blend != 0u && alpha < 1.0 {
            let was = (frame.node_capacity + node) * POSE_FLOATS;
            let previous_translation = vec3<f32>(locals[was], locals[was + 1u], locals[was + 2u]);
            let previous_rotation = vec4<f32>(locals[was + 4u], locals[was + 5u], locals[was + 6u], locals[was + 7u]);
            let previous_scale = vec3<f32>(locals[was + 3u], locals[was + 8u], locals[was + 9u]);
            if alpha == 0.0 {
                local = trs(previous_translation, previous_rotation, previous_scale);
            } else {
                let inverse = 1.0 - alpha;
                local = trs(
                    previous_translation * inverse + now_translation * alpha,
                    blend_rotation(previous_rotation, now_rotation, alpha),
                    previous_scale * inverse + now_scale * alpha,
                );
            }
        }
        let display = base_display * local;
        if admitted(display) {
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

@compute @workgroup_size(CULL_GROUP)
fn shift_previous(@builtin(global_invocation_id) id: vec3<u32>) {
    let node = id.x;
    if node >= arrayLength(&info) || (info[node].y & NODE_ACTIVE) == 0u {
        return;
    }
    let half = arrayLength(&locals_shift) / 2u;
    let start = node * POSE_FLOATS;
    for (var index = start; index < start + POSE_FLOATS; index++) {
        locals_shift[half + index] = locals_shift[index];
    }
}

@compute @workgroup_size(PARENT_GROUP)
fn apply_locals(@builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    let range = node_ranges[group.x];
    for (var index = lane; index < range.y; index += PARENT_GROUP) {
        locals_shift[range.x + index] = node_staging[range.z + index];
    }
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
    let previous = scan[candidate.position] & OCCLUDED;
    var visible = previous;
    let flags = info[candidate.node].y;
    if (flags & NODE_VALID) != 0u {
        visible = previous | VALID;
        var model: mat4x4<f32>;
        if (flags & NODE_ACTIVE) != 0u {
            model = rows_matrix(compose(candidate.node).display);
        } else {
            model = rows_matrix(authority[candidate.node]);
        }
        poses[candidate.slot] = matrix_rows(model);
        let batch = batches[candidate.batch];
        if !outside(model, batch.bounds_min.xyz, batch.bounds_max.xyz) {
            visible |= VISIBLE;
        }
        if (visible & VISIBLE) != 0u && frame.subpixel != 0u
                && covers_no_pixel(frame.view_projection * model, batch.bounds_min.xyz,
                    batch.bounds_max.xyz, frame.viewport.xy) {
            visible = (visible & ~VISIBLE) | SUBPIXEL;
        }
        if (visible & VISIBLE) != 0u {
            if frame.motion != 0u && (flags & NODE_ACTIVE) != 0u {
                visible |= MOVING;
            }
            let occludable = frame.occlusion != 0u && batch.bounds_max.w == 0.0;
            if !occludable || previous == 0u {
                visible |= PHASE1;
            }
            visible |= lod_level(batch, model) << LOD_SHIFT;
        }
    }
    scan[candidate.position] = visible;
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
    let level = (flags >> LOD_SHIFT) & 3u;
    let tag = (candidate.batch & 0xffu) << 23u;
    let state = lod_states[candidate.slot];
    var shown = level;
    var previous = level;
    var progress = LOD_FADED;
    if (state & LOD_STATE) != 0u && (state & (0xffu << 23u)) == tag {
        shown = state & 3u;
        previous = (state >> 2u) & 3u;
        progress = (state >> 4u) & 0xffffu;
        let model = rows_matrix(poses[candidate.slot]);
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
        if !card_switch || !cards || (flags & MOVING) != 0u || candidate.batch >= CARD_OVER {
            progress = LOD_FADED;
        }
        progress = min(progress + u32(frame.fade.x * f32(LOD_FADED)), LOD_FADED);
    }
    lod_states[candidate.slot] = LOD_STATE | tag | shown | (previous << 2u) | (progress << 4u);
    var steady = shown;
    if progress < LOD_FADED {
        steady = select(shown, previous, shown == IMPOSTOR);
        let fade = u32(f32(progress) / f32(LOD_FADED) * 16383.0);
        let word = candidate.batch | CARD_OVER | CARD_FADING | (fade << 16u)
            | select(CARD_FADING_OUT, 0u, shown == IMPOSTOR);
        let index = atomicAdd(&fade_cards[1], 1u);
        atomicStore(&fade_cards[4u + 2u * index], candidate.slot);
        atomicStore(&fade_cards[5u + 2u * index], word);
    }
    scan[candidate.position] = (flags & ~(3u << LOD_SHIFT)) | (steady << LOD_SHIFT);
}

@compute @workgroup_size(CULL_GROUP)
fn occlusion_test(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    var flags = scan[candidate.position];
    if (flags & VISIBLE) == 0u {
        return;
    }
    let batch = batches[candidate.batch];
    if batch.bounds_max.w != 0.0 {
        return;
    }
    let is_hidden = hidden(frame.view_projection * rows_matrix(poses[candidate.slot]),
        batch.bounds_min.xyz, batch.bounds_max.xyz, depth_pyramid, frame.viewport);
    flags = (flags & ~OCCLUDED) | select(0u, OCCLUDED, is_hidden);
    if (flags & PHASE1) == 0u {
        if is_hidden {
            atomicAdd(&occlusion_counts[0], 1u);
        } else {
            flags |= PHASE2;
        }
    }
    scan[candidate.position] = flags;
}

fn mix_key(key: u32, value: u32) -> u32 {
    let x = (key ^ value) * 0x9e3779b1u;
    return x ^ (x >> 15u);
}

fn mix_row(key: u32, row: vec4<f32>) -> u32 {
    let bits = bitcast<vec4<u32>>(row);
    return mix_key(mix_key(mix_key(mix_key(key, bits.x), bits.y), bits.z), bits.w);
}

fn pose_key(pose: Rows) -> u32 {
    return mix_row(mix_row(mix_row(0x2545f491u, pose.r0), pose.r1), pose.r2);
}

@compute @workgroup_size(CULL_GROUP)
fn count_still(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    let state = still_frames[candidate.slot];
    var key = 0u;
    var count = 0u;
    if (scan[candidate.position] & VALID) != 0u {
        key = pose_key(poses[candidate.slot]);
        count = select(1u, min(state.y + 1u, STILL_FRAMES), key == state.x && state.y != 0u);
    }
    if key != state.x || count != state.y {
        still_frames[candidate.slot] = vec2<u32>(key, count);
    }
}

@compute @workgroup_size(CULL_GROUP)
fn probe_cull(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    var flags = scan[candidate.position] & ~PROBE;
    let batch = batches[candidate.batch];
    let still = still_frames[candidate.slot].y >= STILL_FRAMES;
    if (flags & VALID) != 0u && still && batch.bounds_max.w != 1.0 {
        let model = rows_matrix(poses[candidate.slot]);
        if !outside(model, batch.bounds_min.xyz, batch.bounds_max.xyz)
                && !(frame.subpixel != 0u && covers_no_pixel(frame.view_projection * model,
                    batch.bounds_min.xyz, batch.bounds_max.xyz, frame.viewport.xy)) {
            flags |= PROBE;
        }
    }
    scan[candidate.position] = flags;
}

fn bucket(flags: u32) -> u32 {
    return select(0u, (flags >> LOD_SHIFT) & 3u, frame.lod_error > 0.0);
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
        if (flags & SUBPIXEL) != 0u && (frame.mask & (PHASE2 | PROBE)) == 0u {
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
    draws[id.x] = DrawArgs(batch.index_count, instances, batch.first_index, batch.base_vertex, batch.first_instance);
    if (frame.mask & PROBE) == 0u {
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
        draws[LOD_LEVELS * frame.batches] = DrawArgs(IMPOSTOR_VERTICES, impostors, 0u, 0, 0u);
        if impostors != 0u {
            atomicAdd(&occlusion_counts[2], saturating_mul(IMPOSTOR_VERTICES / 3u, impostors));
        }
    }
    let levels = u32(batch.lod_errors.z);
    if levels > 1u {
        draws[frame.batches + id.x] = DrawArgs(batch.lod_ranges.y, visible.y,
            batch.lod_ranges.x, batch.base_vertex, batch.first_instance + batch.first_plane);
    }
    if levels > 2u {
        draws[2u * frame.batches + id.x] = DrawArgs(batch.lod_ranges.w, visible.z,
            batch.lod_ranges.z, batch.base_vertex, batch.first_instance + 2u * batch.first_plane);
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
                candidate.slot | select(0u, MOVED_SLOT, (stored & MOVING) != 0u), candidate.batch);
        } else if level != 0u || global < frame.budget {
            let candidate = candidates[id.x];
            let batch = candidate.batch;
            visible_ids[level * frame.plane + batches[batch].start
                + (global - batch_bases[4u * batch + level])] =
                candidate.slot | select(0u, MOVED_SLOT, (stored & MOVING) != 0u);
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
}
