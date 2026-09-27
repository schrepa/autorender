// GPU presentation (decision 014): blend, compose, validate and cull displayed
// poses for frames whose interpolation runs on the GPU, then compact visible
// slots per draw group for indirect draws. Mirrors the CPU presentation contract
// in f32: normalized linear rotation blending up to the 0.9995 dot threshold,
// spherical beyond it, parent-first composition, the common admission bound
// with fallback to the current pose for the node and its subtree, and the
// conservative transformed-plane culling test. Results are for display only.

struct Frame {
    alpha: f32,
    candidates: u32,
    budget: u32,
    culling: u32,
    groups: u32,
    batches: u32,
    // Node slots per half of `locals`: previous poses start at this slot.
    node_capacity: u32,
    // Zero until previous poses are consistent (a missed copy displays current).
    blend: u32,
    planes: array<vec4<f32>, 6>,
    absolute_view_projection: mat4x4<f32>,
    // Nonzero on temporally antialiased frames (decision 017): moving slots'
    // visible IDs carry MOVED_SLOT, so only they read previous poses.
    motion: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// One local pose per node in 10 floats: [t.xyz, s.x, r.xyzw, s.yz]. Current
// poses fill slots [0, node_capacity), previous poses the second half.
const POSE_FLOATS: u32 = 10u;

// Transposed top three rows of an affine matrix; the fourth row is [0, 0, 0, 1].
struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

// A composed active parent: displayed and current-pose matrices, and whether
// its display fell back to the current pose (inherited by its subtree).
struct Composed {
    display: Rows,
    current: Rows,
    status: vec4<u32>,
}

struct Candidate {
    slot: u32,
    node: u32,
    batch: u32,
    // Index in draw-group order, where its visibility flag goes.
    position: u32,
}

struct Batch {
    start: u32,
    count: u32,
    index_count: u32,
    pad: u32,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}

struct DrawArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// One 16-byte uniform per depth level; each level has its own bind group.
struct Level {
    start: u32,
    end: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> locals: array<f32>;
@group(0) @binding(2) var<storage, read> info: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> authority: array<Rows>;
@group(0) @binding(4) var<storage, read_write> composed: array<Composed>;
@group(0) @binding(5) var<storage, read> parents: array<u32>;
@group(0) @binding(6) var<storage, read> candidates: array<Candidate>;
@group(0) @binding(7) var<storage, read> batches: array<Batch>;
@group(0) @binding(8) var<storage, read_write> poses: array<Rows>;
// [0, candidates): candidate flags (VISIBLE; MOVING marks a moving visible
// slot on temporal frames; VALID a pose this pass wrote, which shadow caster
// selection reads); then one workgroup sum per group, replaced by its
// exclusive prefix; then the total visible count.
@group(0) @binding(9) var<storage, read_write> scan: array<u32>;
@group(0) @binding(10) var<storage, read_write> batch_bases: array<u32>;
@group(0) @binding(11) var<storage, read_write> visible_ids: array<u32>;
@group(0) @binding(12) var<storage, read_write> draws: array<DrawArgs>;
// The same `locals` buffer, written only by `shift_previous`.
@group(0) @binding(13) var<storage, read_write> locals_shift: array<f32>;
// The candidates again, in slot order (close to node order): culling streams
// pose, node and authority records instead of striding across draw groups.
@group(0) @binding(14) var<storage, read> slot_candidates: array<Candidate>;
@group(1) @binding(0) var<uniform> level: Level;

const NO_PARENT: u32 = 0xffffffffu;
const NODE_VALID: u32 = 1u;
const NODE_BLEND: u32 = 2u;
const NODE_ACTIVE: u32 = 4u;
const NODE_PARENT_ACTIVE: u32 = 8u;
// Visible IDs of slots whose displayed pose can change between frames.
const MOVED_SLOT: u32 = 0x80000000u;
const VISIBLE: u32 = 1u;
const MOVING: u32 = 2u;
const VALID: u32 = 4u;
const LINEAR_ROTATION_DOT: f32 = 0.9995;
const GROUP: u32 = 256u;
// Exact hex literals: browsers reject decimal spellings above f32::MAX.
// 32 * f32::EPSILON, as the CPU support test; covers model/view rounding.
const ROUNDING_MARGIN: f32 = 0x1p-18f;
const MAX_COEFFICIENT: f32 = 1.0e15;
// f32::MAX / 8, as the CPU culling bound.
const MAX_PRODUCT_BOUND: f32 = 0x1.fffffep+124f;
const F32_MAX: f32 = 0x1.fffffep+127f;
// The CPU validator's proven common admission path (decision 012).
const MIN_LINEAR_SCALE: f32 = 0x1p-10f;
const COEFFICIENT_LIMIT: f32 = 0x1p+20f;
const COMMON_DETERMINANT_MARGIN: f32 = 0x1p-9f;

var<workgroup> shared_values: array<u32, 256>;

fn rows_matrix(rows: Rows) -> mat4x4<f32> {
    return transpose(mat4x4<f32>(rows.r0, rows.r1, rows.r2, vec4<f32>(0.0, 0.0, 0.0, 1.0)));
}

fn matrix_rows(m: mat4x4<f32>) -> Rows {
    let t = transpose(m);
    return Rows(t[0], t[1], t[2]);
}

// glam's Mat4::from_scale_rotation_translation.
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
    // Below the linear threshold, far from the parallel singularity.
    let theta = acos(cosine);
    let start = sin(theta * (1.0 - alpha));
    let end = sin(theta * alpha);
    return normalize((p * start + c * end) * (1.0 / sin(theta)));
}

fn max_abs(value: vec3<f32>) -> f32 {
    let positive = abs(value);
    return max(positive.x, max(positive.y, positive.z));
}

fn finite(value: vec4<f32>) -> bool {
    // NaN fails every comparison.
    return all(abs(value) <= vec4<f32>(F32_MAX));
}

// The common sufficient admission bound of the CPU validator. Matrices outside
// it display their current pose instead, a display-only difference.
fn admitted(m: mat4x4<f32>) -> bool {
    if !(finite(m[0]) && finite(m[1]) && finite(m[2]) && finite(m[3])) {
        return false;
    }
    let x = m[0].xyz;
    let y = m[1].xyz;
    let z = m[2].xyz;
    let scale = max(max_abs(x), max(max_abs(y), max_abs(z)));
    let determinant = dot(x, cross(y, z));
    return scale >= MIN_LINEAR_SCALE && scale <= COEFFICIENT_LIMIT
        && max_abs(m[3].xyz) <= COEFFICIENT_LIMIT
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

// True only when the box is proven outside one frustum plane.
fn outside(m: mat4x4<f32>, bounds_min: vec3<f32>, bounds_max: vec3<f32>) -> bool {
    if frame.culling == 0u {
        return false;
    }
    let maximum = max(abs(bounds_min), abs(bounds_max));
    let absolute = mat4x4<f32>(abs(m[0]), abs(m[1]), abs(m[2]), abs(m[3]));
    if max_abs(maximum) > MAX_COEFFICIENT
        || max(max_abs(absolute[0].xyz), max(max_abs(absolute[1].xyz),
            max(max_abs(absolute[2].xyz), max_abs(absolute[3].xyz)))) > MAX_COEFFICIENT {
        return false;
    }
    let world_bound = absolute * vec4<f32>(maximum, 1.0);
    let clip_bound = frame.absolute_view_projection * world_bound;
    if !finite(world_bound) || !finite(clip_bound)
        || max(max_abs(world_bound.xyz), world_bound.w) > MAX_PRODUCT_BOUND
        || max(max_abs(clip_bound.xyz), abs(clip_bound.w)) > MAX_PRODUCT_BOUND {
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

// After one quiet fixed tick the last published current poses are this
// interval's previous poses (decision 014): copy them on the GPU instead of
// uploading them. Only nodes active in the last applied gather published a
// pose, and `info` still holds that gather's flags: a gather sends every
// other node's previous pose explicitly before it blends. One node per
// invocation; the capacity follows from the buffer's two halves.
@compute @workgroup_size(256)
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

// Active parents of one depth level, after every shallower level.
@compute @workgroup_size(64)
fn compose_parents(@builtin(global_invocation_id) id: vec3<u32>) {
    let position = level.start + id.x;
    if position >= level.end {
        return;
    }
    let node = parents[position];
    composed[node] = compose(node);
}

// Poses and visibility of every candidate, visited in slot order: each
// flag lands at the candidate's draw-group position for `sum_groups`.
@compute @workgroup_size(256)
fn cull(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.candidates {
        return;
    }
    let candidate = slot_candidates[id.x];
    var visible = 0u;
    let flags = info[candidate.node].y;
    if (flags & NODE_VALID) != 0u {
        visible = VALID;
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
            // Inactive nodes display unchanged authority rows. A node that
            // stopped this interval moves by the rest of its last one: its
            // motion is left to the camera and the history clamp.
            if frame.motion != 0u && (flags & NODE_ACTIVE) != 0u {
                visible |= MOVING;
            }
        }
    }
    scan[candidate.position] = visible;
}

// Visible count of every group of 256 candidates in draw-group order.
@compute @workgroup_size(256)
fn sum_groups(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    var visible = 0u;
    if id.x < frame.candidates {
        visible = scan[id.x] & VISIBLE;
    }
    shared_values[lane] = visible;
    workgroupBarrier();
    for (var stride = GROUP / 2u; stride > 0u; stride = stride >> 1u) {
        if lane < stride {
            shared_values[lane] += shared_values[lane + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u {
        scan[frame.candidates + group.x] = shared_values[0];
    }
}

// Exclusive prefix of the group sums and the total: one workgroup scans 512
// groups per chunk and carries the running sum across chunks.
@compute @workgroup_size(256)
fn scan_groups(@builtin(local_invocation_index) lane: u32) {
    let base = frame.candidates;
    var carry = 0u;
    for (var chunk = 0u; chunk < frame.groups; chunk += 2u * GROUP) {
        let first = chunk + lane * 2u;
        var left = 0u;
        var right = 0u;
        if first < frame.groups {
            left = scan[base + first];
        }
        if first + 1u < frame.groups {
            right = scan[base + first + 1u];
        }
        let pair = left + right;
        shared_values[lane] = pair;
        workgroupBarrier();
        for (var offset = 1u; offset < GROUP; offset = offset << 1u) {
            var addend = 0u;
            if lane >= offset {
                addend = shared_values[lane - offset];
            }
            workgroupBarrier();
            shared_values[lane] += addend;
            workgroupBarrier();
        }
        let exclusive = carry + shared_values[lane] - pair;
        if first < frame.groups {
            scan[base + first] = exclusive;
        }
        if first + 1u < frame.groups {
            scan[base + first + 1u] = exclusive + left;
        }
        carry += shared_values[GROUP - 1u];
        // Every lane read this chunk's total before the next chunk overwrites it.
        workgroupBarrier();
    }
    if lane == 0u {
        scan[base + frame.groups] = carry;
    }
}

fn visible_before(position: u32) -> u32 {
    if position >= frame.candidates {
        return scan[frame.candidates + frame.groups];
    }
    let group = position / GROUP;
    var value = scan[frame.candidates + group];
    for (var index = group * GROUP; index < position; index++) {
        value += scan[index] & VISIBLE;
    }
    return value;
}

// Visible counts become indirect instance counts; the budget keeps the first
// visible instances in submission order.
@compute @workgroup_size(64)
fn batch_args(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame.batches {
        return;
    }
    let batch = batches[id.x];
    let base = visible_before(batch.start);
    let visible = visible_before(batch.start + batch.count) - base;
    var instances = 0u;
    if base < frame.budget {
        instances = min(visible, frame.budget - base);
    }
    batch_bases[id.x] = base;
    draws[id.x] = DrawArgs(batch.index_count, instances, 0u, 0, 0u);
}

// Visible slots keep candidate order within each group's region.
@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    var stored = 0u;
    if id.x < frame.candidates {
        stored = scan[id.x];
    }
    let flag = stored & VISIBLE;
    shared_values[lane] = flag;
    workgroupBarrier();
    for (var offset = 1u; offset < GROUP; offset = offset << 1u) {
        var addend = 0u;
        if lane >= offset {
            addend = shared_values[lane - offset];
        }
        workgroupBarrier();
        shared_values[lane] += addend;
        workgroupBarrier();
    }
    if flag != 0u {
        let global = scan[frame.candidates + group.x] + shared_values[lane] - flag;
        if global < frame.budget {
            let candidate = candidates[id.x];
            let batch = candidate.batch;
            visible_ids[batches[batch].start + (global - batch_bases[batch])] =
                candidate.slot | select(0u, MOVED_SLOT, (stored & MOVING) != 0u);
        }
    }
}
