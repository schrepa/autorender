// Sun shadows (decision 020): caster selection and depth-only drawing.
//
// `select_casters` visits every candidate once. It transforms the candidate's
// mesh bounds into light space and appends its slot to the instance IDs of
// each shadow map it can darken: the near map that follows the camera, the
// far map redrawn every frame, and the region of the kept far map redrawn
// this frame. GPU-presented frames also track how many frames each slot's
// pose stayed the same: still slots move into the kept map, which is drawn
// only when a region of it is dirty, and leave the per-frame map a little
// later, once the kept map surely holds them.

struct Cull {
    // World to light space: the sun's right, up and away-from-sun axes.
    light: array<vec4<f32>, 3>,
    // Light-space rectangles (minimum x and y, maximum x and y) of the near
    // map, the far maps and the region of the kept map redrawn this frame.
    near: vec4<f32>,
    far: vec4<f32>,
    refresh: vec4<f32>,
    // Light-space depth beyond which casters cannot darken a receiver of
    // the near map, the far maps and the refreshed region.
    depths: vec4<f32>,
    candidates: u32,
    groups: u32,
    // Instance IDs per map.
    capacity: u32,
    // STILLNESS | REFRESH | NEAR.
    mode: u32,
    // Kept-map strips: light-space y of the first, the strip height and the
    // strip count.
    strips: vec4<f32>,
    // Start of each draw group's ID region within a map, four per vector.
    starts: array<vec4<u32>, 32>,
}

struct Candidate {
    slot: u32,
    node: u32,
    batch: u32,
    position: u32,
}

// A draw group (the presentation record, whose padding holds the group).
struct Batch {
    start: u32,
    count: u32,
    index_count: u32,
    group: u32,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}

struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> cull: Cull;
@group(0) @binding(1) var<storage, read> candidates: array<Candidate>;
@group(0) @binding(2) var<storage, read> batches: array<Batch>;
@group(0) @binding(3) var<storage, read> poses: array<Rows>;
// GPU-presented frames: presentation's per-candidate flags (VALID marks a
// candidate whose pose this frame's culling wrote).
@group(0) @binding(4) var<storage, read> flags: array<u32>;
// Per slot: its pose key, and its still-frame count with the kept-map
// strips it was last drawn into.
@group(0) @binding(5) var<storage, read_write> stillness: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> args: array<DrawArgs>;
// [0]: kept-map strips to redraw, one bit each; [1..4]: this frame's casters
// of the near, moving and kept maps.
@group(0) @binding(8) var<storage, read_write> dirty: array<atomic<u32>, 4>;

const STILLNESS: u32 = 1u;
const REFRESH: u32 = 2u;
const NEAR: u32 = 4u;
// Presentation's candidate flag of a valid pose.
const VALID: u32 = 4u;
// Frames a pose must stay unchanged before its slot is kept, and until it
// leaves the per-frame map: long enough for the dirty strips to be read
// back and redrawn one per frame.
const KEEP_AFTER: u32 = 8u;
const MOVING_UNTIL: u32 = 64u;
const NEAR_MAP: u32 = 0u;
const MOVING_MAP: u32 = 1u;
const KEPT_MAP: u32 = 2u;

fn mix_key(key: u32, value: u32) -> u32 {
    var x = (key ^ value) * 0x9e3779b1u;
    return x ^ (x >> 15u);
}

// A 32-bit fingerprint of a pose and its draw group.
fn pose_key(pose: Rows, group: u32) -> u32 {
    var key = mix_key(0x2545f491u, group);
    for (var row = 0u; row < 3u; row++) {
        var values: vec4<f32>;
        switch row {
            case 0u: { values = pose.r0; }
            case 1u: { values = pose.r1; }
            default: { values = pose.r2; }
        }
        let bits = bitcast<vec4<u32>>(values);
        key = mix_key(mix_key(mix_key(mix_key(key, bits.x), bits.y), bits.z), bits.w);
    }
    return key;
}

fn overlaps(rectangle: vec4<f32>, low: vec3<f32>, high: vec3<f32>, depth: f32) -> bool {
    return all(high.xy >= rectangle.xy) && all(low.xy <= rectangle.zw) && low.z <= depth;
}

fn group_start(group: u32) -> u32 {
    return cull.starts[group / 4u][group % 4u];
}

fn append(map: u32, group: u32, slot: u32) {
    let offset = atomicAdd(&args[map * cull.groups + group].instance_count, 1u);
    ids[map * cull.capacity + group_start(group) + offset] = slot;
}

// Kept-map strips covering light-space heights `low` to `high`, one bit each.
fn strip_bits(low: f32, high: f32) -> u32 {
    let count = cull.strips.z;
    let first = u32(clamp(floor((low - cull.strips.x) / cull.strips.y), 0.0, count - 1.0));
    let last = u32(clamp(floor((high - cull.strips.x) / cull.strips.y), 0.0, count - 1.0));
    return (0xffffffffu >> (31u - last)) & (0xffffffffu << first);
}

@compute @workgroup_size(256)
fn select_casters(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= cull.candidates {
        return;
    }
    let candidate = candidates[id.x];
    let classify = (cull.mode & STILLNESS) != 0u;
    if classify && (flags[candidate.position] & VALID) == 0u {
        return;
    }
    let pose = poses[candidate.slot];
    let batch = batches[candidate.batch];
    // Meshes beyond the draw group limit cast no shadow.
    if batch.group >= cull.groups {
        return;
    }
    // World bounds of the posed mesh bounds, then their light-space box.
    let center = (batch.bounds_min.xyz + batch.bounds_max.xyz) * 0.5;
    let half = (batch.bounds_max.xyz - batch.bounds_min.xyz) * 0.5;
    let world_center = vec3<f32>(dot(pose.r0.xyz, center) + pose.r0.w,
        dot(pose.r1.xyz, center) + pose.r1.w, dot(pose.r2.xyz, center) + pose.r2.w);
    let world_half = vec3<f32>(dot(abs(pose.r0.xyz), half), dot(abs(pose.r1.xyz), half),
        dot(abs(pose.r2.xyz), half));
    let light_center = vec3<f32>(dot(cull.light[0].xyz, world_center),
        dot(cull.light[1].xyz, world_center), dot(cull.light[2].xyz, world_center));
    let light_half = vec3<f32>(dot(abs(cull.light[0].xyz), world_half),
        dot(abs(cull.light[1].xyz), world_half), dot(abs(cull.light[2].xyz), world_half));
    let low = light_center - light_half;
    let high = light_center + light_half;
    // Non-finite bounds fail every comparison and cast nothing.
    var still = MOVING_UNTIL;
    var kept = false;
    if classify {
        let key = pose_key(pose, batch.group);
        let state = stillness[candidate.slot];
        var count = state.y & 0xffu;
        var strips = state.y >> 8u;
        if state.x == key {
            count = min(count + 1u, 255u);
        } else {
            // A kept slot that moved leaves its shadow in the strips it was
            // drawn into: redraw them.
            if count >= KEEP_AFTER && strips != 0u {
                atomicOr(&dirty[0], strips);
            }
            count = 0u;
            strips = 0u;
        }
        if count == KEEP_AFTER && overlaps(cull.far, low, high, cull.depths.y) {
            // Newly still: its strips must be drawn with it.
            atomicOr(&dirty[0], strip_bits(low.y, high.y));
        }
        still = count;
        kept = count >= KEEP_AFTER;
        if kept && (cull.mode & REFRESH) != 0u && overlaps(cull.refresh, low, high, cull.depths.z) {
            append(KEPT_MAP, batch.group, candidate.slot);
            strips = strip_bits(low.y, high.y);
        }
        let packed = count | (strips << 8u);
        if state.x != key || state.y != packed {
            stillness[candidate.slot] = vec2<u32>(key, packed);
        }
    }
    if (cull.mode & NEAR) != 0u && overlaps(cull.near, low, high, cull.depths.x) {
        append(NEAR_MAP, batch.group, candidate.slot);
    }
    if still < MOVING_UNTIL || !classify {
        if overlaps(cull.far, low, high, cull.depths.y) {
            append(MOVING_MAP, batch.group, candidate.slot);
        }
    }
}

// Totals of this frame's casters per map, read back with the dirty strips
// for profiling counters.
@compute @workgroup_size(1)
fn count_casters() {
    for (var map = 0u; map < 3u; map++) {
        var total = 0u;
        for (var group = 0u; group < cull.groups; group++) {
            total += atomicLoad(&args[map * cull.groups + group].instance_count);
        }
        atomicStore(&dirty[1u + map], total);
    }
}

// Depth-only drawing into one shadow map.

struct View {
    // World to this map's clip space.
    clip: mat4x4<f32>,
}

// Its own binding: the selection pass above owns binding 0.
@group(0) @binding(9) var<uniform> view: View;
@group(1) @binding(0) var<storage, read> slot_poses: array<Rows>;

fn project(world: vec3<f32>) -> vec4<f32> {
    var clip = view.clip * vec4<f32>(world, 1.0);
    // Casters between the sun and the map's near plane still darken its
    // receivers: flatten them onto it.
    clip.z = max(clip.z, 0.0);
    return clip;
}

@vertex
fn vs_affine(@location(0) position: vec3<f32>, @location(2) slot: u32) -> @builtin(position) vec4<f32> {
    let pose = slot_poses[slot];
    let point = vec4<f32>(position, 1.0);
    return project(vec3<f32>(dot(pose.r0, point), dot(pose.r1, point), dot(pose.r2, point)));
}

@vertex
fn vs_full(
    @location(0) position: vec3<f32>,
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(model_0, model_1, model_2, model_3);
    let world = model * vec4<f32>(position, 1.0);
    return project(world.xyz / world.w);
}

// A triangle over the whole map at the far depth: cleared regions of the
// kept map, bounded by the scissor.
@vertex
fn vs_clear(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 1.0, 1.0);
}
