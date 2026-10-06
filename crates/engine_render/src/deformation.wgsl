// Runtime deformation's map (decision 083): around the camera, how deep
// tracks sink the ground and how high their rims raise it, texel by texel,
// and the slopes they leave. Texels wrap: world texel `i` lives at slot
// `i mod resolution` along each axis, so the map follows the camera without
// moving what it holds. The buffer holds, as words: each block's (16 by 16
// texels) deepest track or highest rim, widened by a block so the ground
// beside tracks finds them; each texel's track and rim (`pack2x16float`);
// and each texel's slope of the height tracks and rims leave
// (`pack2x16float`, along x and z). Values are in the map's reference
// units: shading multiplies them by how much of a track is left since the
// reference time, and rebases scale them all when that falls.

struct Job {
    // The map's first texel (world texel indices along x and z) and, for
    // scrolls, the last map's.
    first: vec4<i32>,
    // The map's texels and blocks along a side, the dispatch's first stamp
    // and its stamps.
    counts: vec4<u32>,
    // Meters per texel, and the factor rebases scale every value by.
    scale: vec4<f32>,
}

// A stamp: a capsule from `ends.xy` to `ends.zw` (world x and z); its
// radius, depth along its middle and flatness, and its rim's height; and
// the world texels it reaches, first and last.
struct Stamp {
    ends: vec4<f32>,
    shape: vec4<f32>,
    bounds: vec4<i32>,
}

@group(0) @binding(0) var<uniform> job: Job;
@group(0) @binding(1) var<storage, read> stamps: array<Stamp>;
@group(0) @binding(2) var<storage, read_write> map: array<atomic<u32>>;

const BLOCK: i32 = 16;
// Where a stamp's rim starts and ends, in radii from its middle line
// (`engine_core::RIM_START` and `RIM_REACH`).
const RIM_START: f32 = 0.9;
const RIM_REACH: f32 = 1.6;

fn resolution() -> i32 {
    return i32(job.counts.x);
}

fn blocks() -> i32 {
    return i32(job.counts.y);
}

// The slot of world texel `texel`.
fn slot_of(texel: vec2<i32>) -> vec2<i32> {
    let size = resolution();
    return ((texel % size) + size) % size;
}

fn block_word(slot: vec2<i32>) -> u32 {
    let block = slot / BLOCK;
    return u32(block.y * blocks() + block.x);
}

fn track_word(slot: vec2<i32>) -> u32 {
    let size = resolution();
    return u32(blocks() * blocks() + slot.y * size + slot.x);
}

fn slope_word(slot: vec2<i32>) -> u32 {
    let size = resolution();
    return u32(blocks() * blocks() + size * size + slot.y * size + slot.x);
}

// How deep `stamp` sinks the ground at world `point` and how high its rim
// stands there (`engine_core::Track::at`).
fn stamp_at(stamp: Stamp, point: vec2<f32>) -> vec2<f32> {
    let tail = stamp.ends.xy;
    let along = stamp.ends.zw - tail;
    let span = dot(along, along);
    let t = select(0.0, clamp(dot(point - tail, along) / max(span, 1.0e-12), 0.0, 1.0),
        span > 0.0);
    let s = distance(point, tail + along * t) / stamp.shape.x;
    let cap = sqrt(max(1.0 - s * s, 0.0));
    let edge = saturate((s - 0.7) / 0.3);
    let level = 1.0 - edge * edge * (3.0 - 2.0 * edge);
    let carve = select(0.0, stamp.shape.y * mix(cap, level, stamp.shape.z), s < 1.0);
    let u = (s - RIM_START) / (RIM_REACH - RIM_START);
    let hump = u * (1.0 - u);
    let rim = select(0.0, stamp.shape.w * 16.0 * hump * hump, u >= 0.0 && u <= 1.0);
    return vec2<f32>(carve, rim);
}

// Each workgroup presses 8 by 8 texels of its stamp's reach; every texel
// takes the deepest track and highest rim of all the dispatch's stamps
// reaching it, so texels two stamps share end the same whichever writes
// last.
@compute @workgroup_size(8, 8)
fn press(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>) {
    let own = stamps[job.counts.z + group.z];
    let texel = own.bounds.xy + vec2<i32>(group.xy * 8u + local.xy);
    if any(texel > own.bounds.zw) {
        return;
    }
    let point = (vec2<f32>(texel) + 0.5) * job.scale.x;
    var pressed = vec2<f32>(0.0);
    for (var index = 0u; index < job.counts.w; index++) {
        let stamp = stamps[job.counts.z + index];
        if any(texel < stamp.bounds.xy) || any(texel > stamp.bounds.zw) {
            continue;
        }
        pressed = max(pressed, stamp_at(stamp, point));
    }
    if all(pressed <= vec2<f32>(0.0)) {
        return;
    }
    let word = track_word(slot_of(texel));
    let held = unpack2x16float(atomicLoad(&map[word]));
    atomicStore(&map[word], pack2x16float(max(held, pressed)));
}

// The height tracks and rims leave at world texel `texel`, reference
// units: rims only where no track sinks the ground.
fn height_at(texel: vec2<i32>) -> f32 {
    let held = unpack2x16float(atomicLoad(&map[track_word(slot_of(texel))]));
    return held.y * (1.0 - saturate(4.0 * held.x / max(held.y, 1.0e-6))) - held.x;
}

// The highest track or rim of a workgroup's texels.
var<workgroup> peak: atomic<u32>;

// After a dispatch's stamps all pressed, each workgroup takes 8 by 8
// texels of its stamp's reach and a texel more: their slopes, and their
// highest value into the blocks around them.
@compute @workgroup_size(8, 8)
fn slopes(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        atomicStore(&peak, 0u);
    }
    workgroupBarrier();
    let own = stamps[job.counts.z + group.z];
    let size = resolution();
    let first = job.first.xy;
    let low = max(own.bounds.xy - vec2<i32>(1), first);
    let high = min(own.bounds.zw + vec2<i32>(1), first + vec2<i32>(size - 1));
    let corner = low + vec2<i32>(group.xy * 8u);
    let texel = corner + vec2<i32>(local.xy);
    if all(texel <= high) {
        // Neighbors beyond the map stand in with the texel itself.
        let here = height_at(texel);
        let inside_low = texel > first;
        let inside_high = texel < first + vec2<i32>(size - 1);
        let west = select(here, height_at(texel - vec2<i32>(1, 0)), inside_low.x);
        let east = select(here, height_at(texel + vec2<i32>(1, 0)), inside_high.x);
        let north = select(here, height_at(texel - vec2<i32>(0, 1)), inside_low.y);
        let south = select(here, height_at(texel + vec2<i32>(0, 1)), inside_high.y);
        let span = 2.0 * job.scale.x;
        let slope = vec2<f32>(east - west, south - north) / span;
        let slot = slot_of(texel);
        atomicStore(&map[slope_word(slot)], pack2x16float(slope));
        let held = unpack2x16float(atomicLoad(&map[track_word(slot)]));
        atomicMax(&peak, bitcast<u32>(max(max(held.x, held.y), 0.0)));
    }
    workgroupBarrier();
    // The blocks of the workgroup's texels and one more around them: at
    // most four along each side, a lane each.
    let top = atomicLoad(&peak);
    let last = min(corner + vec2<i32>(7), high);
    let first_block = (corner - first) / BLOCK - vec2<i32>(1);
    let last_block = (last - first) / BLOCK + vec2<i32>(1);
    let block = first_block + vec2<i32>(i32(lane % 4u), i32(lane / 4u));
    if top > 0u && lane < 16u && all(block <= last_block) {
        // Blocks beyond the map wrap onto its far side, which only widens
        // what the far side's ground looks through.
        let size_blocks = blocks();
        let start = slot_of(first) / BLOCK;
        let wrapped = (((start + block) % size_blocks) + size_blocks) % size_blocks;
        atomicMax(&map[u32(wrapped.y * size_blocks + wrapped.x)], top);
    }
}

// The world texel slot `slot` holds while the map starts at `first`.
fn held_at(slot: vec2<i32>, first: vec2<i32>) -> vec2<i32> {
    let size = resolution();
    return first + ((((slot - first) % size) + size) % size);
}

// As the map moves, slots now holding other ground start empty.
@compute @workgroup_size(8, 8)
fn scroll(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = resolution();
    let slot = vec2<i32>(id.xy);
    if any(slot >= vec2<i32>(size)) {
        return;
    }
    if all(held_at(slot, job.first.xy) == held_at(slot, job.first.zw)) {
        return;
    }
    atomicStore(&map[track_word(slot)], 0u);
    atomicStore(&map[slope_word(slot)], 0u);
    if all(slot % BLOCK == vec2<i32>(0)) {
        atomicStore(&map[block_word(slot)], 0u);
    }
}

// Empties the whole map.
@compute @workgroup_size(8, 8)
fn wipe(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = vec2<i32>(id.xy);
    if any(slot >= vec2<i32>(resolution())) {
        return;
    }
    atomicStore(&map[track_word(slot)], 0u);
    atomicStore(&map[slope_word(slot)], 0u);
    if all(slot % BLOCK == vec2<i32>(0)) {
        atomicStore(&map[block_word(slot)], 0u);
    }
}

// Scales every value by `job.scale.y`: tracks refilled since the
// reference time, which moves to now.
@compute @workgroup_size(8, 8)
fn rebase(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = vec2<i32>(id.xy);
    if any(slot >= vec2<i32>(resolution())) {
        return;
    }
    let factor = job.scale.y;
    let track = track_word(slot);
    atomicStore(&map[track], pack2x16float(unpack2x16float(atomicLoad(&map[track])) * factor));
    let slope = slope_word(slot);
    atomicStore(&map[slope], pack2x16float(unpack2x16float(atomicLoad(&map[slope])) * factor));
    if all(slot % BLOCK == vec2<i32>(0)) {
        let block = block_word(slot);
        atomicStore(&map[block], bitcast<u32>(bitcast<f32>(atomicLoad(&map[block])) * factor));
    }
}
