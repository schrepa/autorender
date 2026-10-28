struct Job {
    first: vec4<i32>,
    counts: vec4<u32>,
    scale: vec4<f32>,
}

struct Stamp {
    ends: vec4<f32>,
    shape: vec4<f32>,
    bounds: vec4<i32>,
}

@group(0) @binding(0) var<uniform> job: Job;
@group(0) @binding(1) var<storage, read> stamps: array<Stamp>;
@group(0) @binding(2) var<storage, read_write> map: array<atomic<u32>>;

const BLOCK: i32 = 16;
const RIM_START: f32 = 0.9;
const RIM_REACH: f32 = 1.6;

fn resolution() -> i32 {
    return i32(job.counts.x);
}

fn blocks() -> i32 {
    return i32(job.counts.y);
}

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

fn height_at(texel: vec2<i32>) -> f32 {
    let held = unpack2x16float(atomicLoad(&map[track_word(slot_of(texel))]));
    return held.y * (1.0 - saturate(4.0 * held.x / max(held.y, 1.0e-6))) - held.x;
}

var<workgroup> peak: atomic<u32>;

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
    let top = atomicLoad(&peak);
    let last = min(corner + vec2<i32>(7), high);
    let first_block = (corner - first) / BLOCK - vec2<i32>(1);
    let last_block = (last - first) / BLOCK + vec2<i32>(1);
    let block = first_block + vec2<i32>(i32(lane % 4u), i32(lane / 4u));
    if top > 0u && lane < 16u && all(block <= last_block) {
        let size_blocks = blocks();
        let start = slot_of(first) / BLOCK;
        let wrapped = (((start + block) % size_blocks) + size_blocks) % size_blocks;
        atomicMax(&map[u32(wrapped.y * size_blocks + wrapped.x)], top);
    }
}

fn held_at(slot: vec2<i32>, first: vec2<i32>) -> vec2<i32> {
    let size = resolution();
    return first + ((((slot - first) % size) + size) % size);
}

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
