import package::generated::scatter::{CELLS, Cover, OCCUPANCY_GROUP, QUAD, Splat};
import package::place::placed_tiles::{Pose, Record};
import package::scatter::faces::placed_tiles::placed_shown;

struct ListedDraw {
    vertices: u32,
    instances: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

const NOTHING: f32 = -1.0e4;
const SLOT_BITS: u32 = 0x3fffffffu;
const ACROSS_X: u32 = 0x40000000u;
const ACROSS_Z: u32 = 0x80000000u;

@group(1) @binding(0) var<uniform> splat: Splat;
@group(1) @binding(1) var<storage, read> records: array<Record>;
@group(1) @binding(2) var<storage, read> poses: array<Pose>;
@group(1) @binding(3) var<storage, read> covers: array<Cover>;
@group(1) @binding(4) var<storage, read> listed: array<u32>;
@group(1) @binding(5) var<storage, read_write> listing: array<u32>;
@group(1) @binding(6) var<storage, read_write> listed_draw: ListedDraw;
@group(1) @binding(7) var finer_map: texture_2d<f32>;

const CORNERS = array<vec2<f32>, QUAD>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

struct Footprint {
    root: vec2<f32>,
    radius: f32,
    cover: vec3<f32>,
}

fn footprint_of(slot: u32) -> Footprint {
    var found = Footprint(vec2<f32>(0.0), 0.0, vec3<f32>(0.0));
    if slot >= splat.counts.x {
        return found;
    }
    let record = records[slot];
    if !placed_shown(record) || record.group >= splat.counts.y {
        return found;
    }
    let cover = covers[record.group];
    if cover.density <= 0.0 {
        return found;
    }
    let pose = poses[slot];
    let root = vec3<f32>(pose.r0.w, pose.r1.w, pose.r2.w);
    let texel = splat.area.z / f32(max(splat.counts.w, 1u));
    let radius = cover.footprint * record.scale;
    let widened = max(radius, 0.75 * texel);
    found.root = root.xz;
    found.radius = widened;
    found.cover = vec3<f32>(cover.density * (radius * radius) / (widened * widened),
        root.y + cover.bottom * record.scale - splat.area.w,
        root.y + cover.top * record.scale - splat.area.w);
    return found;
}

fn row_redrawn(row: u32) -> u32 {
    let word = splat.dirty[row / 8u][(row / 2u) % 4u];
    return (word >> ((row % 2u) * CELLS)) & 0xffffu;
}

fn row_averaged(row: u32) -> u32 {
    let word = splat.means[row / 8u][(row / 2u) % 4u];
    return (word >> ((row % 2u) * CELLS)) & 0xffffu;
}

fn redrawn(cell: vec2<u32>) -> bool {
    return ((row_redrawn(cell.y) >> cell.x) & 1u) != 0u;
}

fn reaches_redrawn(first: vec2<f32>, last: vec2<f32>) -> bool {
    let side = f32(splat.place.z);
    let low = vec2<u32>(first / side);
    let high = min(vec2<u32>(ceil(last / side)), vec2<u32>(CELLS)) - vec2<u32>(1u);
    let columns = (0xffffu >> (CELLS - 1u - high.x)) & (0xffffu << low.x);
    for (var row = low.y; row <= high.y; row++) {
        if (row_redrawn(row) & columns) != 0u {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(OCCUPANCY_GROUP)
fn select_covers(@builtin(global_invocation_id) id: vec3<u32>) {
    let found = footprint_of(id.x);
    if found.radius <= 0.0 {
        return;
    }
    let texels = f32(splat.counts.w);
    let middle = (found.root - splat.area.xy) / splat.area.z * texels;
    let reach = found.radius / splat.area.z * texels;
    let low = max(middle - reach, vec2<f32>(0.0));
    let high = min(middle + reach, vec2<f32>(texels));
    if any(low >= high) {
        return;
    }
    for (var part = 0u; part < 4u; part++) {
        let across = vec2<u32>(part & 1u, part >> 1u);
        let shift = vec2<f32>(splat.place.xy) - vec2<f32>(across) * texels;
        let first = max(low + shift, vec2<f32>(0.0));
        let last = min(high + shift, vec2<f32>(texels));
        if any(first >= last) || !reaches_redrawn(first, last) {
            continue;
        }
        let index = atomicAdd(&listed_draw.instances, 1u);
        if index < arrayLength(&listing) {
            listing[index] = id.x | (across.x * ACROSS_X) | (across.y * ACROSS_Z);
        }
    }
}

struct SplatOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) @interpolate(flat) cover: vec3<f32>,
    @location(2) window: vec2<f32>,
}

fn map_clip(texel: vec2<f32>) -> vec4<f32> {
    let texels = f32(splat.counts.w);
    return vec4<f32>(texel.x / texels * 2.0 - 1.0, 1.0 - texel.y / texels * 2.0, 0.5, 1.0);
}

@vertex
fn vs_splat(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> SplatOutput {
    var output: SplatOutput;
    output.clip = vec4<f32>(2.0, 2.0, 0.5, 1.0);
    if instance >= arrayLength(&listed) {
        return output;
    }
    let entry = listed[instance];
    let slot = entry & SLOT_BITS;
    let found = footprint_of(slot);
    if found.radius <= 0.0 {
        return output;
    }
    let corner = CORNERS[vertex];
    let texels = f32(splat.counts.w);
    let window = (found.root + corner * found.radius - splat.area.xy) / splat.area.z * texels;
    let across = vec2<f32>(select(0.0, 1.0, (entry & ACROSS_X) != 0u),
        select(0.0, 1.0, (entry & ACROSS_Z) != 0u));
    output.clip = map_clip(window + vec2<f32>(splat.place.xy) - across * texels);
    output.corner = corner;
    output.cover = found.cover;
    output.window = window;
    return output;
}

@fragment
fn fs_splat(input: SplatOutput) -> @location(0) vec4<f32> {
    let texels = f32(splat.counts.w);
    if any(input.window < vec2<f32>(0.0)) || any(input.window >= vec2<f32>(texels))
        || !redrawn(vec2<u32>(input.clip.xy) / splat.place.z) {
        discard;
    }
    let away = length(input.corner);
    if away > 1.0 {
        discard;
    }
    let one = min(input.cover.x * (1.0 - smoothstep(0.55, 1.0, away)), 1.0);
    return vec4<f32>(input.cover.z, -input.cover.y, 0.0, one);
}

@vertex
fn vs_clear(@builtin(vertex_index) vertex: u32, @builtin(instance_index) cell: u32)
        -> @builtin(position) vec4<f32> {
    let at = vec2<u32>(cell % CELLS, cell / CELLS);
    if !redrawn(at) {
        return vec4<f32>(2.0, 2.0, 0.5, 1.0);
    }
    let corner = (CORNERS[vertex] + vec2<f32>(1.0)) * 0.5;
    return map_clip((vec2<f32>(at) + corner) * f32(splat.place.z));
}

@fragment
fn fs_clear() -> @location(0) vec4<f32> {
    return vec4<f32>(NOTHING, NOTHING, 0.0, 0.0);
}

struct MeansOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) first: vec2<u32>,
    @location(1) @interpolate(flat) finer: vec2<u32>,
}

@vertex
fn vs_means(@builtin(vertex_index) vertex: u32, @builtin(instance_index) cell: u32)
        -> MeansOutput {
    var output: MeansOutput;
    output.clip = vec4<f32>(2.0, 2.0, 0.5, 1.0);
    let at = vec2<u32>(cell % CELLS, cell / CELLS);
    if splat.finer.z == 0 || ((row_averaged(at.y) >> at.x) & 1u) == 0u {
        return output;
    }
    let texels = i32(splat.counts.w);
    let side = texels / i32(2u * CELLS);
    let absolute = splat.finer.xy + vec2<i32>(at);
    let first = ((absolute * side) % texels + texels) % texels;
    let finer = ((absolute * (2 * side)) % texels + texels) % texels;
    let corner = (CORNERS[vertex] + vec2<f32>(1.0)) * 0.5;
    output.clip = map_clip(vec2<f32>(first) + corner * f32(side));
    output.first = vec2<u32>(first);
    output.finer = vec2<u32>(finer);
    return output;
}

@fragment
fn fs_means(input: MeansOutput) -> @location(0) vec4<f32> {
    let local = vec2<u32>(input.clip.xy) - input.first;
    let finer = vec2<i32>(input.finer + local * 2u);
    var coverage = 0.0;
    var heights = vec2<f32>(NOTHING);
    for (var corner = 0u; corner < 4u; corner++) {
        let at = finer + vec2<i32>(i32(corner & 1u), i32(corner >> 1u));
        let texel = textureLoad(finer_map, at, 0);
        coverage += texel.a;
        heights = max(heights, texel.rg);
    }
    return vec4<f32>(heights, 0.0, 0.25 * coverage);
}
