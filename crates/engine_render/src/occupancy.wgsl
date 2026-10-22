// Occupancy: every placed object's footprint drawn into a
// top-down map around the camera, a disc as wide as its draw group's
// footprint at its scale, softening at its rim. One target keeps the
// densest cover over each texel; the other the highest top and lowest
// bottom over it, as heights above the map's reference height. The far
// map adds up covers instead, each footprint at least a
// texel wide carrying its share of the texel, so its texels keep how much
// of their ground covers hide.
//
// A map wraps: a world point falls on the texel its place in its map-wide
// page gives, and a window CELLS cells a side follows the camera over the
// map. A frame redraws some of its cells: a selection lists the objects
// whose footprints reach them inside the window, a footprint crossing an
// edge of the map once more on the other side; the cells are cleared, then
// the listed footprints drawn into them alone.

struct Splat {
    // The window's first corner (x and z, world), meters a side and the
    // reference height; then slots, draw groups, 1 for the far map, and the
    // map's texels a side.
    area: vec4<f32>,
    counts: vec4<u32>,
    // The texel the window's first corner falls on (x and y), and texels a
    // cell side.
    place: vec4<u32>,
    // The cells redrawn, a bit each: the map's cell at column x and row y
    // is bit CELLS y + x.
    dirty: array<vec4<u32>, 2>,
}

// `scatter_cull.wgsl`'s record.
struct Record {
    center: vec3<f32>,
    radius: f32,
    group: u32,
    scale: f32,
    terrain: u32,
    sink: f32,
    state: u32,
    pad: array<u32, 3>,
}

// A placed object's pose: the rows of its affine world matrix.
struct Pose {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

// A draw group's cover in mesh units: its footprint's radius, the bottom
// and top of what it covers above its origin, and its density.
struct Cover {
    footprint: f32,
    bottom: f32,
    top: f32,
    density: f32,
}

// The arguments of the draw of the listed footprints, whose instances the
// selection counts.
struct ListedDraw {
    vertices: u32,
    instances: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

const DEAD: u32 = 0xffffffffu;
// Objects standing far away while tiles holding them instead are resident
// (`scatter.rs`).
const HIDDEN: u32 = 0x80000000u;
// Cells along a side of the map, and vertices of a footprint's or a cell's
// quad.
const CELLS: u32 = 16u;
const QUAD: u32 = 6u;
// What heights hold where nothing covers: no top, no bottom.
const NOTHING: f32 = -1.0e4;
// A listed footprint's slot, and its bits of being drawn past the map's
// edge on its other side along x and along z.
const SLOT_BITS: u32 = 0x3fffffffu;
const ACROSS_X: u32 = 0x40000000u;
const ACROSS_Z: u32 = 0x80000000u;

@group(0) @binding(0) var<uniform> splat: Splat;
@group(0) @binding(1) var<storage, read> records: array<Record>;
@group(0) @binding(2) var<storage, read> poses: array<Pose>;
@group(0) @binding(3) var<storage, read> covers: array<Cover>;
// How each terrain's placement layers thin (`scatter_cull.wgsl`):
// the far map weighs an object standing far away by the share standing
// there, as it stands for the rest of its layer's objects.
@group(0) @binding(4) var<storage, read> thins: array<vec4<f32>>;
// The listed footprints as the draw reads them, as the selection writes
// them, and the draw's arguments.
@group(0) @binding(5) var<storage, read> listed: array<u32>;
@group(0) @binding(6) var<storage, read_write> listing: array<u32>;
@group(0) @binding(7) var<storage, read_write> listed_draw: ListedDraw;

const THIN_LAYERS: u32 = 64u;
const TERRAIN_INDEX: u32 = 0x1ffu;
const LAYER_SHIFT: u32 = 9u;
const NO_LAYER: u32 = 0x7fu;
const NEAR_LEVEL_SHIFT: u32 = 16u;

const CORNERS = array<vec2<f32>, QUAD>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

// Where a slot's cover lies: its footprint's root (x and z, world) and
// radius, none where it covers nothing; and the cover's density, bottom
// and top above the reference height.
struct Footprint {
    root: vec2<f32>,
    radius: f32,
    cover: vec3<f32>,
}

// Slot `slot`'s footprint in the map: dead and hidden slots and groups
// without cover have none.
fn footprint_of(slot: u32) -> Footprint {
    var found = Footprint(vec2<f32>(0.0), 0.0, vec3<f32>(0.0));
    if slot >= splat.counts.x {
        return found;
    }
    let record = records[slot];
    if record.group == DEAD || (record.terrain & HIDDEN) != 0u
        || record.group >= splat.counts.y {
        return found;
    }
    let cover = covers[record.group];
    if cover.density <= 0.0 {
        return found;
    }
    let pose = poses[slot];
    let root = vec3<f32>(pose.r0.w, pose.r1.w, pose.r2.w);
    var radius = cover.footprint * record.scale;
    var density = cover.density;
    if splat.counts.z != 0u {
        // The far map: footprints at least most of a texel wide, carrying
        // their share of it.
        let texel = splat.area.z / f32(max(splat.counts.w, 1u));
        let widened = max(radius, 0.75 * texel);
        density *= (radius * radius) / (widened * widened);
        radius = widened;
    }
    found.root = root.xz;
    found.radius = radius;
    found.cover = vec3<f32>(density, root.y + cover.bottom * record.scale - splat.area.w,
        root.y + cover.top * record.scale - splat.area.w);
    return found;
}

// `density` of `record`'s object in the far map: an object standing far
// away stands for its layer's objects that do not.
fn thinned(record: Record, density: f32) -> f32 {
    let layer = (record.terrain >> LAYER_SHIFT) & NO_LAYER;
    let thin = (record.terrain & TERRAIN_INDEX) * THIN_LAYERS + layer;
    if ((record.terrain >> NEAR_LEVEL_SHIFT) & 0x1fu) != 0u && layer != NO_LAYER
        && thin < arrayLength(&thins) && thins[thin].w > 0.0 {
        return density / max(thins[thin].x, 0.25);
    }
    return density;
}

// The bits of the cells redrawn in row `row` of the map, a bit a column.
fn row_redrawn(row: u32) -> u32 {
    let word = splat.dirty[row / 8u][(row / 2u) % 4u];
    return (word >> ((row % 2u) * CELLS)) & 0xffffu;
}

// Whether the map's cell at column `cell.x` and row `cell.y` redraws.
fn redrawn(cell: vec2<u32>) -> bool {
    return ((row_redrawn(cell.y) >> cell.x) & 1u) != 0u;
}

// Whether a cell redrawn holds any of the map's texels from `first` to
// `last`.
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

// Lists slot `id.x`'s footprint for the draw where it reaches a cell
// redrawn inside the window: once where it falls in the map from the texel
// the window's first corner falls on, and once more on the other side of
// each edge of the map it crosses.
@compute @workgroup_size(64)
fn select_covers(@builtin(global_invocation_id) id: vec3<u32>) {
    let found = footprint_of(id.x);
    if found.radius <= 0.0 {
        return;
    }
    let texels = f32(splat.counts.w);
    // The footprint's texels inside the window, from its first corner.
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
    // The corner around the footprint's center (-1 to 1), the cover's
    // density, bottom and top above the reference height, and the texels
    // from the window's first corner.
    @location(0) corner: vec2<f32>,
    @location(1) @interpolate(flat) cover: vec3<f32>,
    @location(2) window: vec2<f32>,
}

// Clip space of the map's texel `texel`: texels run with x across and z
// down.
fn map_clip(texel: vec2<f32>) -> vec4<f32> {
    let texels = f32(splat.counts.w);
    return vec4<f32>(texel.x / texels * 2.0 - 1.0, 1.0 - texel.y / texels * 2.0, 0.5, 1.0);
}

@vertex
fn vs_splat(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32)
        -> SplatOutput {
    var output: SplatOutput;
    // Footprints covering nothing, and instances past the list, collapse
    // outside the map.
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
    var density = found.cover.x;
    if splat.counts.z != 0u {
        density = thinned(records[slot], density);
    }
    let corner = CORNERS[vertex];
    let texels = f32(splat.counts.w);
    let window = (found.root + corner * found.radius - splat.area.xy) / splat.area.z * texels;
    // Where it falls in the map from the texel the window's first corner
    // falls on, past the map's edge on its other side where listed so.
    let across = vec2<f32>(select(0.0, 1.0, (entry & ACROSS_X) != 0u),
        select(0.0, 1.0, (entry & ACROSS_Z) != 0u));
    output.clip = map_clip(window + vec2<f32>(splat.place.xy) - across * texels);
    output.corner = corner;
    output.cover = vec3<f32>(density, found.cover.yz);
    output.window = window;
    return output;
}

struct SplatTargets {
    // The cover's density here.
    @location(0) coverage: vec4<f32>,
    // Its top, and its bottom negated, so both keep their extremes under
    // a maximum.
    @location(1) heights: vec4<f32>,
}

@fragment
fn fs_splat(input: SplatOutput) -> SplatTargets {
    // Only inside the window, in the cells redrawn.
    let texels = f32(splat.counts.w);
    if any(input.window < vec2<f32>(0.0)) || any(input.window >= vec2<f32>(texels))
        || !redrawn(vec2<u32>(input.clip.xy) / splat.place.z) {
        discard;
    }
    let away = length(input.corner);
    if away > 1.0 {
        discard;
    }
    var targets: SplatTargets;
    targets.coverage = vec4<f32>(input.cover.x * (1.0 - smoothstep(0.55, 1.0, away)), 0.0, 0.0,
        0.0);
    targets.heights = vec4<f32>(input.cover.z, -input.cover.y, 0.0, 0.0);
    return targets;
}

// The quad of the map's cell `cell` (CELLS a row) where it redraws: the
// others collapse outside the map.
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

// A cell redrawn holds no cover until its footprints draw.
@fragment
fn fs_clear() -> SplatTargets {
    var targets: SplatTargets;
    targets.coverage = vec4<f32>(0.0);
    targets.heights = vec4<f32>(NOTHING, NOTHING, 0.0, 0.0);
    return targets;
}
