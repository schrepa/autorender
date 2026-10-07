// Occupancy (decision 059): every placed object's footprint drawn into a
// top-down map around the camera, a disc as wide as its draw group's
// footprint at its scale, softening at its rim. One target keeps the
// densest cover over each texel; the other the highest top and lowest
// bottom over it, as heights above the map's reference height. The far
// map (decision 101) adds up covers instead, each footprint at least a
// texel wide carrying its share of the texel, so its texels keep how much
// of their ground covers hide.

struct Splat {
    // The map's center (x and z, world), meters a side and reference
    // height; then slots, draw groups, 1 for the far map, and its texels
    // a side.
    area: vec4<f32>,
    counts: vec4<u32>,
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
    pad_0: u32,
    pad_1: u32,
    pad_2: u32,
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

const DEAD: u32 = 0xffffffffu;
// Objects standing far away while tiles holding them instead are resident
// (decision 061; `scatter.rs`).
const HIDDEN: u32 = 0x80000000u;

@group(0) @binding(0) var<uniform> splat: Splat;
@group(0) @binding(1) var<storage, read> records: array<Record>;
@group(0) @binding(2) var<storage, read> poses: array<Pose>;
@group(0) @binding(3) var<storage, read> covers: array<Cover>;
// How each terrain's placement layers thin (decision 100; `scatter_cull.wgsl`):
// the far map weighs an object standing far away by the share standing
// there, as it stands for the rest of its layer's objects.
@group(0) @binding(4) var<storage, read> thins: array<vec4<f32>>;

const THIN_LAYERS: u32 = 64u;
const TERRAIN_INDEX: u32 = 0x1ffu;
const LAYER_SHIFT: u32 = 9u;
const NO_LAYER: u32 = 0x7fu;
const NEAR_LEVEL_SHIFT: u32 = 16u;

struct SplatOutput {
    @builtin(position) clip: vec4<f32>,
    // The corner around the footprint's center (-1 to 1), and the cover's
    // density, bottom and top above the reference height.
    @location(0) corner: vec2<f32>,
    @location(1) @interpolate(flat) cover: vec3<f32>,
}

const CORNERS = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
);

@vertex
fn vs_splat(@builtin(vertex_index) vertex: u32, @builtin(instance_index) slot: u32)
        -> SplatOutput {
    var output: SplatOutput;
    // Dead and hidden slots and groups without cover collapse outside the
    // map.
    output.clip = vec4<f32>(2.0, 2.0, 0.5, 1.0);
    if slot >= splat.counts.x {
        return output;
    }
    let record = records[slot];
    if record.group == DEAD || (record.terrain & HIDDEN) != 0u
        || record.group >= splat.counts.y {
        return output;
    }
    let cover = covers[record.group];
    if cover.density <= 0.0 {
        return output;
    }
    let pose = poses[slot];
    let root = vec3<f32>(pose.r0.w, pose.r1.w, pose.r2.w);
    let corner = CORNERS[vertex];
    var footprint = cover.footprint * record.scale;
    var density = cover.density;
    if splat.counts.z != 0u {
        // The far map: footprints at least most of a texel wide, carrying
        // their share of it.
        let texel = splat.area.z / f32(max(splat.counts.w, 1u));
        let widened = max(footprint, 0.75 * texel);
        density *= (footprint * footprint) / (widened * widened);
        footprint = widened;
        // An object standing far away stands for its layer's objects that
        // do not.
        let layer = (record.terrain >> LAYER_SHIFT) & NO_LAYER;
        let thin = (record.terrain & TERRAIN_INDEX) * THIN_LAYERS + layer;
        if ((record.terrain >> NEAR_LEVEL_SHIFT) & 0x1fu) != 0u && layer != NO_LAYER
            && thin < arrayLength(&thins) && thins[thin].w > 0.0 {
            density /= max(thins[thin].x, 0.25);
        }
    }
    let point = root.xz + corner * footprint;
    // Map texels run with x across and z down.
    let uv = (point - splat.area.xy) / splat.area.z;
    output.clip = vec4<f32>(uv.x * 2.0, -uv.y * 2.0, 0.5, 1.0);
    output.corner = corner;
    output.cover = vec3<f32>(density, root.y + cover.bottom * record.scale - splat.area.w,
        root.y + cover.top * record.scale - splat.area.w);
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
