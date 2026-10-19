// Placed objects on the terrain as drawn: each frame the eye moved or
// objects or the resident heights changed, before culling, every live
// object's base moves onto its terrain's ground as the terrain draws it at
// the object's distance (`terrain_ground.wgsl`), as ground cover's roots
// do (`cover_grow.wgsl`), so objects stand on the ground drawn under them
// as its levels morph and settle as finer tiles arrive. Its pose's height
// and its bounding sphere follow its base. Objects standing far away hide
// where tiles of the level holding them instead are resident.

struct Snap {
    // Slots and terrains.
    counts: vec4<u32>,
    // The eye (world), whose distance the ground's levels morph by.
    eye: vec4<f32>,
}

// A terrain: its origin (world) and level-0 sample spacing, where its
// levels start in the directory and how many there are, and the meters
// from the eye up to which each of its first eight levels draws
// (`terrain_ground.wgsl`'s `GroundTerrain`).
struct TerrainInfo {
    origin: vec4<f32>,
    levels: vec4<u32>,
    ranges: array<vec4<f32>, 2>,
}

// `scatter_cull.wgsl`'s record: the bounding sphere, draw group and scale,
// then the terrain word (`scatter.rs`) and how far its base sinks.
struct Record {
    center: vec3<f32>,
    radius: f32,
    group: u32,
    scale: f32,
    terrain: u32,
    sink: f32,
    // How the camera shows it (`scatter_cull.wgsl`).
    state: u32,
    pad: array<u32, 3>,
}

// An object's pose: the rows of its affine world matrix.
struct Pose {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

const DEAD: u32 = 0xffffffffu;
const TERRAIN_INDEX: u32 = 0x1ffu;
const NEAR_LEVEL_SHIFT: u32 = 16u;
const HIDDEN: u32 = 0x80000000u;

@group(0) @binding(0) var<uniform> snap: Snap;
@group(0) @binding(1) var<storage, read_write> records: array<Record>;
@group(0) @binding(2) var<storage, read_write> poses: array<Pose>;
@group(0) @binding(3) var<storage, read> terrains: array<TerrainInfo>;
@group(0) @binding(4) var<storage, read> directory: array<u32>;
@group(0) @binding(5) var heights: texture_2d_array<f32>;

// The ground of terrain `info` at terrain-local `local` from its finest
// resident level: its height there, its samples blended, whether one is
// resident there (none beyond its edges), the level and its tile.
struct FinestGround {
    height: f32,
    found: bool,
    level: u32,
    tile: LevelTile,
}

fn finest_ground(info: TerrainInfo, terrain: GroundTerrain, local: vec2<f32>) -> FinestGround {
    var ground: FinestGround;
    let extent = vec2<f32>(f32(directory[info.levels.x + 1u]),
        f32(directory[info.levels.x + 2u])) * 256.0 * info.origin.w;
    if any(local < vec2<f32>(0.0)) || any(local > extent) {
        return ground;
    }
    for (var index = 0u; index < info.levels.y; index++) {
        let samples = local / (info.origin.w * f32(1u << index));
        let tile = level_tile(terrain, index, samples);
        if tile.layer == NOT_RESIDENT {
            continue;
        }
        let within = clamp(samples - vec2<f32>(tile.first), vec2<f32>(0.0), vec2<f32>(256.0));
        let cell = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(cell);
        let corner = tile.first + cell;
        let a = tile_height(tile, corner);
        let b = tile_height(tile, corner + vec2<u32>(1u, 0u));
        let c = tile_height(tile, corner + vec2<u32>(0u, 1u));
        let d = tile_height(tile, corner + vec2<u32>(1u, 1u));
        ground.height = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        ground.found = true;
        ground.level = index;
        ground.tile = tile;
        return ground;
    }
    return ground;
}

@compute @workgroup_size(64)
fn snap_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= snap.counts.x {
        return;
    }
    let record = records[id.x];
    let terrain = record.terrain & TERRAIN_INDEX;
    if record.group == DEAD || terrain >= snap.counts.y {
        return;
    }
    let info = terrains[terrain];
    let ground_terrain = GroundTerrain(info.levels.x, info.levels.y, info.origin.w,
        info.ranges);
    let pose = poses[id.x];
    let local = vec2<f32>(pose.r0.w, pose.r2.w) - info.origin.xz;
    let ground = finest_ground(info, ground_terrain, local);
    // Far away, an object hides where the level holding it instead is
    // resident, or a finer one.
    let near = (record.terrain >> NEAR_LEVEL_SHIFT) & 0x1fu;
    let hidden = near != 0u && ground.found && ground.level < near;
    records[id.x].terrain = select(record.terrain & ~HIDDEN, record.terrain | HIDDEN, hidden);
    if !ground.found {
        return;
    }
    // The ground drawn under it morphs by its distance from the eye, as
    // ground cover's does by its cell's.
    let away = distance(vec3<f32>(pose.r0.w, info.origin.y + ground.height, pose.r2.w),
        snap.eye.xyz);
    let base = info.origin.y
        + drawn_height(ground_terrain, local, away, ground.level, ground.tile) - record.sink;
    poses[id.x].r1.w = base;
    records[id.x].center.y = record.center.y + (base - pose.r1.w);
}
