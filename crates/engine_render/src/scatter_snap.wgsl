// Placed objects on the terrain as drawn (decision 057): each frame, before
// culling, every live object's base moves onto its terrain's finest
// resident heights, as ground cover's roots do (`cover_grow.wgsl`), so
// objects of coarse tiles sit on the finer ground drawn near the camera and
// settle as finer tiles arrive. Its pose's height and its bounding sphere
// follow its base. Objects standing far away (decision 061) hide where
// tiles of the level holding them instead are resident.

struct Snap {
    // Slots and terrains.
    counts: vec4<u32>,
}

// A terrain: its origin (world) and level-0 sample spacing, then where its
// levels start in the directory and how many there are.
struct TerrainInfo {
    origin: vec4<f32>,
    levels: vec4<u32>,
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
    // How the camera shows it (decision 104; `scatter_cull.wgsl`).
    state: u32,
    pad_0: u32,
    pad_1: u32,
    pad_2: u32,
}

// An object's pose: the rows of its affine world matrix.
struct Pose {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

const DEAD: u32 = 0xffffffffu;
const NOT_RESIDENT: u32 = 0xffffffffu;
const TERRAIN_INDEX: u32 = 0x1ffu;
const NEAR_LEVEL_SHIFT: u32 = 16u;
const HIDDEN: u32 = 0x80000000u;

@group(0) @binding(0) var<uniform> snap: Snap;
@group(0) @binding(1) var<storage, read_write> records: array<Record>;
@group(0) @binding(2) var<storage, read_write> poses: array<Pose>;
@group(0) @binding(3) var<storage, read> terrains: array<TerrainInfo>;
@group(0) @binding(4) var<storage, read> directory: array<u32>;
@group(0) @binding(5) var heights: texture_2d_array<f32>;

// The finest resident height of terrain `info` at terrain-local `local`, 1
// where one is resident there (0 beyond its edges or with none), and its
// level.
fn ground_at(info: TerrainInfo, local: vec2<f32>) -> vec3<f32> {
    let base = info.levels.x;
    let spacing = info.origin.w;
    let extent = vec2<f32>(f32(directory[base + 1u]), f32(directory[base + 2u])) * 256.0 * spacing;
    if any(local < vec2<f32>(0.0)) || any(local > extent) {
        return vec3<f32>(0.0);
    }
    for (var index = 0u; index < info.levels.y; index++) {
        let header = base + index * 4u;
        let samples = local / (spacing * f32(1u << index));
        let tiles = vec2<u32>(directory[header + 1u], directory[header + 2u]);
        let tile = min(vec2<u32>(samples / 256.0), tiles - vec2<u32>(1u));
        let layer = directory[directory[header] + tile.y * tiles.x + tile.x];
        if layer == NOT_RESIDENT {
            continue;
        }
        // Layers keep a border sample before the tile's first.
        let within = clamp(samples - vec2<f32>(tile * 256u), vec2<f32>(0.0), vec2<f32>(256.0));
        let corner = min(vec2<u32>(within), vec2<u32>(255u));
        let f = within - vec2<f32>(corner);
        let texel = corner + vec2<u32>(1u);
        let a = textureLoad(heights, texel, layer, 0).r;
        let b = textureLoad(heights, texel + vec2<u32>(1u, 0u), layer, 0).r;
        let c = textureLoad(heights, texel + vec2<u32>(0u, 1u), layer, 0).r;
        let d = textureLoad(heights, texel + vec2<u32>(1u, 1u), layer, 0).r;
        return vec3<f32>(mix(mix(a, b, f.x), mix(c, d, f.x), f.y), 1.0, f32(index));
    }
    return vec3<f32>(0.0);
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
    let pose = poses[id.x];
    let ground = ground_at(info, vec2<f32>(pose.r0.w, pose.r2.w) - info.origin.xz);
    // Far away, an object hides where the level holding it instead is
    // resident, or a finer one.
    let near = (record.terrain >> NEAR_LEVEL_SHIFT) & 0x1fu;
    let hidden = near != 0u && ground.y != 0.0 && u32(ground.z) < near;
    records[id.x].terrain = select(record.terrain & ~HIDDEN, record.terrain | HIDDEN, hidden);
    if ground.y == 0.0 {
        return;
    }
    let base = info.origin.y + ground.x - record.sink;
    poses[id.x].r1.w = base;
    records[id.x].center.y = record.center.y + (base - pose.r1.w);
}
