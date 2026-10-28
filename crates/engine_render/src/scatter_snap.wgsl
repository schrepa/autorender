struct Snap {
    counts: vec4<u32>,
    eye: vec4<f32>,
}

struct TerrainInfo {
    origin: vec4<f32>,
    levels: vec4<u32>,
}

struct PaletteEntry {
    color: vec4<f32>,
    params: vec4<f32>,
    maps: vec4<f32>,
    far: vec4<f32>,
    relief: vec4<f32>,
    canopy: vec4<f32>,
    placed: vec4<f32>,
    placed_size: vec4<f32>,
    heights: vec4<f32>,
}

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
@group(0) @binding(6) var<storage, read_write> grounds: array<vec4<u32>>;
@group(0) @binding(7) var control: texture_2d_array<u32>;
const PALETTE_ENTRIES: u32 = 113u;
@group(0) @binding(8) var<uniform> palette: array<PaletteEntry, PALETTE_ENTRIES>;
@group(0) @binding(9) var macro_maps: texture_2d_array<f32>;

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

fn macro_colors(info: TerrainInfo, local: vec2<f32>) -> vec4<f32> {
    let layer = info.levels.w;
    if layer == 0u {
        return vec4<f32>(0.5);
    }
    let extent = vec2<f32>(f32(directory[info.levels.x + 1u]),
        f32(directory[info.levels.x + 2u])) * 256.0 * info.origin.w;
    let size = vec2<i32>(textureDimensions(macro_maps));
    let at = local / extent * vec2<f32>(size) - 0.5;
    let first = vec2<i32>(floor(at));
    let f = at - floor(at);
    let last = size - vec2<i32>(1);
    let a = textureLoad(macro_maps, clamp(first, vec2<i32>(0), last), layer - 1u, 0);
    let b = textureLoad(macro_maps, clamp(first + vec2<i32>(1, 0), vec2<i32>(0), last),
        layer - 1u, 0);
    let c = textureLoad(macro_maps, clamp(first + vec2<i32>(0, 1), vec2<i32>(0), last),
        layer - 1u, 0);
    let d = textureLoad(macro_maps, clamp(first + vec2<i32>(1, 1), vec2<i32>(0), last),
        layer - 1u, 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn ground_color(info: TerrainInfo, local: vec2<f32>, ground: FinestGround) -> u32 {
    let samples = local / (info.origin.w * f32(1u << ground.level));
    let texel = min(vec2<u32>(round(samples)) - ground.tile.first, vec2<u32>(256u))
        + vec2<u32>(1u);
    let value = textureLoad(control, texel, ground.tile.layer, 0).x;
    let share = f32((value >> 16u) & 0xffu) / 255.0;
    let first = palette[min(info.levels.z + (value & 0xffu), PALETTE_ENTRIES - 1u)];
    let second = palette[min(info.levels.z + ((value >> 8u) & 0xffu), PALETTE_ENTRIES - 1u)];
    let far = mix(first.far, second.far, share);
    let vegetated = mix(step(1.0e-6, first.canopy.w), step(1.0e-6, second.canopy.w), share);
    let tint = macro_tint(macro_colors(info, local), vegetated);
    return pack4x8unorm(vec4<f32>(sqrt(saturate(far.rgb * tint)),
        max(saturate(far.w), 1.0 / 255.0)));
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
    let ground_terrain = GroundTerrain(info.levels.x, info.levels.y, info.origin.w);
    let pose = poses[id.x];
    let local = vec2<f32>(pose.r0.w, pose.r2.w) - info.origin.xz;
    let ground = finest_ground(info, ground_terrain, local);
    let near = (record.terrain >> NEAR_LEVEL_SHIFT) & 0x1fu;
    let hidden = near != 0u && ground.found && ground.level < near;
    records[id.x].terrain = select(record.terrain & ~HIDDEN, record.terrain | HIDDEN, hidden);
    if !ground.found {
        return;
    }
    let relative = vec3<f32>(pose.r0.w, info.origin.y + ground.height, pose.r2.w) - snap.eye.xyz;
    let base = info.origin.y
        + true_height(ground_terrain, local, ground.level, ground.tile) - record.sink;
    poses[id.x].r1.w = base;
    records[id.x].center.y = record.center.y + (base - pose.r1.w);
    if grounds[id.x].z != 0u {
        grounds[id.x].w = ground_color(info, local, ground);
    }
}
