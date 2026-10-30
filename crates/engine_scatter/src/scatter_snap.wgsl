import package::face::cover::cover_tint;
import package::face::ground::{
    GroundTerrain, LevelTile, NOT_RESIDENT, TerrainInfo, ground_colors, ground_extent, ground_of,
    level_tile, macro_color, material_look, tile_height, tile_material, true_height
};
import package::face::placed::{DEAD, HIDDEN, Pose, Record};

struct Snap {
    counts: vec4<u32>,
    eye: vec4<f32>,
}

const TERRAIN_INDEX: u32 = 0x1ffu;
const NEAR_LEVEL_SHIFT: u32 = 16u;

@group(0) @binding(0) var<uniform> snap: Snap;
@group(0) @binding(1) var<storage, read_write> records: array<Record>;
@group(0) @binding(2) var<storage, read_write> poses: array<Pose>;
@group(0) @binding(3) var<storage, read> terrains: array<TerrainInfo>;
@group(0) @binding(6) var<storage, read_write> grounds: array<vec4<u32>>;

struct FinestGround {
    height: f32,
    found: bool,
    level: u32,
    tile: LevelTile,
}

fn finest_ground(info: TerrainInfo, terrain: GroundTerrain, local: vec2<f32>) -> FinestGround {
    var ground: FinestGround;
    let extent = ground_extent(terrain);
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

fn ground_color(info: TerrainInfo, terrain: GroundTerrain, local: vec2<f32>,
        ground: FinestGround) -> u32 {
    let samples = local / (info.origin.w * f32(1u << ground.level));
    let material = tile_material(ground.tile, vec2<u32>(round(samples)));
    let share = f32(material.share) / 255.0;
    let first = material_look(info.levels.z, material.first);
    let second = material_look(info.levels.z, material.second);
    let far = mix(first.far, second.far, share);
    let vegetated = mix(step(1.0e-6, first.canopy.w), step(1.0e-6, second.canopy.w), share);
    let colors = ground_colors(terrain, info.levels.w, local, 0u);
    let tint = macro_color(colors) * cover_tint(colors.a, vegetated);
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
    let ground_terrain = ground_of(info);
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
        grounds[id.x].w = ground_color(info, ground_terrain, local, ground);
    }
}
