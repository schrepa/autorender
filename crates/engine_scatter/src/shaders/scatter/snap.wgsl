import package::face::cover::{CoverSite, cover_ground, cover_tint};
import package::face::ground_tiles::{
    GroundTerrain, LevelTile, TerrainInfo, ground_colors, ground_extent, ground_of, ground_tile,
    macro_color, material_look, terrain_info, tile_height, tile_material, true_height
};
import package::generated::scatter::{
    DEAD, PlacedValues, SCATTER_GROUP, Snap, scatter_terrain_index, scatter_terrain_near,
    scatter_terrain_with_hidden
};
import package::place::placed_tiles::{Pose, Record};

struct SnapList {
    head: Snap,
    items: array<vec4<u32>>,
}

@group(1) @binding(0) var<storage, read> snap: SnapList;
@group(1) @binding(1) var<storage, read_write> records: array<Record>;
@group(1) @binding(2) var<storage, read_write> poses: array<Pose>;
@group(1) @binding(3) var<storage, read_write> values: array<PlacedValues>;

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
    let finest = ground_tile(terrain, local, 0u);
    if !finest.found {
        return ground;
    }
    let tile = finest.tile;
    let samples = local / (info.origin.w * f32(1u << finest.level));
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
    ground.level = finest.level;
    ground.tile = tile;
    return ground;
}

fn snap_vegetated(index: u32, material: u32, color: vec3<f32>, world: vec3<f32>, lush: f32,
        footprint: f32) -> f32 {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    return cover_ground(CoverSite(index, material, color, world, lush, up, up, 0.0, footprint))
        .vegetated;
}

fn ground_color(index: u32, info: TerrainInfo, terrain: GroundTerrain, local: vec2<f32>,
        ground: FinestGround) -> u32 {
    let samples = local / (info.origin.w * f32(1u << ground.level));
    let material = tile_material(ground.tile, vec2<u32>(round(samples)));
    let share = f32(material.share) / 255.0;
    let first = material_look(info.levels.z, material.first);
    let second = material_look(info.levels.z, material.second);
    let far = mix(first.far, second.far, share);
    let colors = ground_colors(terrain, info.levels.w, local, 0u);
    let world = info.origin.xyz + vec3<f32>(local.x, 0.0, local.y);
    var low = 0.0;
    var high = 0.0;
    for (var part = 0u; part < 2u; part++) {
        let latter = part == 1u;
        let shown = snap_vegetated(index, select(material.first, material.second, latter),
            select(first.far.rgb, second.far.rgb, latter), world, colors.a, info.origin.w);
        if latter {
            high = shown;
        } else {
            low = shown;
        }
    }
    let vegetated = mix(low, high, share);
    let tint = macro_color(colors) * cover_tint(colors.a, vegetated);
    return pack4x8unorm(vec4<f32>(sqrt(saturate(far.rgb * tint)),
        max(saturate(far.w), 1.0 / 255.0)));
}

fn snapped_slot(thread: u32) -> vec2<u32> {
    var low = 0u;
    var high = snap.head.counts.z;
    while high - low > 1u {
        let middle = (low + high) / 2u;
        if snap.items[middle].y <= thread {
            low = middle;
        } else {
            high = middle;
        }
    }
    let span = snap.items[low];
    return vec2<u32>(span.x + (thread - span.y), span.z);
}

fn ground_moved(at: vec2<f32>) -> bool {
    let first = snap.head.counts.z;
    for (var index = 0u; index < snap.head.counts.w; index++) {
        let rect = bitcast<vec4<f32>>(snap.items[first + index]);
        if all(at >= rect.xy) && all(at <= rect.zw) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(SCATTER_GROUP)
fn snap_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= snap.head.counts.x {
        return;
    }
    let found = snapped_slot(id.x);
    let slot = found.x;
    if found.y != 0u && !ground_moved(vec2<f32>(poses[slot].r0.w, poses[slot].r2.w)) {
        return;
    }
    let record = records[slot];
    let terrain = scatter_terrain_index(record.terrain);
    if record.group == DEAD || terrain >= snap.head.counts.y {
        return;
    }
    let info = terrain_info(terrain);
    let ground_terrain = ground_of(info);
    let pose = poses[slot];
    let local = vec2<f32>(pose.r0.w, pose.r2.w) - info.origin.xz;
    let ground = finest_ground(info, ground_terrain, local);
    let near = scatter_terrain_near(record.terrain);
    let hidden = near != 0u && ground.found && ground.level < near;
    records[slot].terrain = scatter_terrain_with_hidden(record.terrain, hidden);
    if !ground.found {
        return;
    }
    let base = info.origin.y
        + true_height(ground_terrain, local, ground.level, ground.tile) - record.sink;
    poses[slot].r1.w = base;
    records[slot].center.y = record.center.y + (base - pose.r1.w);
    let held = values[slot];
    if held.reach != 0.0 {
        let tilt = unpack2x16snorm(held.tilt);
        let up = sqrt(max(1.0 - dot(tilt, tilt), 0.01));
        values[slot].ground = base + record.sink
            + dot(tilt, vec2<f32>(pose.r0.w, pose.r2.w)) / up;
        values[slot].texel = ground_color(terrain, info, ground_terrain, local, ground);
    }
}
