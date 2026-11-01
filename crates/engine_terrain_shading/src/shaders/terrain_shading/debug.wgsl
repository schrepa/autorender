import package::face::lighting::sun_debug;
import package::engine::view::eye_world;
import package::face::debug::{DEBUG_KIND_TERRAIN, DEBUG_LEVEL_SHIFT, DEBUG_NO_MATERIAL, DebugTargets,
    debug_targets};
import package::generated::terrain_shading::{DEBUG_PATCH_EDGE, DEBUG_PATCH_SHIFT, DEBUG_TILE_EDGE};
import package::terrain_shading::draw::{TerrainOutput, terrain_forward_look, terrain_holes_test, terrain_seen, terrain_shared_ground, terrain_shares};
import package::terrain_shading::faces::heightfield::{terrain, use_terrain};
import package::terrain_shading::sky::terrain_sky_map;

@fragment
fn fs_debug_terrain(input: TerrainOutput) -> DebugTargets {
    return debug_terrain(input, false);
}

@fragment
fn fs_debug_terrain_holes(input: TerrainOutput) -> DebugTargets {
    return debug_terrain(input, true);
}

fn debug_terrain(input: TerrainOutput, holes: bool) -> DebugTargets {
    use_terrain(input.terrain);
    terrain_holes_test(input.cell, holes);
    let relative = input.relative.xyz;
    let world = eye_world(relative);
    let seen = terrain_seen(relative, terrain_forward_look(relative));
    let shares = terrain_shares(seen.site, seen.shading);
    let ground = terrain_shared_ground(shares, seen.site);
    let local = world.xz - terrain.origin.xz;
    let sky = terrain_sky_map(local).r;
    let normal = seen.site.geometric;
    let world_dx = dpdx(world);
    let world_dy = dpdy(world);
    let spacing = input.relative.w;
    let level = u32(clamp(round(log2(spacing / terrain.anchor.w)), 0.0, 7.0));
    let tile = vec2<i32>(floor(input.cell / 256.0));
    let within = vec2<u32>(clamp(floor(fract(input.cell / 256.0) * 16.0), vec2<f32>(0.0),
        vec2<f32>(15.0)));
    let width = fwidth(input.cell) * 1.5;
    let tile_edge = abs(fract(input.cell / 256.0 + 0.5) - 0.5) * 256.0;
    let patch_edge = abs(fract(input.cell / 16.0 + 0.5) - 0.5) * 16.0;
    let record = DEBUG_KIND_TERRAIN | (level << DEBUG_LEVEL_SHIFT)
        | select(0u, DEBUG_TILE_EDGE, any(tile_edge < width * 2.0))
        | select(0u, DEBUG_PATCH_EDGE, any(patch_edge < width))
        | ((within.x | (within.y << 4u)) << DEBUG_PATCH_SHIFT);
    let made = pack4x8unorm(vec4<f32>(sky, ground.detailed, ground.averaged, ground.canopy));
    var targets = debug_targets(record, made, pack4x8unorm(shares.weights), normal,
        sun_debug(world, normal, world_dx, world_dy, true));
    let ids = min(shares.ids, vec4<u32>(DEBUG_NO_MATERIAL));
    let tiles = bitcast<vec2<u32>>(tile) & vec2<u32>(0xffffu);
    targets.terrain = vec2<u32>(tiles.x | (tiles.y << 16u),
        ids.x | (ids.y << 8u) | (ids.z << 16u) | (ids.w << 24u));
    return targets;
}
