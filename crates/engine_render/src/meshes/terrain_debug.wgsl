// Terrain in the debug views (`meshes/debug.wgsl`, `debug_view.wgsl`): what
// its ground is made of, as shading computes it where no virtual texture
// page stands in (`meshes/terrain.wgsl`). Its record holds its leaf's
// level (the pyramid level whose spacing its leaf's meshlet vertices stand
// apart), whether it lies on a level-0 tile's edge or a patch's (sixteen
// samples) and, above DEBUG_PATCH_SHIFT, the patch it lies in within that
// tile (x in the low four bits, z in the next four); its object word the
// share of the sky it sees and its detail's shares (showing its maps'
// detail, of their slopes averaged into roughness, canopy), a byte each;
// its detail word its materials' shares, a byte each; its
// normal the terrain's smooth one. The third target holds its tile (x and
// z, signed 16 bits each) and its materials (a byte each,
// DEBUG_NO_MATERIAL for none).

// Its leaf's level, from its meshlet's vertex spacing; its level-0 tile and
// patch, from where it lies on the terrain; and its materials, sky and
// detail.
@fragment
fn fs_debug_terrain(input: TerrainOutput) -> DebugTargets {
    terrain_holes_test(input.cell);
    let relative = input.relative.xyz;
    let world = eye_world(relative);
    let seen = terrain_seen(relative, terrain_forward_look(relative));
    let shares = terrain_shares(seen.site, seen.shading);
    let ground = terrain_shared_ground(shares, seen.site);
    let local = world.xz - terrain.origin.xz;
    let sky = terrain_sky_map(local).r;
    var point = debug_point(world, seen.site.geometric);
    point.on_terrain = true;
    let spacing = input.relative.w;
    let level = u32(clamp(round(log2(spacing / terrain.anchor.w)), 0.0, 7.0));
    // The level-0 tile it lies in, and its patch within it.
    let tile = vec2<i32>(floor(input.cell / 256.0));
    let within = vec2<u32>(clamp(floor(fract(input.cell / 256.0) * 16.0), vec2<f32>(0.0),
        vec2<f32>(15.0)));
    let width = fwidth(input.cell) * 1.5; // filter: none, the view's edge lines a pixel and a half wide
    let tile_edge = abs(fract(input.cell / 256.0 + 0.5) - 0.5) * 256.0;
    let patch_edge = abs(fract(input.cell / 16.0 + 0.5) - 0.5) * 16.0;
    let record = DEBUG_KIND_TERRAIN | (level << DEBUG_LEVEL_SHIFT)
        | select(0u, DEBUG_TILE_EDGE, any(tile_edge < width * 2.0))
        | select(0u, DEBUG_PATCH_EDGE, any(patch_edge < width))
        | ((within.x | (within.y << 4u)) << DEBUG_PATCH_SHIFT);
    let made = pack4x8unorm(vec4<f32>(sky, ground.detailed, ground.averaged, ground.canopy));
    var targets = debug_targets(record, made, pack4x8unorm(shares.weights), point);
    let ids = min(shares.ids, vec4<u32>(DEBUG_NO_MATERIAL));
    let tiles = bitcast<vec2<u32>>(tile) & vec2<u32>(0xffffu);
    targets.terrain = vec2<u32>(tiles.x | (tiles.y << 16u),
        ids.x | (ids.y << 8u) | (ids.z << 16u) | (ids.w << 24u));
    return targets;
}
