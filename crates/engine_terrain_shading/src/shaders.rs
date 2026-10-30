engine_render::shader_modules! {
    "face::ground" in "terrain_shading" = "faces/ground.wgsl";
    "face::heightfield" in "terrain_shading" = "faces/heightfield.wgsl";
    "face::horizons" in "terrain_shading" = "faces/horizons.wgsl";
    "terrain::field" in "terrain_shading" = "terrain/field.wgsl";
    "terrain::water_blocks" in "terrain_shading" = "terrain/water_blocks.wgsl";
    "terrain::piece" in "terrain_shading" = "terrain/piece.wgsl";
    "terrain::horizons" in "terrain_shading" = "terrain/horizons.wgsl";
    "terrain::bounce" in "terrain_shading" = "terrain/bounce.wgsl";
    "terrain::tile" in "terrain_shading" = "terrain_tile.wgsl";
    "terrain::normals" in "terrain_shading" = "terrain_normals.wgsl";
    "terrain::control" in "terrain_shading" = "terrain_control.wgsl";
    "terrain::bounce_maps" in "terrain_shading" = "terrain_bounce.wgsl";
    "terrain::shadow" in "terrain_shading" = "terrain_shadow.wgsl";
    "terrain::surface" in "terrain_shading" = "meshes/terrain_surface.wgsl";
    "terrain::draw_shared" in "terrain_shading" = "meshes/terrain_draw_shared.wgsl";
    "terrain::draw" in "terrain_shading" = "meshes/terrain.wgsl";
    "terrain::subdivision" in "terrain_shading" = "meshes/terrain_subdivision.wgsl";
    "terrain::sky" in "terrain_shading" = "meshes/terrain_sky.wgsl";
    "terrain::debug" in "terrain_shading" = "meshes/terrain_debug.wgsl";
}

engine_render::entry_points! {
    terrain {
        vertex: [
            vs_terrain @ "terrain::draw",
            vs_fullscreen @ "engine::fullscreen",
        ],
        fragment: [
            fs_terrain_ids @ "terrain::draw",
            fs_terrain_resolve_post @ "terrain::draw",
            fs_terrain_resolve_temporal @ "terrain::draw",
            fs_terrain_probe @ "terrain::draw",
            fs_debug_terrain @ "terrain::debug",
        ],
        compute: [
            subdivide_begin @ "terrain::subdivision",
            subdivide_decide @ "terrain::subdivision",
            subdivide_apply @ "terrain::subdivision",
        ],
    }
    terrain_normals {
        compute: [tile_normals @ "terrain::normals"],
    }
    terrain_control {
        compute: [tile_control @ "terrain::control"],
    }
    terrain_bounce {
        compute: [
            bounce_ground @ "terrain::bounce_maps",
            bounce_down @ "terrain::bounce_maps",
            bounce_gather @ "terrain::bounce_maps",
        ],
    }
    terrain_shadow {
        vertex: [vs_terrain_shadow @ "terrain::shadow"],
    }
}
