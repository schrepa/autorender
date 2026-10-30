engine_render::shader_modules! {
    "sky::common" in "sky" = "common.wgsl";
    "sky::light" in "sky" = "light.wgsl";
    "sky::air" in "sky" = "air.wgsl";
    "sky::environment" in "sky" = "environment.wgsl";
    "sky::piece" in "sky" = "piece.wgsl";
    "face::sky" in "sky" = "faces/sky.wgsl";
    "sky::tables" in "sky" = "tables.wgsl";
    "sky::clouds" in "sky" = "clouds.wgsl";
    "sky::environment_maps" in "sky" = "environment_maps.wgsl";
    "sky::draw" in "sky" = "draw.wgsl";
    "sky::debug" in "sky" = "debug.wgsl";
    "sky::haze" in "sky" = "haze.wgsl";
    "sky::haze_march" in "sky" = "haze_march.wgsl";
    "sky::haze_filter" in "sky" = "haze_filter.wgsl";
}

engine_render::entry_points! {
    draw {
        vertex: [vs_sky @ "sky::draw"],
        fragment: [
            fs_probe_sky @ "sky::draw",
            fs_view_clouds_post @ "sky::draw",
            fs_sky_post @ "sky::draw",
            fs_sky_temporal @ "sky::draw",
            fs_debug_sky @ "sky::debug",
        ],
    }
    clouds {
        compute: [
            shadow_clouds @ "sky::clouds",
            light_clouds @ "sky::clouds",
            march_clouds @ "sky::clouds",
            march_view @ "sky::clouds",
        ],
    }
    sky {
        compute: [build_sky_view @ "sky::tables", derive_light @ "sky::tables"],
    }
    haze {
        vertex: [vs_fullscreen @ "engine::fullscreen"],
        fragment: [fs_haze @ "sky::haze_march", fs_haze_filter @ "sky::haze_filter"],
    }
    environment {
        vertex: [vs_environment @ "sky::environment_maps"],
        fragment: [
            fs_environment_sky @ "sky::environment_maps",
            fs_environment_down @ "sky::environment_maps",
            fs_environment_prefilter @ "sky::environment_maps",
        ],
    }
}
