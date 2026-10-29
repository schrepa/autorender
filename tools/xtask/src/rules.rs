use std::path::Path;
use std::sync::OnceLock;

use engine_checks::{Errors, Exceptions, Facade, Forbidden, Repository, Rules};

pub const RULES: Rules = Rules {
    exceptions: Exceptions {
        long_files: &[
            ("crates/engine_render/src/meshes/terrain.wgsl", 1643),
            ("crates/engine_render/src/water_sim.wgsl", 1645),
            ("crates/engine_terrain/src/steady_water.rs", 2360),
            ("tools/lodbuild/src/simplify.rs", 2009),
        ],
        wide_structs: &[
            ("crates/engine_core/src/frame_stats.rs", "RenderStats", 42),
            (
                "crates/engine_core/src/interpolation/gpu.rs",
                "GpuInputs",
                35,
            ),
            ("crates/engine_render/src/clouds.rs", "CloudPass", 48),
            ("crates/engine_render/src/post.rs", "Post", 32),
            ("crates/engine_render/src/scatter.rs", "ScatterPass", 40),
            ("crates/engine_render/src/terrain/atlas.rs", "TileAtlas", 32),
            (
                "crates/engine_terrain/src/runtime/residency.rs",
                "TerrainState",
                32,
            ),
        ],
        unsafe_files: &[
            "crates/engine_profile/src/allocator.rs",
            "crates/engine_profiler/src/counters.rs",
            "crates/engine_profiler/src/load.rs",
            "crates/engine_profiler/src/pix.rs",
            "crates/engine_profiler/src/renderdoc.rs",
            "crates/engine_profiler/src/sampler/linux.rs",
            "crates/engine_profiler/src/sampler/macos.rs",
            "crates/engine_profiler/src/sampler/mod.rs",
            "crates/engine_profiler/src/sampler/windows.rs",
            "crates/engine_profiler/src/stack.rs",
            "crates/engine_render/src/renderer/native.rs",
        ],
        core_uses_brick: &[
            ("engine", "engine_collision"),
            ("engine", "engine_layer_erosion"),
            ("engine", "engine_layer_heightmap"),
            ("engine", "engine_layer_hydrology"),
            ("engine", "engine_layer_noise"),
            ("engine", "engine_layer_paths"),
            ("engine", "engine_layer_relief"),
            ("engine", "engine_layer_stamps"),
            ("engine", "engine_navigation"),
            ("engine", "engine_physics"),
            ("engine", "engine_terrain"),
            ("engine", "engine_time_of_day"),
            ("engine", "engine_world_streaming"),
            ("engine_dev", "engine_collision"),
            ("engine_dev", "engine_navigation"),
            ("engine_dev", "engine_physics"),
            ("engine_dev", "engine_terrain"),
            ("engine_dev", "engine_time_of_day"),
            ("engine_dev", "engine_world_streaming"),
            ("engine_native", "engine_layer_erosion"),
            ("engine_native", "engine_layer_heightmap"),
            ("engine_native", "engine_layer_hydrology"),
            ("engine_native", "engine_layer_noise"),
            ("engine_native", "engine_layer_paths"),
            ("engine_native", "engine_layer_relief"),
            ("engine_native", "engine_layer_stamps"),
            ("engine_native", "engine_navigation"),
            ("engine_native", "engine_terrain"),
            ("engine_scene", "engine_collision"),
            ("engine_scene", "engine_navigation"),
            ("engine_scene", "engine_physics"),
            ("engine_scene", "engine_terrain"),
            ("engine_scene", "engine_world_streaming"),
        ],
        unsmoked: &["debug_panic"],
    },
    facade: Some(Facade {
        package: "engine",
        inner: "engine_core",
        internal: &[
            "CHUNK_LENGTH",
            "GroundedAt",
            "JobBudget",
            "JobFailures",
            "JobWorker",
            "LOD_RECORD",
            "LodInput",
            "MAX_DEV_TICKS",
            "RenderTables",
            "StaticTable",
            "StaticTables",
            "atmosphere_fields",
            "check_dev_ticks",
            "job_output",
            "lod_input",
            "state_atmosphere_rules",
            "table_data",
        ],
    }),
    forbidden: &[Forbidden {
        within: "crates/engine_terrain/src/runtime/",
        except: "crates/engine_terrain/src/runtime/tile_map.rs",
        code: "HashMap<TileKey",
        instead: "TileMap",
    }],
    errors: Some(Errors {
        within: "crates/",
        file: "crates/engine_settings/src/error.rs",
    }),
};

pub fn repository(root: &Path) -> Result<&'static Repository, String> {
    static REPOSITORY: OnceLock<Result<Repository, String>> = OnceLock::new();
    REPOSITORY
        .get_or_init(|| Repository::read(root))
        .as_ref()
        .map_err(Clone::clone)
}
