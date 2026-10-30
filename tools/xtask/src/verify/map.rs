use Check::{
    AllSmokes, Browser, CoreSmokes, Equivalence, Everything, Project, Pure, PureWasm, Shaders,
    Tools,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Check {
    Everything,
    Pure(&'static str),
    PureWasm,
    Shaders,
    CoreSmokes,
    AllSmokes,
    Project,
    Equivalence,
    Browser,
    Tools(&'static str),
}

pub(crate) struct Rule {
    pub paths: &'static [&'static str],
    pub checks: &'static [Check],
}

pub(crate) const MAP: &[Rule] = &[
    Rule {
        paths: &[
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config.toml",
            "vendor/**",
        ],
        checks: &[Everything],
    },
    Rule {
        paths: &[
            ".gitignore",
            "clippy.toml",
            "scripts/*.ps1",
            "faces.json",
            "crates/engine_checks/**",
        ],
        checks: &[],
    },
    Rule {
        paths: &[
            "crates/engine_checks/src/gpu.rs",
            "crates/engine_checks/src/gpu/**",
        ],
        checks: &[Shaders],
    },
    Rule {
        paths: &["crates/engine_checks/src/bench.rs"],
        checks: &[Equivalence, Tools("measure")],
    },
    Rule {
        paths: &["**/*.wgsl"],
        checks: &[Shaders, Equivalence, Browser],
    },
    Rule {
        paths: &["bench/equivalence.json"],
        checks: &[Equivalence],
    },
    Rule {
        paths: &["bricks.json"],
        checks: &[CoreSmokes],
    },
    Rule {
        paths: &[
            "bench/scenes/**",
            "bench/settings/**",
            "bench/metrics.json",
            "bench/shader-paths.json",
        ],
        checks: &[Equivalence, Tools("measure")],
    },
    Rule {
        paths: &["bench/ladders/**"],
        checks: &[],
    },
    Rule {
        paths: &["crates/engine_math/**"],
        checks: &[Everything],
    },
    Rule {
        paths: &[
            "crates/engine_core/**",
            "crates/engine/**",
            "crates/engine_world_streaming/**",
            "crates/engine_time_of_day/**",
        ],
        checks: &[CoreSmokes, Pure("caches"), Pure("scenes"), Equivalence],
    },
    Rule {
        paths: &[
            "crates/engine/src/**/browser*",
            "crates/engine/src/**/dev_host*",
        ],
        checks: &[CoreSmokes, Browser],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/hierarchy*",
            "crates/engine_core/src/**/hierarchy/**",
            "crates/engine_core/src/**/interpolation*",
            "crates/engine_core/src/**/interpolation/**",
            "crates/engine_core/src/**/retained_render*",
            "crates/engine_core/src/**/transform_edits*",
            "crates/engine_core/src/**/presentation_bounds*",
            "crates/engine_core/src/**/entity_indices*",
            "crates/engine_core/src/**/extraction*",
        ],
        checks: &[CoreSmokes],
    },
    Rule {
        paths: &[
            "crates/engine_navigation/**",
            "crates/engine_core/src/**/grounded*",
            "crates/engine_scene/src/**/terrain*",
            "crates/engine_dev/src/inspect/terrain*",
            "crates/engine_collision/src/**/terrain*",
            "crates/engine_terrain/**",
            "crates/engine_layer_*/**",
        ],
        checks: &[Pure("terrain")],
    },
    Rule {
        paths: &[
            "crates/engine_terrain/**",
            "crates/engine_layer_*/**",
            "crates/engine_navigation/**",
        ],
        checks: &[PureWasm, Equivalence],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/level_of_detail*",
            "crates/engine_render/src/**/impostor*",
            "crates/engine_render/src/**/impostors/**",
            "crates/engine_render/src/**/meshes/**",
            "crates/engine_render/src/**/mesh_arena*",
            "crates/engine_meshes/**",
            "crates/engine_scatter/src/**/impostor*",
            "crates/engine_scatter/src/**/impostors/**",
            "crates/engine_assets/src/**/lod*",
            "crates/engine_core/src/**/mesh_lods*",
            "tools/lodbuild/**",
        ],
        checks: &[Pure("simplify")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/jobs*",
            "crates/engine_worker/**",
            "crates/engine_sources/**",
        ],
        checks: &[CoreSmokes, PureWasm, Pure("terrain")],
    },
    Rule {
        paths: &[
            "crates/engine_sources/**",
            "crates/engine_core/src/**/records*",
        ],
        checks: &[CoreSmokes, Pure("terrain"), Pure("caches")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/tables*",
            "crates/engine_tables/**",
        ],
        checks: &[CoreSmokes, Pure("tables"), PureWasm],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/debug_view*",
            "crates/engine_render/src/**/debug_view*",
            "crates/engine_render/src/**/meshes/debug*",
            "crates/engine_render/src/**/renderer/probe*",
            "crates/engine_debug_view/**",
            "crates/engine_probes/**",
        ],
        checks: &[Shaders],
    },
    Rule {
        paths: &["crates/engine_render/**"],
        checks: &[Shaders, Equivalence, Browser, AllSmokes],
    },
    Rule {
        paths: &[
            "crates/engine_faces/**",
            "crates/engine_sun/**",
            "crates/engine_lighting/**",
            "crates/engine_meshes/**",
            "crates/engine_terrain_shading/**",
            "crates/engine_scatter/**",
            "crates/engine_shadows/**",
            "crates/engine_lights/**",
            "crates/engine_sky/**",
            "crates/engine_probes/**",
            "crates/engine_ground_cover/**",
            "crates/engine_water/**",
            "crates/engine_ambient_occlusion/**",
            "crates/engine_reflections/**",
            "crates/engine_bloom/**",
            "crates/engine_dither/**",
            "crates/engine_particles/**",
            "crates/engine_taa/**",
            "crates/engine_auto_exposure/**",
            "crates/engine_tonemap/**",
            "crates/engine_debug_view/**",
        ],
        checks: &[Shaders, Equivalence, Browser, AllSmokes],
    },
    Rule {
        paths: &[
            "crates/engine_textures/**",
            "crates/engine_core/src/records*",
        ],
        checks: &[CoreSmokes, Pure("textures"), PureWasm],
    },
    Rule {
        paths: &["crates/engine_assets/**"],
        checks: &[CoreSmokes, Pure("gltf"), Pure("simplify"), Equivalence],
    },
    Rule {
        paths: &[
            "crates/engine_collision/**",
            "crates/engine_dev/src/features/collision/**",
        ],
        checks: &[Pure("caches"), Equivalence],
    },
    Rule {
        paths: &[
            "crates/engine_physics/**",
            "crates/engine_dev/src/features/physics*",
            "crates/engine_core/src/physics*",
            "crates/engine_core/src/water*",
            "crates/engine_terrain/src/runtime/flow*",
            "crates/engine_terrain/src/flow*",
        ],
        checks: &[Pure("caches"), Equivalence],
    },
    Rule {
        paths: &["crates/engine_scene/**", "crates/engine_settings/**"],
        checks: &[
            CoreSmokes,
            Pure("scenes"),
            Pure("requests"),
            Pure("gltf"),
            Equivalence,
        ],
    },
    Rule {
        paths: &[
            "crates/engine_collision/src/**/scene*",
            "crates/engine_physics/src/**/scene*",
            "crates/engine_navigation/src/**/scene*",
        ],
        checks: &[Pure("scenes"), Pure("requests")],
    },
    Rule {
        paths: &["crates/engine_dev/**"],
        checks: &[CoreSmokes, Pure("requests"), Equivalence],
    },
    Rule {
        paths: &["crates/engine_native/**", "crates/engine_panic/**"],
        checks: &[CoreSmokes, Equivalence],
    },
    Rule {
        paths: &[
            "crates/engine_native/src/**/measure*",
            "crates/engine_profile/**",
            "crates/engine_profiler/**",
        ],
        checks: &[CoreSmokes, Tools("measure")],
    },
    Rule {
        paths: &["crates/engine_player/**"],
        checks: &[AllSmokes, Browser],
    },
    Rule {
        paths: &["tools/assetc/**"],
        checks: &[Pure("import")],
    },
    Rule {
        paths: &["tools/coresmoke/**"],
        checks: &[Pure("caches")],
    },
    Rule {
        paths: &[
            "tools/xtask/Cargo.toml",
            "tools/xtask/src/*.rs",
            "tools/xtask/src/verify/**",
        ],
        checks: &[],
    },
    Rule {
        paths: &["tools/xtask/src/pure.rs", "tools/xtask/pure/**"],
        checks: &[
            Pure("terrain"),
            Pure("scenes"),
            Pure("requests"),
            Pure("simplify"),
            Pure("gltf"),
            Pure("import"),
            Pure("caches"),
            Pure("textures"),
            Pure("tables"),
            PureWasm,
        ],
    },
    Rule {
        paths: &["tools/xtask/src/pure/terrain.rs"],
        checks: &[Pure("terrain"), PureWasm],
    },
    Rule {
        paths: &["tools/xtask/src/pure/tables.rs"],
        checks: &[Pure("tables"), PureWasm],
    },
    Rule {
        paths: &["tools/xtask/src/pure/scenes.rs"],
        checks: &[Pure("scenes"), Pure("requests")],
    },
    Rule {
        paths: &["tools/xtask/src/pure/requests.rs"],
        checks: &[Pure("requests")],
    },
    Rule {
        paths: &["tools/xtask/src/pure/simplify.rs"],
        checks: &[Pure("simplify")],
    },
    Rule {
        paths: &["tools/xtask/src/pure/gltf.rs"],
        checks: &[Pure("gltf")],
    },
    Rule {
        paths: &["tools/xtask/src/pure/import.rs"],
        checks: &[Pure("import")],
    },
    Rule {
        paths: &["tools/xtask/src/pure/caches.rs"],
        checks: &[Pure("caches")],
    },
    Rule {
        paths: &["tools/xtask/src/pure/textures.rs"],
        checks: &[Pure("textures"), PureWasm],
    },
    Rule {
        paths: &["tools/xtask/src/equivalence/**", "tools/xtask/src/bench.rs"],
        checks: &[Equivalence],
    },
    Rule {
        paths: &[
            "tools/xtask/src/equivalence/browser.rs",
            "web/scripts/equivalence-browser.mjs",
        ],
        checks: &[Browser],
    },
    Rule {
        paths: &[
            "tools/xtask/src/measure.rs",
            "crates/engine_checks/src/formats.rs",
            "tools/xtask/src/shader_report.rs",
            "tools/xtask/src/bench.rs",
            "tools/xtask/src/selfcheck.rs",
            "tools/xtask/src/checkouts.rs",
        ],
        checks: &[Tools("measure")],
    },
    Rule {
        paths: &[
            "tools/xtask/src/checks.rs",
            "tools/xtask/src/serve.rs",
            "tools/xtask/src/selfcheck.rs",
            "tools/xtask/src/checkouts.rs",
        ],
        checks: &[Tools("checks")],
    },
    Rule {
        paths: &[
            "tools/xtask/src/smoke.rs",
            "tools/xtask/src/serve.rs",
            "tools/xtask/src/heavy.rs",
            "tools/xtask/src/pipeline.rs",
        ],
        checks: &[CoreSmokes],
    },
    Rule {
        paths: &["tools/xtask/src/importing.rs"],
        checks: &[Pure("gltf")],
    },
    Rule {
        paths: &[
            "web/scripts/browser-launch.mjs",
            "web/scripts/smoke-browser.mjs",
        ],
        checks: &[AllSmokes],
    },
    Rule {
        paths: &[
            "web/scripts/heavy-run.mjs",
            "web/scripts/workspace.mjs",
            "web/scripts/exit-codes.mjs",
            "web/scripts/exercised.mjs",
        ],
        checks: &[CoreSmokes],
    },
    Rule {
        paths: &[
            "web/scripts/fixtures/terrain*",
            "web/scripts/fixtures/terrain-smoke/**",
        ],
        checks: &[Pure("terrain"), Pure("import")],
    },
    Rule {
        paths: &["web/scripts/fixtures/**"],
        checks: &[CoreSmokes, Pure("scenes")],
    },
    Rule {
        paths: &[
            "web/scripts/agent.mjs",
            "web/scripts/agent-protocol.mjs",
            "web/scripts/agent-browser.mjs",
            "web/scripts/agent-native.mjs",
            "web/scripts/agent-ab.mjs",
            "web/scripts/agent-profile.mjs",
        ],
        checks: &[CoreSmokes],
    },
    Rule {
        paths: &["web/scripts/pure-wasm.mjs", "web/scripts/pack-wasm.mjs"],
        checks: &[PureWasm],
    },
    Rule {
        paths: &[
            "web/scripts/check-*.mjs",
            "web/scripts/iteration*.mjs",
            "web/scripts/tsconfig.json",
            "web/scripts/*.d.ts",
            "web/src/generated/**",
            "web/public/favicon.svg",
        ],
        checks: &[],
    },
    Rule {
        paths: &["web/src/**", "web/*.html"],
        checks: &[CoreSmokes, Browser],
    },
    Rule {
        paths: &[
            "crates/engine/src/host/frame_loop*",
            "crates/engine_native/src/window*",
            "web/src/engine/host.ts",
        ],
        checks: &[CoreSmokes],
    },
    Rule {
        paths: &[
            "crates/engine/src/host/input*",
            "crates/engine/src/game_host*",
            "crates/engine_native/src/input*",
            "web/src/engine/input*",
        ],
        checks: &[CoreSmokes],
    },
    Rule {
        paths: &["web/src/engine/job*"],
        checks: &[CoreSmokes],
    },
    Rule {
        paths: &["web/src/engine/ktx2*", "web/src/engine/decode*"],
        checks: &[CoreSmokes],
    },
    Rule {
        paths: &[
            "web/vite.config.ts",
            "web/package.json",
            "web/package-lock.json",
            "web/tsconfig.json",
        ],
        checks: &[AllSmokes, Browser],
    },
];

pub(crate) fn matches(pattern: &str, path: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('/').collect();
    let path: Vec<&str> = path.split('/').collect();
    segments(&pattern, &path)
}

fn segments(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((&"**", rest)) => (0..=path.len()).any(|skip| segments(rest, &path[skip..])),
        Some((first, rest)) => path
            .split_first()
            .is_some_and(|(segment, tail)| segment_matches(first, segment) && segments(rest, tail)),
    }
}

fn segment_matches(pattern: &str, text: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == text,
        Some((prefix, rest)) => {
            text.starts_with(prefix)
                && text[prefix.len()..]
                    .char_indices()
                    .map(|(at, _)| at)
                    .chain([text.len() - prefix.len()])
                    .any(|at| segment_matches(rest, &text[prefix.len() + at..]))
        }
    }
}

pub(crate) fn project_of(path: &str) -> Option<&'static str> {
    crate::process::clients()
        .iter()
        .filter(|client| client.game.is_some())
        .map(|client| client.folder.as_str())
        .filter(|folder| {
            path.strip_prefix(folder)
                .is_some_and(|rest| rest.starts_with('/'))
        })
        .max_by_key(|folder| folder.len())
}

pub(crate) fn checks_of(path: &str) -> Option<Vec<Check>> {
    let mut found = None;
    for rule in MAP {
        if rule.paths.iter().any(|pattern| matches(pattern, path)) {
            found
                .get_or_insert_with(Vec::new)
                .extend_from_slice(rule.checks);
        }
    }
    if project_of(path).is_some() {
        found.get_or_insert_with(Vec::new).push(Project);
    }
    found
}
