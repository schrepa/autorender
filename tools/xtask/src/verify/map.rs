use Check::{
    AllSmokes, Browser, Equivalence, Everything, ItsSmoke, Project, Pure, PureWasm, Shaders,
    Smokes, Tools,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Check {
    Everything,
    Pure(&'static str),
    PureWasm,
    Shaders,
    Smokes(&'static str),
    ItsSmoke,
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
            "tools/policy/**",
        ],
        checks: &[],
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
        paths: &["crates/engine_core/**", "crates/engine/**"],
        checks: &[
            Pure("caches"),
            Pure("scenes"),
            Equivalence,
            Smokes("plugins"),
        ],
    },
    Rule {
        paths: &[
            "crates/engine/src/**/browser*",
            "crates/engine/src/**/dev_host*",
        ],
        checks: &[Smokes("parity"), Browser],
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
        checks: &[
            Smokes("hierarchy"),
            Smokes("interpolation"),
            Smokes("presentation"),
            Smokes("culling"),
        ],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/navigation*",
            "crates/engine_core/src/**/navigation/**",
            "crates/engine_core/src/**/grounded*",
            "crates/engine_scene/src/**/terrain*",
            "crates/engine_dev/src/features/terrain/**",
            "crates/engine_dev/src/features/navigation*",
            "crates/engine_collision/src/**/terrain*",
            "crates/engine_terrain/**",
        ],
        checks: &[Pure("terrain"), Smokes("terrain"), Smokes("jobs")],
    },
    Rule {
        paths: &["crates/engine_terrain/**"],
        checks: &[PureWasm, Equivalence],
    },
    Rule {
        paths: &[
            "crates/engine_terrain/src/runtime/ground_cover*",
            "crates/engine_terrain/src/runtime/wind*",
            "crates/engine_terrain/src/runtime/occupancy*",
            "crates/engine_render/src/**/ground_cover*",
            "crates/engine_render/src/**/cover_grow*",
            "crates/engine_render/src/**/occupancy*",
            "crates/engine_render/src/**/shadow*",
            "crates/engine_render/src/**/shadows/**",
        ],
        checks: &[Smokes("surface")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/water*",
            "crates/engine_terrain/src/runtime/water*",
            "crates/engine_core/src/**/clouds*",
            "crates/engine_core/src/**/sky*",
            "crates/engine_core/src/**/time_of_day*",
            "crates/engine_core/src/**/time_of_day/**",
            "crates/engine_core/src/**/atmosphere*",
            "crates/engine_render/src/**/clouds*",
            "crates/engine_render/src/**/sky*",
            "crates/engine_render/src/**/water*",
            "crates/engine_render/src/**/environment*",
            "crates/engine_render/src/**/terrain_sky*",
            "crates/engine_render/src/**/terrain_horizon*",
            "crates/engine_core/src/**/tables*",
            "crates/engine_tables/**",
        ],
        checks: &[Smokes("sky")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/*water*",
            "crates/engine_render/src/**/*water*",
            "crates/engine_terrain/src/**/*water*",
        ],
        checks: &[Smokes("torrent")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/gpu_culling*",
            "crates/engine_render/src/**/depth_pyramid*",
            "crates/engine_render/src/**/scatter*",
        ],
        checks: &[Smokes("culling")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/level_of_detail*",
            "crates/engine_render/src/**/impostor*",
            "crates/engine_render/src/**/impostors/**",
            "crates/engine_render/src/**/meshes/**",
            "crates/engine_render/src/**/mesh_arena*",
            "crates/engine_assets/src/**/lod*",
            "crates/engine_core/src/**/mesh_lods*",
            "tools/lodbuild/**",
        ],
        checks: &[Smokes("lod"), Pure("simplify")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/jobs*",
            "crates/engine_worker/**",
            "crates/engine_sources/**",
        ],
        checks: &[Smokes("jobs"), PureWasm, Pure("terrain")],
    },
    Rule {
        paths: &[
            "crates/engine_sources/**",
            "crates/engine_core/src/**/records*",
        ],
        checks: &[Pure("terrain"), Pure("caches"), Smokes("jobs")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/tables*",
            "crates/engine_tables/**",
        ],
        checks: &[Pure("tables"), PureWasm, Smokes("jobs")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/spline_mesh*",
            "crates/engine_scene/src/**/spline*",
        ],
        checks: &[Smokes("spline")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/streaming*",
            "crates/engine_core/src/**/streaming/**",
            "crates/engine_terrain/src/runtime/streaming*",
        ],
        checks: &[Smokes("streaming")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/debug_view*",
            "crates/engine_render/src/**/debug_view*",
            "crates/engine_render/src/**/meshes/debug*",
            "crates/engine_render/src/**/renderer/probe*",
        ],
        checks: &[Smokes("debug"), Shaders],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/effects*",
            "crates/engine_core/src/**/antialiasing*",
            "crates/engine_render/src/**/temporal*",
            "crates/engine_render/src/**/post*",
            "crates/engine_render/src/**/haze*",
        ],
        checks: &[Smokes("taa")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/foliage*",
            "crates/engine_render/src/meshes/cover*",
            "crates/engine_render/src/meshes/impostors*",
        ],
        checks: &[Smokes("taa")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/probes*",
            "crates/engine_render/src/**/probes*",
            "crates/engine_render/src/**/exposure*",
        ],
        checks: &[Smokes("sky")],
    },
    Rule {
        paths: &["crates/engine_render/**"],
        checks: &[Shaders, Equivalence, Browser],
    },
    Rule {
        paths: &[
            "crates/engine_render/src/**/terrain*",
            "crates/engine_render/src/**/terrain*/**",
            "crates/engine_render/src/**/bounce*",
            "crates/engine_render/src/**/scatter*",
        ],
        checks: &[Smokes("terrain")],
    },
    Rule {
        paths: &[
            "crates/engine_render/src/**/gpu_presentation/**",
            "crates/engine_render/src/**/presentation*",
        ],
        checks: &[Smokes("presentation"), Smokes("interpolation")],
    },
    Rule {
        paths: &[
            "crates/engine_render/src/**/materials*",
            "crates/engine_render/src/**/mipmaps*",
            "crates/engine_assets/src/**/texture*",
            "crates/engine_assets/src/**/ktx2*",
        ],
        checks: &[Smokes("texture"), Smokes("surfaces")],
    },
    Rule {
        paths: &[
            "crates/engine_textures/**",
            "crates/engine_core/src/records*",
        ],
        checks: &[
            Pure("textures"),
            PureWasm,
            Smokes("texture"),
            Smokes("jobs"),
        ],
    },
    Rule {
        paths: &["crates/engine_assets/**"],
        checks: &[
            Pure("gltf"),
            Pure("simplify"),
            Smokes("meshes"),
            Smokes("texture"),
            Equivalence,
        ],
    },
    Rule {
        paths: &[
            "crates/engine_collision/**",
            "crates/engine_dev/src/features/collision/**",
        ],
        checks: &[
            Smokes("collision"),
            Smokes("terrain"),
            Pure("caches"),
            Equivalence,
        ],
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
        checks: &[
            Smokes("physics"),
            Smokes("plugins"),
            Smokes("sky"),
            Pure("caches"),
            Equivalence,
        ],
    },
    Rule {
        paths: &["crates/engine_scene/**", "crates/engine_settings/**"],
        checks: &[
            Pure("scenes"),
            Pure("requests"),
            Pure("gltf"),
            Smokes("agent"),
            Smokes("plugins"),
            Equivalence,
        ],
    },
    Rule {
        paths: &["crates/engine_dev/**"],
        checks: &[
            Pure("requests"),
            Smokes("agent"),
            Smokes("native"),
            Smokes("parity"),
            Equivalence,
        ],
    },
    Rule {
        paths: &["crates/engine_native/**", "crates/engine_panic/**"],
        checks: &[
            Smokes("native"),
            Smokes("parity"),
            Smokes("agent"),
            Equivalence,
        ],
    },
    Rule {
        paths: &[
            "crates/engine_native/src/**/measure*",
            "crates/engine_profile/**",
            "crates/engine_profiler/**",
        ],
        checks: &[Smokes("native"), Tools("measure")],
    },
    Rule {
        paths: &["crates/engine_player/**"],
        checks: &[AllSmokes, Browser],
    },
    Rule {
        paths: &["tools/assetc/**"],
        checks: &[Smokes("files"), Pure("import")],
    },
    Rule {
        paths: &["tools/coresmoke/**"],
        checks: &[Pure("caches")],
    },
    Rule {
        paths: &[
            "tools/xtask/Cargo.toml",
            "tools/xtask/src/*.rs",
            "tools/xtask/src/gpu/**",
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
            "tools/xtask/src/formats.rs",
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
        checks: &[Smokes("lifecycle")],
    },
    Rule {
        paths: &["tools/xtask/src/importing.rs"],
        checks: &[Pure("gltf")],
    },
    Rule {
        paths: &["web/scripts/*-smoke.mjs"],
        checks: &[ItsSmoke],
    },
    Rule {
        paths: &[
            "web/scripts/harness.mjs",
            "web/scripts/browser-launch.mjs",
            "web/scripts/heavy-run.mjs",
            "web/scripts/workspace.mjs",
            "web/scripts/exit-codes.mjs",
            "web/scripts/exercised.mjs",
        ],
        checks: &[AllSmokes],
    },
    Rule {
        paths: &[
            "web/scripts/terrain-fixture.mjs",
            "web/scripts/fixtures/terrain*",
            "web/scripts/fixtures/terrain-smoke/**",
        ],
        checks: &[Smokes("terrain"), Pure("terrain"), Pure("import")],
    },
    Rule {
        paths: &["web/scripts/fixtures/**"],
        checks: &[AllSmokes, Pure("scenes")],
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
        checks: &[Smokes("agent"), Smokes("parity"), Smokes("native")],
    },
    Rule {
        paths: &["web/scripts/pure-wasm.mjs"],
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
        checks: &[
            Smokes("parity"),
            Smokes("presentation"),
            Smokes("agent"),
            Browser,
        ],
    },
    Rule {
        paths: &[
            "crates/engine/src/host/frame_loop*",
            "crates/engine_native/src/window*",
            "web/src/engine/host.ts",
        ],
        checks: &[Smokes("loop"), Smokes("streaming"), Smokes("terrain")],
    },
    Rule {
        paths: &[
            "crates/engine/src/host/input*",
            "crates/engine/src/game_host*",
            "crates/engine_native/src/input*",
            "web/src/engine/input*",
        ],
        checks: &[Smokes("input"), Smokes("interpolation")],
    },
    Rule {
        paths: &["web/src/engine/job*"],
        checks: &[Smokes("jobs")],
    },
    Rule {
        paths: &["web/src/engine/ktx2*", "web/src/engine/decode*"],
        checks: &[Smokes("texture")],
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

pub(crate) fn covers(topic: &str, smoke: &str) -> bool {
    smoke.split('-').any(|word| word == topic)
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
