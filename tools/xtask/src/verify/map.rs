//! The path map: which checks a change of each tracked path calls for.
//! Every tracked file matches some rule (one that calls for no check says
//! so with an empty list), and every smoke check and equivalence cell is
//! called for by some rule, so lint (`cargo xtask lint`'s path map stage)
//! fails on a file or a check nobody placed.

use Check::{
    AllSmokes, Browser, Cells, Equivalence, Everything, ItsSmoke, Pure, PureWasm, Shaders, Smoke,
    Tools,
};

/// What a change calls for, beyond what its level always runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Check {
    /// Everything the level runs: the toolchain, the workspace or a
    /// dependency changed.
    Everything,
    /// A corpus of `cargo xtask pure`.
    Pure(&'static str),
    /// The terrain corpus's jobs replayed in the browser worker's Wasm
    /// (`cargo xtask pure --wasm`), at the merge level.
    PureWasm,
    /// The shaders against their Rust layouts (`cargo xtask gpu`).
    Shaders,
    /// A smoke check, by name.
    Smoke(&'static str),
    /// The smoke check the changed script is (`web/scripts/NAME-smoke.mjs`).
    ItsSmoke,
    /// Every smoke check: what they share changed.
    AllSmokes,
    /// Equivalence cells by name, beyond the smoke tier.
    Cells(&'static [&'static str]),
    /// Equivalence's smoke tier, natively.
    Equivalence,
    /// Equivalence's browser leg, at the merge level.
    Browser,
    /// A self-check of the factory's tools (`cargo xtask selfcheck KIND`):
    /// `measure` or `checks`.
    Tools(&'static str),
}

/// One rule: the paths it covers, as globs (`*` within a segment, `**`
/// across any number of them), and what their changes call for.
pub(crate) struct Rule {
    pub paths: &'static [&'static str],
    pub checks: &'static [Check],
}

/// The terrain smokes, which a terrain change calls for.
const TERRAIN_SMOKES: [Check; 4] = [
    Smoke("terrain"),
    Smoke("terrain-surface"),
    Smoke("terrain-water-sky"),
    Smoke("terrain-files"),
];

/// Every rule; a path's checks are those of every rule it matches.
pub(crate) const MAP: &[Rule] = &[
    // What every build reads: the toolchain, the workspace, its
    // dependencies and their vendored sources.
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
    // Repository settings and scripts lint alone checks, or nothing does.
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
        paths: &["crates/engine_render/src/**", "web/scripts/linear-light-*"],
        checks: &[Smoke("linear-light-transparency")],
    },
    // Shaders, wherever they are.
    Rule {
        paths: &["**/*.wgsl"],
        checks: &[Shaders, Equivalence, Browser],
    },
    // bench/: the matrix, and the workloads, settings and catalogs measuring
    // and equivalence read.
    Rule {
        paths: &["bench/equivalence.json"],
        checks: &[Equivalence],
    },
    Rule {
        paths: &[
            "bench/scenes/**",
            "bench/settings/**",
            "bench/options.json",
            "bench/metrics.json",
            "bench/shader-paths.json",
        ],
        checks: &[Equivalence, Tools("measure")],
    },
    // The math everything computes with.
    Rule {
        paths: &["crates/engine_math/**"],
        checks: &[Everything],
    },
    // The runtime: its caches and scenes, the smoke tier, and what each part
    // draws or does.
    Rule {
        paths: &["crates/engine_core/**", "crates/engine/**"],
        checks: &[
            Pure("caches"),
            Pure("scenes"),
            Equivalence,
            Smoke("plugins"),
        ],
    },
    // Native range reads, which terrain files stream by.
    Rule {
        paths: &["crates/engine/src/range_reads*"],
        checks: &[Smoke("terrain-files"), Smoke("native")],
    },
    Rule {
        paths: &[
            "crates/engine/src/**/browser*",
            "crates/engine/src/**/dev_host*",
        ],
        checks: &[Smoke("host-parity"), Browser],
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
            Smoke("hierarchy"),
            Smoke("interpolation"),
            Smoke("gpu-presentation"),
            Smoke("culling"),
        ],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/terrain*",
            "crates/engine_core/src/**/terrain/**",
            "crates/engine_core/src/**/bridges*",
            "crates/engine_core/src/**/navigation*",
            "crates/engine_core/src/**/navigation/**",
            "crates/engine_core/src/**/grounded*",
            "crates/engine_scene/src/**/terrain*",
            "crates/engine_dev/src/features/terrain/**",
            "crates/engine_dev/src/features/navigation*",
            "crates/engine_dev/src/features/deformation*",
            "crates/engine_collision/src/**/terrain*",
            "crates/engine_terrain/**",
        ],
        checks: &[
            Pure("terrain"),
            TERRAIN_SMOKES[0],
            TERRAIN_SMOKES[1],
            TERRAIN_SMOKES[2],
            TERRAIN_SMOKES[3],
            Smoke("jobs"),
            Cells(&["highlands", "highlands-views", "terrain-frames"]),
        ],
    },
    Rule {
        paths: &["crates/engine_terrain/**"],
        checks: &[PureWasm, Equivalence],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/ground_cover*",
            "crates/engine_core/src/**/wind*",
            "crates/engine_core/src/**/occupancy*",
            "crates/engine_core/src/**/deformation*",
            "crates/engine_core/src/**/deformation/**",
            "crates/engine_core/src/**/contact_shadows*",
            "crates/engine_render/src/**/ground_cover*",
            "crates/engine_render/src/**/cover_grow*",
            "crates/engine_render/src/**/occupancy*",
            "crates/engine_render/src/**/deformation*",
            "crates/engine_render/src/**/shadow*",
            "crates/engine_render/src/**/shadows/**",
        ],
        checks: &[
            Smoke("terrain-surface"),
            Cells(&[
                "highlands",
                "metro-shadows-off",
                "metro-contact-shadows-off",
            ]),
        ],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/water*",
            "crates/engine_core/src/**/clouds*",
            "crates/engine_core/src/**/sky*",
            "crates/engine_core/src/**/time_of_day*",
            "crates/engine_core/src/**/time_of_day/**",
            "crates/engine_core/src/**/atmosphere*",
            "crates/engine_core/src/**/ocean/**",
            "crates/engine_dev/src/features/ocean*",
            "crates/engine_render/src/**/clouds*",
            "crates/engine_render/src/**/sky*",
            "crates/engine_render/src/**/water*",
            "crates/engine_render/src/**/environment*",
            "crates/engine_render/src/**/terrain_sky*",
            "crates/engine_render/src/**/terrain_horizon*",
        ],
        checks: &[
            Smoke("terrain-water-sky"),
            Cells(&[
                "garden-clouds",
                "garden-sky",
                "garden-time-of-day",
                "metro-sky",
                "metro-golden",
                "metro-dusk",
                "metro-night",
                "metro-day",
            ]),
        ],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/culling*",
            "crates/engine_core/src/**/gpu_culling*",
            "crates/engine_render/src/**/visibility*",
            "crates/engine_render/src/**/depth_pyramid*",
            "crates/engine_render/src/**/scatter*",
        ],
        checks: &[
            Smoke("culling"),
            Cells(&[
                "dense-visibility",
                "metro-occlusion-culling-off",
                "metro-subpixel-culling-off",
            ]),
        ],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/level_of_detail*",
            "crates/engine_render/src/**/impostor*",
            "crates/engine_render/src/**/meshes/**",
            "crates/engine_render/src/**/mesh_arena*",
            "crates/engine_assets/src/**/simplify*",
            "crates/engine_assets/src/**/lod*",
        ],
        checks: &[
            Smoke("lod"),
            Pure("simplify"),
            Cells(&["metro-lod-off", "metro-impostors-off"]),
        ],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/virtual_geometry*",
            "crates/engine_render/src/**/virtual_geometry*",
            "crates/engine_render/src/**/virtual_geometry/**",
            "crates/engine_render/src/**/virtual_streaming*",
            "crates/engine_assets/src/**/virtual_mesh*",
            "web/src/engine/virtual-mesh*",
        ],
        checks: &[Smoke("virtual-geometry")],
    },
    // The dense workload's virtual meshes come from vgbuild.
    Rule {
        paths: &["tools/vgbuild/**"],
        checks: &[
            Smoke("virtual-geometry"),
            Cells(&["dense-views", "dense-visibility"]),
            Equivalence,
        ],
    },
    Rule {
        paths: &["crates/engine_core/src/**/jobs*", "crates/engine_worker/**"],
        checks: &[Smoke("jobs"), PureWasm],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/animation*",
            "crates/engine_core/src/**/skin*",
            "crates/engine_assets/src/**/animation*",
            "crates/engine_scene/src/**/animation*",
        ],
        checks: &[Smoke("meshes")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/spline_mesh*",
            "crates/engine_scene/src/**/spline*",
        ],
        checks: &[Smoke("spline")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/streaming*",
            "crates/engine_core/src/**/streaming/**",
        ],
        checks: &[Smoke("streaming")],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/decals*",
            "crates/engine_render/src/**/decals*",
            "crates/engine_scene/src/**/decal*",
        ],
        checks: &[Smoke("surfaces"), Cells(&["metro-decals-off"])],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/effects*",
            "crates/engine_core/src/**/antialiasing*",
            "crates/engine_render/src/**/temporal*",
            "crates/engine_render/src/**/post*",
            "crates/engine_render/src/**/shafts*",
        ],
        checks: &[
            Smoke("taa"),
            Cells(&[
                "metro-ambient-occlusion-off",
                "metro-bloom-off",
                "metro-no-shafts",
                "metro-no-reflections",
                "metro-no-refraction",
                "garden-sun-shafts",
                "garden-reflections",
                "garden-refraction",
            ]),
        ],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/lights*",
            "crates/engine_render/src/**/lights*",
        ],
        checks: &[Cells(&["metro-local-lights-off"])],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/particles*",
            "crates/engine_render/src/**/particles*",
        ],
        checks: &[Cells(&["metro-particles-off"])],
    },
    Rule {
        paths: &[
            "crates/engine_core/src/**/probes*",
            "crates/engine_render/src/**/probes*",
        ],
        checks: &[
            Smoke("terrain-water-sky"),
            Cells(&["metro-reflection-probes-off", "garden-reflections"]),
        ],
    },
    Rule {
        paths: &["crates/engine_render/src/**/transparent*"],
        checks: &[Cells(&["metro-no-refraction", "garden-refraction"])],
    },
    // The renderer: shaders against layouts, the smoke tier in both hosts,
    // and each pass's own checks.
    Rule {
        paths: &["crates/engine_render/**"],
        checks: &[Shaders, Equivalence, Browser],
    },
    Rule {
        paths: &[
            "crates/engine_render/src/**/renderer*",
            "crates/engine_render/src/**/renderer/**",
            "crates/engine_render/src/**/render_graph*",
            "crates/engine_render/src/**/targets*",
        ],
        checks: &[Cells(&[
            "field-views",
            "motion-views",
            "city-views",
            "metro-views",
            "dense-views",
        ])],
    },
    Rule {
        paths: &[
            "crates/engine_render/src/**/terrain*",
            "crates/engine_render/src/**/terrain/**",
            "crates/engine_render/src/**/scatter*",
        ],
        checks: &[
            TERRAIN_SMOKES[0],
            TERRAIN_SMOKES[1],
            TERRAIN_SMOKES[2],
            TERRAIN_SMOKES[3],
            Cells(&["highlands", "highlands-views"]),
        ],
    },
    Rule {
        paths: &[
            "crates/engine_render/src/**/gpu_presentation/**",
            "crates/engine_render/src/**/presentation*",
        ],
        checks: &[Smoke("gpu-presentation"), Smoke("interpolation")],
    },
    Rule {
        paths: &[
            "crates/engine_render/src/**/materials*",
            "crates/engine_render/src/**/mipmaps*",
            "crates/engine_assets/src/**/texture*",
            "crates/engine_assets/src/**/ktx2*",
        ],
        checks: &[
            Smoke("texture"),
            Smoke("surfaces"),
            Cells(&["metro-bc7-off"]),
        ],
    },
    // Assets: meshes, materials, glTF, simplification.
    Rule {
        paths: &["crates/engine_assets/**"],
        checks: &[
            Pure("gltf"),
            Pure("simplify"),
            Smoke("meshes"),
            Smoke("texture"),
            Equivalence,
        ],
    },
    Rule {
        paths: &[
            "crates/engine_collision/**",
            "crates/engine_dev/src/features/collision/**",
        ],
        checks: &[
            Smoke("collision"),
            Smoke("terrain"),
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
            Smoke("agent"),
            Smoke("plugins"),
            Equivalence,
        ],
    },
    // The development tools and the hosts.
    Rule {
        paths: &["crates/engine_dev/**"],
        checks: &[
            Pure("requests"),
            Smoke("agent"),
            Smoke("native"),
            Smoke("host-parity"),
            Equivalence,
        ],
    },
    Rule {
        paths: &["crates/engine_native/**", "crates/engine_panic/**"],
        checks: &[
            Smoke("native"),
            Smoke("host-parity"),
            Smoke("agent"),
            Equivalence,
        ],
    },
    // The native host's measured runs and the load monitor measure uses.
    Rule {
        paths: &[
            "crates/engine_native/src/**/measure*",
            "crates/engine_profile/**",
            "crates/engine_profiler/**",
        ],
        checks: &[Smoke("native"), Tools("measure")],
    },
    // The Stress Lab: every workload's cells and its content digests.
    Rule {
        paths: &["apps/stress-lab/**"],
        checks: &[Pure("caches"), Equivalence],
    },
    Rule {
        paths: &["apps/stress-lab/src/metro/**"],
        checks: &[Cells(&["metro-views", "metro-bc7-off"])],
    },
    Rule {
        paths: &["apps/stress-lab/src/highlands/**"],
        checks: &[Cells(&["highlands", "highlands-views"])],
    },
    Rule {
        paths: &["apps/stress-lab/src/dense*"],
        checks: &[Cells(&["dense-views", "dense-visibility"])],
    },
    Rule {
        paths: &["apps/stress-lab/src/city/**"],
        checks: &[Cells(&["city-views"])],
    },
    // The content kit every world builds with.
    Rule {
        paths: &["apps/stress-lab/src/kit/**"],
        checks: &[Cells(&[
            "field-views",
            "motion-views",
            "city-views",
            "metro-views",
            "dense-views",
            "highlands",
            "highlands-views",
        ])],
    },
    Rule {
        paths: &[
            "apps/stress-lab/src/scene*",
            "apps/stress-lab/src/world*",
            "apps/stress-lab/src/field*",
            "apps/stress-lab/src/game*",
            "apps/stress-lab/src/config*",
            "apps/stress-lab/src/catalog*",
            "apps/stress-lab/src/presets*",
        ],
        checks: &[Cells(&["field-views", "motion-views"])],
    },
    Rule {
        paths: &["apps/stress-lab/src/authored*"],
        checks: &[Smoke("plugins"), Smoke("collision"), Smoke("hierarchy")],
    },
    Rule {
        paths: &["apps/stress-lab/src/browser*"],
        checks: &[Smoke("host-parity"), Browser],
    },
    Rule {
        paths: &[
            "apps/stress-lab/src/native*",
            "apps/stress-lab/src/main*",
            "apps/stress-lab/src/observation*",
            "apps/stress-lab/src/metrics*",
        ],
        checks: &[Smoke("native"), Tools("measure")],
    },
    // The tools.
    Rule {
        paths: &["tools/assetc/**"],
        checks: &[Smoke("terrain-files"), Pure("import")],
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
            PureWasm,
        ],
    },
    Rule {
        paths: &["tools/xtask/src/pure/terrain.rs"],
        checks: &[Pure("terrain"), PureWasm],
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
    // The factory's tools, which check themselves.
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
    // The smoke runner, its server and lock: one quick check through them.
    Rule {
        paths: &[
            "tools/xtask/src/smoke.rs",
            "tools/xtask/src/serve.rs",
            "tools/xtask/src/heavy.rs",
            "tools/xtask/src/pipeline.rs",
        ],
        checks: &[Smoke("agent-lifecycle")],
    },
    Rule {
        paths: &["tools/xtask/src/importing.rs"],
        checks: &[Pure("gltf"), Smoke("virtual-geometry")],
    },
    // The smokes and what they share.
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
        checks: &[
            TERRAIN_SMOKES[0],
            TERRAIN_SMOKES[1],
            TERRAIN_SMOKES[2],
            TERRAIN_SMOKES[3],
            Pure("terrain"),
            Pure("import"),
            Equivalence,
        ],
    },
    Rule {
        paths: &["web/scripts/fixtures/garden.scene.json"],
        checks: &[
            Smoke("hierarchy"),
            Smoke("texture"),
            Smoke("agent"),
            Equivalence,
        ],
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
        checks: &[
            Smoke("agent"),
            Smoke("agent-lifecycle"),
            Smoke("host-parity"),
            Smoke("native"),
        ],
    },
    Rule {
        paths: &["web/scripts/pure-wasm.mjs"],
        checks: &[PureWasm],
    },
    // Lint's own checks of the web host and its scripts (their types too),
    // the iteration loop's scripts, and generated files lint compares.
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
    // The web host.
    Rule {
        paths: &["web/src/**", "web/stress.html"],
        checks: &[
            Smoke("host-parity"),
            Smoke("gpu-presentation"),
            Smoke("agent"),
            Browser,
        ],
    },
    // The frame loop every host runs.
    Rule {
        paths: &[
            "crates/engine/src/host/frame_loop*",
            "crates/engine_native/src/window*",
            "web/src/engine/host.ts",
        ],
        checks: &[Smoke("frame-loop"), Smoke("streaming"), Smoke("terrain")],
    },
    // The input every host gathers and frames take.
    Rule {
        paths: &[
            "crates/engine/src/host/input*",
            "crates/engine/src/game_host*",
            "crates/engine_native/src/input*",
            "web/src/engine/input*",
        ],
        checks: &[Smoke("input"), Smoke("interpolation")],
    },
    Rule {
        paths: &["web/src/engine/job*"],
        checks: &[Smoke("jobs")],
    },
    Rule {
        paths: &[
            "web/src/engine/ktx2*",
            "web/src/engine/decode*",
            "web/public/basis/**",
        ],
        checks: &[Smoke("texture")],
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

/// Whether `path` (`/`-separated) matches `pattern`: `**` matches any
/// number of segments, none included, and `*` any run of characters
/// within one.
pub(crate) fn matches(pattern: &str, path: &str) -> bool {
    let pattern: Vec<&str> = pattern.split('/').collect();
    let path: Vec<&str> = path.split('/').collect();
    segments(&pattern, &path)
}

/// Whether the segments `path` match the segments `pattern`.
fn segments(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.split_first() {
        None => path.is_empty(),
        Some((&"**", rest)) => (0..=path.len()).any(|skip| segments(rest, &path[skip..])),
        Some((first, rest)) => path
            .split_first()
            .is_some_and(|(segment, tail)| segment_matches(first, segment) && segments(rest, tail)),
    }
}

/// Whether one segment `text` matches `pattern`, whose `*` match any run
/// of characters.
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

/// Every check a change of `path` calls for: those of each rule it
/// matches; none when it matches no rule.
pub(crate) fn checks_of(path: &str) -> Option<Vec<Check>> {
    let mut found = None;
    for rule in MAP {
        if rule.paths.iter().any(|pattern| matches(pattern, path)) {
            found
                .get_or_insert_with(Vec::new)
                .extend_from_slice(rule.checks);
        }
    }
    found
}
