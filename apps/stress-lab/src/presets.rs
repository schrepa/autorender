//! The workloads the Stress Lab offers by name, on the page and natively:
//! each a configuration over the maximum visual load, with its title and
//! what it shows.

use serde::Serialize;

use crate::{StressConfig, StressScene};

/// A named workload.
#[derive(Clone, Debug, Serialize)]
pub struct Preset {
    /// What hosts select it by.
    pub value: &'static str,
    /// Its name in menus.
    pub label: &'static str,
    /// Its name over a running workload.
    pub title: &'static str,
    /// What it shows.
    pub note: &'static str,
    pub config: StressConfig,
}

/// What every preset changes: the most objects the tower field draws,
/// textured, culled and still.
fn baseline() -> StressConfig {
    StressConfig {
        seed: 1337,
        renderables: 90_000,
        moving_percent: 0,
        hierarchy: false,
        interpolation: false,
        colliders: 0,
        sensor_percent: 0,
        dense_colliders: false,
        queries_per_tick: 0,
        mesh_variety: 32,
        textured: true,
        culling: true,
        scene: StressScene::Field,
        compressed_textures: false,
    }
}

/// Every preset, the default first.
pub fn presets() -> Vec<Preset> {
    let base = baseline();
    vec![
        Preset {
            value: "visual",
            label: "Maximum visual · 90k objects",
            title: "MAXIMUM VISUAL LOAD",
            note: "Rendering, extraction, batching and texture sampling at maximum object count.",
            config: base.clone(),
        },
        Preset {
            value: "city",
            label: "Wild city · 90k objects",
            title: "WILD CITY",
            note: "52 building, vehicle and machine shapes: skyscrapers, traffic, trains, drones, cranes and Ferris wheels. Moving objects are the traffic.",
            config: StressConfig {
                scene: StressScene::City,
                moving_percent: 10,
                hierarchy: true,
                interpolation: true,
                mesh_variety: 64,
                ..base.clone()
            },
        },
        Preset {
            value: "metro",
            label: "Metro · every feature · 90k objects",
            title: "METRO",
            note: "The city with every engine feature: glTF maps over procedural textures (BC7 where supported), refracting glass, masked foliage, tints and rotating ads, skinned people, keyframed props, thousands of lights and particles.",
            config: StressConfig {
                scene: StressScene::Metro,
                moving_percent: 10,
                hierarchy: true,
                interpolation: true,
                mesh_variety: 64,
                compressed_textures: true,
                ..base.clone()
            },
        },
        Preset {
            value: "dense",
            label: "Dense geometry · virtual meshes · 2k objects",
            title: "DENSE GEOMETRY",
            note: "Boulders and fluted columns of 330,000 to 590,000 triangles each on terrain tiles, drawn as virtual geometry: the GPU picks the clusters each view needs. Build the files with `cargo run -p engine_vgbuild --release --bin dense_assets`.",
            config: StressConfig {
                scene: StressScene::Dense,
                renderables: 2_000,
                moving_percent: 5,
                interpolation: true,
                mesh_variety: 8,
                ..base.clone()
            },
        },
        Preset {
            value: "highlands",
            label: "Highlands · 8 km streamed terrain · 4k objects",
            title: "HIGHLANDS",
            note: "An 8 km terrain the engine generates as the camera moves: eroded mountains from noise, then hand-placed stamps level a plateau for a hamlet of towers, flatten a valley floor and cut a cave mouth. Coarse levels keep the whole world visible; finer tiles stream around the camera.",
            config: StressConfig {
                scene: StressScene::Highlands,
                renderables: 4_000,
                moving_percent: 10,
                hierarchy: true,
                interpolation: true,
                ..base.clone()
            },
        },
        Preset {
            value: "mixed",
            label: "Mixed · 10k objects + collision",
            title: "MIXED WORKLOAD",
            note: "A moving hierarchy, interpolation, filtered queries and sensors share the frame budget.",
            config: StressConfig {
                renderables: 10_000,
                moving_percent: 25,
                hierarchy: true,
                interpolation: true,
                colliders: 1_000,
                sensor_percent: 10,
                queries_per_tick: 32,
                mesh_variety: 8,
                ..base.clone()
            },
        },
        Preset {
            value: "motion",
            label: "Transforms · 50k moving objects",
            title: "TRANSFORM PRESSURE",
            note: "50,000 moving objects with parents and interpolation. Toggle each feature to isolate its cost.",
            config: StressConfig {
                renderables: 50_000,
                moving_percent: 100,
                hierarchy: true,
                interpolation: true,
                mesh_variety: 1,
                textured: false,
                ..base.clone()
            },
        },
        Preset {
            value: "queries",
            label: "Queries · 10k colliders",
            title: "SPATIAL QUERY PRESSURE",
            note: "10,000 colliders with moving bounds and 256 ray, overlap and sweep queries per fixed tick.",
            config: StressConfig {
                renderables: 10_000,
                moving_percent: 10,
                colliders: 10_000,
                queries_per_tick: 256,
                mesh_variety: 8,
                textured: false,
                ..base.clone()
            },
        },
        Preset {
            value: "sensors",
            label: "Dense sensors · budget pressure",
            title: "DENSE SENSOR PRESSURE",
            note: "Clusters deliberately exceed pair budgets. Incomplete detection is visible and retained in recordings.",
            config: StressConfig {
                renderables: 10_000,
                moving_percent: 25,
                colliders: 2_500,
                sensor_percent: 100,
                dense_colliders: true,
                mesh_variety: 8,
                ..base
            },
        },
    ]
}

/// The preset named `value`.
pub fn preset(value: &str) -> Option<Preset> {
    presets().into_iter().find(|preset| preset.value == value)
}
