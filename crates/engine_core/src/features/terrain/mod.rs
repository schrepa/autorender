mod bakes;
mod bounce;
mod bridges;
mod components;
mod cook;
mod dispatch;
mod extract;
mod flow;
mod ground_cover;
mod identity;
mod lighting;
mod macro_colors;
mod occupancy;
mod plan;
mod queries;
mod residency;
mod rewater;
mod settings;
mod settled;
mod shading;
mod stats;
mod store;
mod streaming;
mod surroundings;
mod tile_map;
mod water;
mod water_sim;
mod water_source;
mod waterfalls;
mod white_water;
mod wind;

use bakes::*;
use bounce::{Bounce, bounce, bounces, waits};
pub use bridges::*;
pub use components::*;
use cook::CookTiles;
use dispatch::*;
pub use extract::*;
pub use ground_cover::*;
use identity::Generation;
use lighting::{Basis, Lighting, ground_tiles, light, lighting_level, lit};
use macro_colors::{MacroColors, ask_colors, want_colors};
pub use occupancy::*;
use plan::*;
pub use queries::*;
pub use residency::*;
pub use settings::*;
pub use settled::{SETTLED_KIND, SETTLED_PARITY, settled_alike};
use settled::{SettleQueue, Settled};
use shading::Shadings;
pub use stats::*;
use store::*;
pub(crate) use streaming::*;
pub use surroundings::*;
use tile_map::TileMap;
pub use water::*;
pub use water_sim::{
    LIVE_PIXELS, LONGEST_FRAME, SIM_TOUCHES, WaterRead, WaterSourceCell, WaterWindow, live_steps,
};
pub use wind::*;

use crate::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy_ecs::{
    prelude::{Component, Entity, FromWorld, Local, Resource, With, World},
    query::QueryState,
    schedule::IntoScheduleConfigs,
};
use glam::{Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use engine_assets::{AssetLibrary, Material, MaterialHandle, MeshHandle};
pub use engine_terrain::bounce::TerrainBounce;
use engine_terrain::jobs::{BakeJob, BounceJob, LightingJob, StackView, TerrainRecipe, TileJob};
pub use engine_terrain::lighting::TerrainLighting;
use engine_terrain::{
    Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, Bridge, ErosionStage, ErosionStats, Fall,
    GivenRiver, HydrologyStage, HydrologyStats, Layer, Layers, LodRanges, Mask, NoiseMask,
    Obstacle, PATCH_QUADS, PATCHES_PER_TILE, PathStage, Pyramid, Remap, Route, RouteOptions, Solid,
    Stamp, TILE_QUADS, TILE_STRIDE, Tile, TileKey, TileSample, WantedTile, WantedView, WaterPoint,
};

use crate::{
    Camera, Cooking, GlobalTransforms, IDLE_JOBS, JobBudget, JobFailures, JobId, Jobs, RecordKey,
    RenderMaterial, RenderMesh, RenderSnapshot, RenderTexture, URGENT_JOBS, VIEW_JOBS,
    content_hash, job_output, job_priority,
};
use engine_settings::{ExtraRules, InvalidSetting, Rule, Setting, settings};

#[derive(Default)]
pub struct TerrainPlugin;

impl bevy_app::Plugin for TerrainPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        crate::install(app, crate::Feature::Terrain);
        app.world_mut()
            .resource_mut::<crate::WaterSources>()
            .register("terrain", water_source::terrain_water);
        app.init_resource::<Terrains>()
            .insert_resource(crate::Ground {
                sample: ground_sample,
                revision: |world| {
                    world
                        .get_resource::<Terrains>()
                        .map_or(0, Terrains::revision)
                },
            })
            .add_systems(
                crate::ViewSync,
                stream_terrains.in_set(crate::ViewSyncSystems::Stream),
            )
            .add_systems(
                crate::FixedSimulation,
                flow::step.in_set(crate::FixedSimulationSystems::Ride),
            )
            .add_systems(
                crate::WorldSync,
                (
                    bridges::sync,
                    waterfalls::sync,
                    white_water::sync,
                    water_sim::place_water_windows,
                )
                    .chain()
                    .in_set(crate::WorldSyncSystems::Spawn),
            );
    }
}

fn ground_sample(world: &World, x: f32, z: f32) -> Option<crate::GroundSample> {
    let sample = world
        .get_resource::<Terrains>()?
        .sample(x, z)
        .filter(|sample| !sample.hole)?;
    Some(crate::GroundSample {
        height: sample.height,
        level: sample.level,
    })
}
