//! Terrain: heightfields whose tiles stream around the camera,
//! drawn in continuous level of detail and sampled by queries and collision.
//!
//! An entity's [`Terrain`] describes one: where it lies, its tile pyramid,
//! the layer stack its tiles generate from (they load as records, the
//! cook's or made by the same jobs) and its surface: the rules
//! choosing each sample's materials and how those look. The [`Terrains`] resource
//! keeps every terrain's resident tiles. Each extraction streams toward the
//! camera: the coarsest levels stay resident everywhere, so the whole
//! terrain always draws; finer levels load nearest and ahead of the camera
//! first, and level-0 tiles load around the camera and [`StreamingFocus`](crate::StreamingFocus)
//! entities for collision. Past a tile budget or the device's capacity the
//! least important tiles drop and count as dropped. Frames list each
//! terrain's resident tiles and the patches the camera draws in
//! [`RenderSnapshot::terrain`].
//!
//! Its bakes and tiles are records ([`crate::Jobs`]): the host hands each
//! back from the cook's packs or what it kept, or runs its job on a worker
//! (in place on a host without workers) and keeps the output. Frames share
//! each terrain's layers with the jobs, ask for the most important missing
//! tiles and make the records that arrive resident within their time and
//! uploads, larger until the camera's view first loads whole than after.
//! While a session cooks ([`crate::Cooking`]), every tile a camera could
//! want is asked for, for the host's pack.
//!
//! Tiles load in the order the view needs them: the pinned
//! levels; the tiles the view draws and shows, with those under focus
//! entities; the tiles it draws out of sight, then collision's; those that
//! only shade what it shows; where the eye is headed; the rest. Distances
//! run to each tile's ground, so a high eye wants fine tiles only as near
//! as their ground is. Jobs carry that order as their priority, the first
//! tiers taking a third job per worker, which workers run first; jobs for
//! tiles no longer kept stop at once.
//!
//! Queries ([`Terrains::sample`], [`Terrains::height_at`] and the rest)
//! answer from the finest resident tile at a point, on the triangles
//! renderers and collision use, and return `None` where no tile is
//! resident.
//!
//! The [`TerrainSettings`] are inserted by default; without them, or when
//! disabled, terrains neither stream nor draw.

mod bakes;
mod bridges;
mod components;
mod cook;
mod dispatch;
mod extract;
mod flow;
mod ground_cover;
mod identity;
mod lighting;
mod occupancy;
mod plan;
mod queries;
mod residency;
mod rewater;
mod settings;
mod settled;
mod stats;
mod store;
mod streaming;
mod surroundings;
mod tile_map;
mod water;
mod water_sim;
mod water_source;
mod waterfalls;
mod wind;

use bakes::*;
pub use bridges::*;
pub use components::*;
use cook::CookTiles;
use dispatch::*;
pub use extract::*;
pub use ground_cover::*;
use identity::Generation;
use lighting::{Lighting, light, lit};
pub use occupancy::*;
use plan::*;
pub use queries::*;
pub use residency::*;
pub use settings::*;
use settled::Settled;
pub use stats::*;
use store::*;
pub(crate) use streaming::*;
pub use surroundings::*;
use tile_map::TileMap;
pub use water::*;
pub use water_sim::{GHOST_DRY, WaterGhost, WaterSourceCell, WaterWindow};
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
use engine_terrain::jobs::{BakeJob, LightingJob, PortalsJob, StackView, TerrainRecipe, TileJob};
pub use engine_terrain::lighting::TerrainLighting;
use engine_terrain::{
    Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, Bridge, ErosionStage, ErosionStats, Fall,
    Gait, HydrologyStage, HydrologyStats, Layer, Layers, LodRanges, Mask, NoiseMask, Obstacle,
    PATCH_QUADS, PATCHES_PER_TILE, Patch, PathStage, Pyramid, Remap, Route, RouteOptions,
    Selection, Stamp, TILE_QUADS, TILE_STRIDE, Tile, TileKey, TileSample, WantedTile, WantedView,
    WaterPoint,
};

use crate::{
    Camera, Cooking, GlobalTransforms, IDLE_JOBS, JobBudget, JobFailures, JobId, Jobs, RecordKey,
    RenderMaterial, RenderMesh, RenderSnapshot, RenderTexture, URGENT_JOBS, VIEW_JOBS,
    content_hash, job_output, job_priority,
};
use engine_settings::{ExtraRules, InvalidSetting, Rule, Setting, settings};

/// Heightfield terrains streaming around the camera, with their bakes,
/// placed objects, roads, bridges, inland water and waterfalls:
/// [`crate::Feature::Terrain`]. It provides the [`crate::Ground`] grounded
/// entities stand on and deformers press into.
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
            // Rivers follow what stands in them once bodies have moved.
            .add_systems(
                crate::FixedSimulation,
                flow::step.in_set(crate::FixedSimulationSystems::Ride),
            )
            .add_systems(
                crate::WorldSync,
                // Bridges spawn before waterfalls; water's simulation
                // windows follow the camera.
                (
                    bridges::sync,
                    waterfalls::sync,
                    water_sim::place_water_windows,
                )
                    .chain()
                    .in_set(crate::WorldSyncSystems::Spawn),
            );
    }
}

/// The terrains' ground under world `(x, z)`: their finest resident sample
/// there, none over a hole.
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
