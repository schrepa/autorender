//! Terrain: heightfields whose tiles stream around the camera,
//! drawn in continuous level of detail and sampled by queries and collision.
//!
//! An entity's [`Terrain`] describes one: where it lies, its tile pyramid,
//! where its tiles come from (generated from a layer stack, or read from an
//! `ETR1` file by byte ranges) and its surface: the rules
//! choosing each sample's materials and how those look. The [`Terrains`] resource
//! keeps every terrain's resident tiles. Each extraction streams toward the
//! camera: the coarsest levels stay resident everywhere, so the whole
//! terrain always draws; finer levels load nearest and ahead of the camera
//! first, and level-0 tiles load around the camera and [`StreamingFocus`](crate::StreamingFocus)
//! entities for collision. Generating, decoding and baking share a frame's
//! work time, larger until the camera's view first loads
//! whole than after; past a tile budget or the device's capacity the least
//! important tiles drop and count as dropped. Frames list each terrain's
//! resident tiles and the patches the camera draws in
//! [`RenderSnapshot::terrain`].
//!
//! Where the host runs workers, terrains bake and generate
//! tiles on them instead: frames share each terrain's layers with them,
//! hand them the most important missing tiles and make the tiles they
//! finish resident, which takes frames little time.
//!
//! Tiles load in the order the view needs them: the pinned
//! levels; the tiles the view draws and shows, with those under focus
//! entities; the tiles it draws out of sight, then collision's; those that
//! only shade what it shows; where the eye is headed; the rest. Distances
//! run to each tile's ground, so a high eye wants fine tiles only as near
//! as their ground is. Jobs carry that order as their priority, the first
//! tiers taking a third job per worker, which workers run first; jobs for
//! tiles no longer kept stop at once. Far tiles' objects standing far away
//! at their level are placed by jobs of their own after their tiles, so
//! the tiles under the view stop waiting for far forests.
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
mod extract;
mod files;
mod ground_cover;
mod identity;
mod in_place;
mod occupancy;
mod plan;
mod queries;
mod residency;
mod settings;
mod stats;
mod store;
mod streaming;
mod surroundings;
mod water;
mod waterfalls;
mod wind;
mod workers;

use bakes::*;
pub use bridges::*;
pub use components::*;
pub use extract::*;
pub use files::*;
pub use ground_cover::*;
use in_place::*;
pub use occupancy::*;
use plan::*;
pub use queries::*;
pub use residency::*;
pub use settings::*;
pub use stats::*;
use store::*;
pub(crate) use streaming::*;
pub use surroundings::*;
pub use water::*;
pub use wind::*;
use workers::*;

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
use engine_terrain::jobs::{
    BAKE_JOB, BAND_JOB, FAR_JOB, PLACE_JOB, StackView, TILE_JOB, TerrainRecipe,
};
use engine_terrain::{
    Baked, Biome, BiomeError, BiomeLayer, BiomeRules, Biomes, Bridge, ErosionStage, ErosionStats,
    Fall, FileHeader, Generator, HydrologyStage, HydrologyStats, Layer, Layers, LodRanges, Mask,
    NavOptions, NavPath, NoiseMask, Obstacle, PATCH_QUADS, PATCHES_PER_TILE, Patch, PathStage,
    Placing, Pyramid, Remap, Route, RouteOptions, Selection, Stamp, TILE_QUADS, TILE_STRIDE,
    TerrainBake, Tile, TileBand, TileKey, TileSample, WantedTile, WantedView, decode_tile,
    merge_placed,
};

use crate::{
    Camera, GlobalTransforms, IDLE_JOBS, JobId, Jobs, RenderMaterial, RenderMesh, RenderSnapshot,
    RenderTexture, URGENT_JOBS, VIEW_JOBS, content_hash, job_class, job_priority,
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
                crate::WorldSync,
                // Bridges spawn before waterfalls.
                (bridges::sync, waterfalls::sync)
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
