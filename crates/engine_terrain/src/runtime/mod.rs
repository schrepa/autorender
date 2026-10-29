mod bakes;
mod bounce;
mod brick;
mod bridges;
mod components;
mod cook;
mod dispatch;
mod edits;
mod extract;
mod flow;
mod ground_cover;
mod identity;
mod lighting;
mod lookups;
mod macro_colors;
mod occupancy;
mod operations;
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
mod views;
mod water;
mod water_sim;
mod water_source;
mod waterfalls;
mod white_water;
mod wind;

use bakes::*;
use bounce::{Bounce, bounce, bounces, waits};
pub use brick::TerrainBrick;
use brick::join;
pub use bridges::*;
pub use components::*;
use cook::CookTiles;
use dispatch::*;
pub use edits::{
    TerrainEditControl, TerrainEdited, TerrainRedoControl, TerrainRiverControl, TerrainUndoControl,
};
pub use extract::*;
pub use ground_cover::*;
use identity::Generation;
use lighting::{Basis, Lighting, ground_tiles, light, lighting_level, lit};
pub use lookups::{
    HeightSample, RouteFound, StandingWater, TerrainHeightQuery, TerrainHeightResult,
    TerrainRouteQuery, TerrainRouteResult,
};
use macro_colors::{MacroColors, ask_colors, want_colors};
pub use occupancy::*;
pub use operations::{TerrainControl, TerrainControlResult, TerrainKey, terrain_key};
use plan::*;
pub use queries::*;
pub use residency::*;
pub use settings::*;
pub use settled::{SETTLED_KIND, SETTLED_PARITY, settled_alike};
use settled::{SettleQueue, Settled};
use shading::Shadings;
pub use stats::*;
use store::*;
pub use streaming::*;
pub use surroundings::*;
use tile_map::TileMap;
pub use views::*;
pub use water::*;
pub use water_sim::{
    LIVE_PIXELS, LONGEST_FRAME, SIM_TOUCHES, WaterRead, WaterSourceCell, WaterWindow, live_steps,
};
pub use wind::*;

use engine_core::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy_ecs::{
    prelude::{Component, Entity, FromWorld, Local, Resource, With, World},
    query::QueryState,
    schedule::IntoScheduleConfigs,
};
use glam::{Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

pub use crate::bounce::TerrainBounce;
use crate::jobs::{BakeJob, BounceJob, LightingJob, StackView, TerrainRecipe, TileJob};
pub use crate::lighting::TerrainLighting;
use crate::{
    Biome, BiomeLayer, BiomeRules, Biomes, Bridge, ErosionStage, ErosionStats, Fall, GivenRiver,
    HydrologyStage, HydrologyStats, Layer, Layers, LodRanges, Mask, NoiseMask, Obstacle,
    PATCH_QUADS, PATCHES_PER_TILE, PathStage, Pyramid, Remap, Route, RouteOptions, Solid, Stamp,
    TILE_QUADS, TILE_STRIDE, Tile, TileKey, TileSample, WantedTile, WantedView, WaterPoint,
};
use engine_assets::{AssetLibrary, Material, MaterialHandle, MeshHandle};

use engine_core::{
    Camera, Cooking, GlobalTransforms, IDLE_JOBS, JobBudget, JobFailures, JobId, Jobs, RecordKey,
    RenderMaterial, RenderMesh, RenderSnapshot, RenderTexture, URGENT_JOBS, VIEW_JOBS, Viewport,
    content_hash, job_output, job_priority,
};
use engine_settings::{ExtraRules, InvalidSetting, Rule, Setting, settings};
