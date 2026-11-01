mod bakes;
mod brick;
mod bridges;
mod components;
mod cook;
mod dispatch;
mod edits;
mod extract;
mod ground_face;
mod identity;
mod lookups;
mod operations;
mod plan;
mod queries;
mod residency;
mod settings;
mod solids;
mod stats;
mod store;
mod streaming;
mod surroundings;
mod tile_map;
mod views;

use bakes::*;
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
use identity::Generation;
pub use lookups::{
    HeightSample, RouteFound, StandingWater, TerrainHeightQuery, TerrainHeightResult,
    TerrainRouteQuery, TerrainRouteResult,
};
pub use operations::{TerrainControl, TerrainControlResult, TerrainKey, terrain_key};
use plan::*;
pub use queries::*;
pub use residency::*;
pub use settings::*;
pub use stats::*;
use store::*;
pub use streaming::*;
pub use surroundings::*;
use tile_map::TileMap;
pub use views::*;

use engine_core::collections::{HashMap, HashSet};
use std::sync::Arc;

use bevy_ecs::{
    prelude::{Component, Entity, FromWorld, Local, Resource, With, World},
    query::QueryState,
    schedule::IntoScheduleConfigs,
};
use glam::{Mat4, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::jobs::{BakeJob, BounceJob, LightingJob, StackView, TerrainRecipe, TileJob};
use crate::{
    Biome, BiomeLayer, BiomeRules, Biomes, Bridge, Expr, Fall, GivenRiver, HydrologyStats, Layers,
    LodRanges, NoiseMask, PATCH_QUADS, PATCHES_PER_TILE, Pyramid, Route, RouteOptions, Solid,
    Stamp, TILE_QUADS, TILE_STRIDE, Tile, TileKey, TileSample, WantedTile, WantedView, WaterPoint,
};
use engine_assets::{AssetLibrary, Material, MaterialHandle, MeshHandle};

use engine_core::{
    Camera, Cooking, GlobalTransforms, IDLE_JOBS, JobBudget, JobFailures, JobId, Jobs, RecordKey,
    RenderMaterial, RenderMesh, RenderSnapshot, RenderTexture, URGENT_JOBS, VIEW_JOBS, Viewport,
    content_hash, job_output, job_priority,
};
use engine_settings::{ExtraRules, InvalidSetting, Rule, Setting, settings};
