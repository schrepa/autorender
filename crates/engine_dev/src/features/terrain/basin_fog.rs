//! Read-only samples of the current terrain fog solution.

use super::{TerrainKey, terrain_key};
use crate::features::{EngineQuery, feature};
use crate::protocol::bounds::MAX_POINTS;
use crate::results::result;
use crate::{DevError, coordinate};
use engine_core::{BasinFogs, Engine, Feature, Terrain, bevy_ecs::prelude::Entity, glam::Vec2};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Samples floor, top, depth and pool identity at world x/z points.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct BasinFogSampleQuery {
    /// Authored terrain ID, or its runtime entity bits.
    pub terrain: String,
    /// World horizontal coordinates, bounded by the normal query limit.
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = MAX_POINTS)))]
    pub points: Vec<[f64; 2]>,
}

/// A valid grid cell's published heights and stable pool identity.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct BasinFogSample {
    /// False for a hole in the terrain grid.
    pub valid: bool,
    /// World height of the effective ground or water floor, meters.
    pub floor: f32,
    /// World height of the horizontal fog top, meters.
    pub top: f32,
    /// Fog thickness, meters; zero in a dry cell.
    pub depth: f32,
    /// Stable row-major pool label; zero when dry.
    pub pool: u32,
}

/// Samples from one terrain; null outside its current published solution.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct BasinFogSampleResult {
    /// The terrain queried.
    pub terrain: TerrainKey,
    /// Samples in request order; null outside or pending.
    pub samples: Vec<Option<BasinFogSample>>,
}

impl EngineQuery for BasinFogSampleQuery {
    fn answer(
        &self,
        engine: &mut Engine,
        scene: Option<&SceneInstance>,
    ) -> Result<Value, DevError> {
        if self.points.is_empty() || self.points.len() > MAX_POINTS {
            return Err(DevError::new(
                ErrorCode::OutOfRange,
                "$.points",
                format!("name 1 to {MAX_POINTS} points"),
            ));
        }
        let world = engine.app.world_mut();
        let entity = scene
            .and_then(|s| s.entity(&self.terrain))
            .or_else(|| {
                world
                    .query::<(Entity, &Terrain)>()
                    .iter(world)
                    .take(crate::MAX_SCAN_ENTITIES)
                    .find(|(id, _)| id.to_bits().to_string() == self.terrain)
                    .map(|(id, _)| id)
            })
            .filter(|id| world.get::<Terrain>(*id).is_some())
            .ok_or_else(|| {
                DevError::new(
                    ErrorCode::UnknownTerrain,
                    "$.terrain",
                    format!("no terrain {}", self.terrain),
                )
            })?;
        let fog = feature::<BasinFogs>(world, Feature::Terrain)?;
        let samples = self
            .points
            .iter()
            .enumerate()
            .map(|(index, p)| {
                let x = coordinate(p[0], &format!("$.points[{index}][0]"))?;
                let z = coordinate(p[1], &format!("$.points[{index}][1]"))?;
                Ok(fog
                    .sample(entity, Vec2::new(x, z))
                    .map(|(valid, [floor, top, depth], pool)| BasinFogSample {
                        valid,
                        floor,
                        top,
                        depth,
                        pool,
                    }))
            })
            .collect::<Result<Vec<_>, DevError>>()?;
        Ok(result::basin_fog_sample(BasinFogSampleResult {
            terrain: terrain_key(world, entity),
            samples,
        }))
    }
}
