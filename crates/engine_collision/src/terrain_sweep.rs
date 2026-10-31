use bevy_ecs::prelude::{Entity, World};
use engine_core::{Error, ErrorCode, Operation, Slot, bounds, missing_brick};
use engine_places::{Ground, GroundSlot};
#[cfg(feature = "schema")]
use engine_settings::stated;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::queries::{bounded, reply};
use crate::{CollisionRay, CollisionWorld, QueryFilter, refresh_collision_world};

const PLACED_RAY_HITS: usize = 16;

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainSweepQuery {
    pub center: [f64; 2],
    #[serde(default = "default_half_extent")]
    #[cfg_attr(feature = "schema", schemars(transform = stated(bounds::SWEEP_EXTENT)))]
    pub half_extent: f64,
    #[serde(default = "default_sweep_samples")]
    #[cfg_attr(feature = "schema", schemars(transform = stated(bounds::SWEEP_SAMPLES)))]
    pub samples: u32,
}

impl Operation for TerrainSweepQuery {
    const NAME: &'static str = "terrain_sweep";
    const CONTROL: bool = false;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        let Self {
            center,
            half_extent,
            samples,
        } = request;
        let cx = coordinate(center[0], "$.center[0]")?;
        let cz = coordinate(center[1], "$.center[1]")?;
        bounded(
            half_extent,
            bounds::SWEEP_EXTENT,
            ErrorCode::OutOfRange,
            "$.half_extent",
        )?;
        bounded(
            samples,
            bounds::SWEEP_SAMPLES,
            ErrorCode::OutOfRange,
            "$.samples",
        )?;
        sweep(world, Vec2::new(cx, cz), half_extent as f32, samples)
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }

    #[cfg(feature = "schema")]
    fn result_schema() -> Option<schemars::Schema> {
        Some(schemars::schema_for!(TerrainSweepResult))
    }
}

fn default_half_extent() -> f64 {
    64.0
}

fn default_sweep_samples() -> u32 {
    16
}

fn coordinate(value: f64, path: &str) -> Result<f32, Error> {
    bounded(value, bounds::COORDINATE, ErrorCode::OutOfRange, path).map(|value| value as f32)
}

fn sweep(world: &mut World, center: Vec2, half_extent: f32, samples: u32) -> Result<Value, Error> {
    let collision = world.contains_resource::<CollisionWorld>();
    if collision {
        refresh_collision_world(world)?;
    }
    let world = &*world;
    if !world.contains_resource::<Ground>() {
        return Err(missing_brick(GroundSlot::NAME, "$.op"));
    }
    let grounds: Vec<Entity> = world
        .get_resource::<CollisionWorld>()
        .into_iter()
        .flat_map(|collision| collision.ground_solids())
        .map(|solid| solid.terrain)
        .collect();
    let mut queried = 0;
    let mut level_zero = 0;
    let mut holes = 0;
    let mut rays = Worst::default();
    let (mut ray_checks, mut ray_misses, mut placed_first) = (0, 0, 0);
    let mut collected = Vec::new();
    for row in 0..samples {
        for column in 0..samples {
            let unit = Vec2::new(column as f32, row as f32) / (samples - 1) as f32 * 2.0 - 1.0;
            let point = center + unit * half_extent;
            let Some(sample) = Ground::point_in(world, point.x, point.y) else {
                continue;
            };
            queried += 1;
            if sample.hole {
                holes += 1;
                continue;
            }
            level_zero += usize::from(sample.level == 0);
            if sample.level != 0 {
                continue;
            }
            if collision {
                let mut hits = Vec::new();
                world.resource::<CollisionWorld>().raycast_all(
                    CollisionRay {
                        origin: Vec3::new(point.x, sample.height + 1000.0, point.y),
                        direction: Vec3::NEG_Y,
                        max_distance: 2000.0,
                    },
                    QueryFilter::default(),
                    &mut hits,
                    PLACED_RAY_HITS,
                )?;
                placed_first += usize::from(hits.first().is_some_and(|hit| hit.placed.is_some()));
                let hit = hits.into_iter().find(|hit| hit.placed.is_none());
                match hit.filter(|hit| grounds.contains(&hit.entity)) {
                    Some(hit) => {
                        ray_checks += 1;
                        rays.add((hit.point.y - sample.height).abs(), point);
                    }
                    None => ray_misses += 1,
                }
            }
            if collected.len() < 8 {
                collected.push(SweepExample {
                    at: point.to_array(),
                    height: sample.height,
                });
            }
        }
    }
    Ok(reply(TerrainSweepResult {
        points: samples * samples,
        resident: queried,
        level_zero,
        holes,
        collision: collision.then_some(SweepRays {
            compared: ray_checks,
            missed: ray_misses,
            placed_first,
            worst: rays,
        }),
        examples: collected,
    }))
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainSweepResult {
    pub points: u32,
    pub resident: usize,
    pub level_zero: usize,
    pub holes: usize,
    pub collision: Option<SweepRays>,
    pub examples: Vec<SweepExample>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SweepRays {
    pub compared: usize,
    pub missed: usize,
    pub placed_first: usize,
    pub worst: Worst,
}

#[derive(Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Worst {
    #[serde(rename = "max_difference")]
    pub difference: f32,
    pub at: Option<[f32; 2]>,
}

impl Worst {
    fn add(&mut self, difference: f32, point: Vec2) {
        if difference > self.difference || self.at.is_none() {
            self.difference = engine_core::math::max(self.difference, difference);
            self.at = Some(point.to_array());
        }
    }
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SweepExample {
    pub at: [f32; 2],
    pub height: f32,
}
