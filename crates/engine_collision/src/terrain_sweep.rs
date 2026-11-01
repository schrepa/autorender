use bevy_ecs::prelude::*;
use engine_core::{Error, ErrorCode, Operation, bounds, stated};
use engine_places::ground::Ground;
use engine_places::physics::{Collide, Collision, CollisionRay, QueryFilter};
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::queries::{bounded, reply, schema_of};
use crate::world::{CollisionWorld, world_of};

const PLACED_RAY_HITS: usize = 16;

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct TerrainSweepQuery {
    pub center: [f64; 2],
    #[serde(default = "default_half_extent")]
    #[schemars(transform = stated(bounds::SWEEP_EXTENT))]
    pub half_extent: f64,
    #[serde(default = "default_sweep_samples")]
    #[schemars(transform = stated(bounds::SWEEP_SAMPLES))]
    pub samples: u32,
}

impl Operation for TerrainSweepQuery {
    const NAME: &'static str = "terrain_sweep";
    type Params = (Res<'static, Collision>, Res<'static, Ground>);
    type Request = Self;

    fn run(
        (collision, ground): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
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
        sweep(
            world_of(&collision),
            &ground,
            Vec2::new(cx, cz),
            half_extent as f32,
            samples,
        )
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }

    fn result_schema() -> Option<Value> {
        Some(schema_of::<TerrainSweepResult>())
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

fn sweep(
    collision: Option<&CollisionWorld>,
    ground: &Ground,
    center: Vec2,
    half_extent: f32,
    samples: u32,
) -> Result<Value, Error> {
    let grounds: Vec<Entity> = collision
        .into_iter()
        .flat_map(|collision| collision.terrain.owners())
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
            let Some(sample) = ground.point(point) else {
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
            if let Some(collision) = collision {
                let mut hits = Vec::new();
                collision.raycast_all(
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
        collision: collision.map(|_| SweepRays {
            compared: ray_checks,
            missed: ray_misses,
            placed_first,
            worst: rays,
        }),
        examples: collected,
    }))
}

#[derive(Serialize, schemars::JsonSchema)]
struct TerrainSweepResult {
    points: u32,
    resident: usize,
    level_zero: usize,
    holes: usize,
    collision: Option<SweepRays>,
    examples: Vec<SweepExample>,
}

#[derive(Serialize, schemars::JsonSchema)]
struct SweepRays {
    compared: usize,
    missed: usize,
    placed_first: usize,
    worst: Worst,
}

#[derive(Default, Serialize, schemars::JsonSchema)]
struct Worst {
    #[serde(rename = "max_difference")]
    difference: f32,
    at: Option<[f32; 2]>,
}

impl Worst {
    fn add(&mut self, difference: f32, point: Vec2) {
        if difference > self.difference || self.at.is_none() {
            self.difference = engine_math::max(self.difference, difference);
            self.at = Some(point.to_array());
        }
    }
}

#[derive(Serialize, schemars::JsonSchema)]
struct SweepExample {
    at: [f32; 2],
    height: f32,
}
