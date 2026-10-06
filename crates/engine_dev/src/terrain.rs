//! Read-only terrain observations (decision 051): heights at points, a
//! sweep comparing what terrain queries, collision rays and generated
//! terrains' layers say the ground is, and routes for paths (decision
//! 069).

use engine_collision::{CollisionRay, CollisionWorld, QueryFilter, refresh_collision_world};
use engine_core::{
    Engine, RouteOptions, Terrain, Terrains,
    bevy_ecs::prelude::World,
    glam::{Vec2, Vec3},
};
use engine_scene::{SceneEntity, SceneInstance};
use serde_json::{Value, json};

use crate::{DevError, DevRequest};

/// The most points a height request names, and samples along a sweep's
/// side.
pub(crate) const MAX_POINTS: usize = 256;
pub(crate) const MAX_SWEEP_SAMPLES: u32 = 64;
/// The largest coordinate a request names, as collision queries bound them.
const LIMIT: f64 = 1.0e6;
/// Hits a sweep ray may meet: placed objects (decision 064) above the
/// ground, and the ground itself.
const PLACED_RAY_HITS: usize = 16;

pub(crate) fn terrain_request(request: &DevRequest) -> bool {
    matches!(
        request,
        DevRequest::TerrainHeight { .. }
            | DevRequest::TerrainSweep { .. }
            | DevRequest::TerrainRoute { .. }
    )
}

fn coordinate(value: f64, path: &str) -> Result<f32, DevError> {
    if value.is_finite() && value.abs() <= LIMIT {
        Ok(value as f32)
    } else {
        Err(DevError::new(
            "invalid_request",
            path,
            "coordinates must be finite and within +/-1e6",
        ))
    }
}

/// A terrain's authored ID, or its entity bits outside scenes.
fn terrain_id(world: &World, entity: engine_core::bevy_ecs::prelude::Entity) -> Value {
    world.get::<SceneEntity>(entity).map_or_else(
        || json!(entity.to_bits()),
        |source| json!(source.authored_id),
    )
}

pub(crate) fn execute(
    engine: &mut Engine,
    _scene: Option<&SceneInstance>,
    request: DevRequest,
) -> Result<Value, DevError> {
    match request {
        DevRequest::TerrainHeight { points, .. } => {
            if points.is_empty() || points.len() > MAX_POINTS {
                return Err(DevError::new(
                    "invalid_request",
                    "$.points",
                    format!("name 1 to {MAX_POINTS} points"),
                ));
            }
            let world = engine.app.world();
            let terrains = world.resource::<Terrains>();
            let samples = points
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let x = coordinate(point[0], &format!("$.points[{index}][0]"))?;
                    let z = coordinate(point[1], &format!("$.points[{index}][1]"))?;
                    Ok(terrains.sample(x, z).map_or(Value::Null, |sample| {
                        // What its hydrology bake (decision 066) found there,
                        // as the finest level sees it, and the water standing
                        // there (decision 067): its surface (world height)
                        // and depth.
                        let bake = terrains.layers(sample.terrain).and_then(|layers| {
                            let terrain = terrains.terrain(sample.terrain)?;
                            let water = layers.hydrology()?;
                            let local = (x - terrain.origin.x, z - terrain.origin.z);
                            // Above the ground as tiles hold it, which layers
                            // after the bake may raise over the water.
                            let standing = water
                                .water
                                .at(local.0, local.1)
                                .map(|(surface, _)| terrain.origin.y + surface)
                                .filter(|&surface| surface > sample.height)
                                .map(|surface| {
                                    json!({"surface": surface, "depth": surface - sample.height})
                                });
                            Some((
                                water.at(local.0, local.1, terrain.pyramid.spacing(0)),
                                standing,
                            ))
                        });
                        // Meters from the nearest road's edge (decision
                        // 069), where a path's masks reach.
                        let path = terrains.layers(sample.terrain).and_then(|layers| {
                            let terrain = terrains.terrain(sample.terrain)?;
                            let road = layers.path_at(x - terrain.origin.x, z - terrain.origin.z);
                            (road.distance < engine_core::terrain_data::PATH_FAR)
                                .then_some(road.edge())
                        });
                        json!({
                            "terrain": terrain_id(world, sample.terrain),
                            "height": sample.height,
                            "normal": sample.normal.to_array(),
                            "hole": sample.hole,
                            "level": sample.level,
                            "water": bake.as_ref().map(|(found, _)| found),
                            "standing_water": bake.and_then(|(_, standing)| standing),
                            "path": path,
                        })
                    }))
                })
                .collect::<Result<Vec<_>, DevError>>()?;
            Ok(json!({"samples": samples, "stats": terrains.stats()}))
        }
        DevRequest::TerrainSweep {
            center,
            half_extent,
            samples,
            ..
        } => {
            let cx = coordinate(center[0], "$.center[0]")?;
            let cz = coordinate(center[1], "$.center[1]")?;
            if !(half_extent.is_finite() && (0.0..=4096.0).contains(&half_extent)) {
                return Err(DevError::new(
                    "invalid_request",
                    "$.half_extent",
                    "half_extent must be in [0, 4096] meters",
                ));
            }
            if !(2..=MAX_SWEEP_SAMPLES).contains(&samples) {
                return Err(DevError::new(
                    "invalid_request",
                    "$.samples",
                    format!("samples must be 2 to {MAX_SWEEP_SAMPLES} along each side"),
                ));
            }
            sweep(engine, Vec2::new(cx, cz), half_extent as f32, samples)
        }
        DevRequest::TerrainRoute {
            from,
            to,
            level,
            max_grade,
            grade_cost,
            water_cost,
            turn_cost,
            ..
        } => {
            let from = Vec2::new(
                coordinate(from[0], "$.from[0]")?,
                coordinate(from[1], "$.from[1]")?,
            );
            let to = Vec2::new(coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?);
            let defaults = RouteOptions::default();
            let ranged =
                |value: Option<f64>, low: f64, high: f64, path: &str, fallback: f32| match value {
                    None => Ok(fallback),
                    Some(value) if value.is_finite() && (low..=high).contains(&value) => {
                        Ok(value as f32)
                    }
                    Some(_) => Err(DevError::new(
                        "invalid_request",
                        path,
                        format!("must be in [{low}, {high}]"),
                    )),
                };
            let options = RouteOptions {
                level: match level {
                    None => defaults.level,
                    Some(level) if level <= 15 => level,
                    Some(_) => {
                        return Err(DevError::new(
                            "invalid_request",
                            "$.level",
                            "must be in [0, 15]",
                        ));
                    }
                },
                max_grade: ranged(max_grade, 0.001, 10.0, "$.max_grade", defaults.max_grade)?,
                grade_cost: ranged(grade_cost, 0.0, 1.0e6, "$.grade_cost", defaults.grade_cost)?,
                water_cost: ranged(water_cost, 0.0, 1.0e6, "$.water_cost", defaults.water_cost)?,
                turn_cost: ranged(turn_cost, 0.0, 1.0e6, "$.turn_cost", defaults.turn_cost)?,
            };
            let world = engine.app.world();
            let terrains = world.resource::<Terrains>();
            let Some(sample) = terrains.sample(from.x, from.y) else {
                return Err(DevError::new(
                    "invalid_request",
                    "$.from",
                    "no resident terrain lies under from",
                ));
            };
            let Some(terrain) = terrains.terrain(sample.terrain) else {
                return Err(DevError::new(
                    "invalid_request",
                    "$.from",
                    "no terrain lies under from",
                ));
            };
            let origin = Vec2::new(terrain.origin.x, terrain.origin.z);
            let route = terrains.route(sample.terrain, from, to, &options);
            Ok(json!({
                "terrain": terrain_id(world, sample.terrain),
                "options": options,
                "origin": origin.to_array(),
                // Centimeters are plenty, and keep long routes within the
                // response limit.
                "route": route.map(|route| json!({
                    "points": route
                        .points
                        .iter()
                        .map(|point| point.map(|value| (f64::from(value) * 100.0).round() / 100.0))
                        .collect::<Vec<_>>(),
                    "length": route.length,
                    "steepest": route.steepest,
                    "hairpin_steps": route.hairpin_steps,
                })),
            }))
        }
        _ => unreachable!("only terrain requests reach here"),
    }
}

/// The largest difference so far and where.
#[derive(Default)]
struct Worst {
    difference: f32,
    at: Option<[f32; 2]>,
}

impl Worst {
    fn add(&mut self, difference: f32, point: Vec2) {
        if difference > self.difference || self.at.is_none() {
            self.difference = engine_core::math::max(self.difference, difference);
            self.at = Some(point.to_array());
        }
    }

    fn value(&self) -> Value {
        json!({"max_difference": self.difference, "at": self.at})
    }
}

/// The ground under a grid of `samples` by `samples` points across
/// `half_extent` meters of `center`, by terrain queries (the resident
/// tiles renderers draw), collision rays and generated terrains' layers.
fn sweep(
    engine: &mut Engine,
    center: Vec2,
    half_extent: f32,
    samples: u32,
) -> Result<Value, DevError> {
    let collision = engine.app.world().contains_resource::<CollisionWorld>();
    if collision {
        refresh_collision_world(engine.app.world_mut()).map_err(crate::collision::native_error)?;
    }
    let world = engine.app.world();
    let terrains = world.resource::<Terrains>();
    let mut queried = 0;
    let mut level_zero = 0;
    let mut holes = 0;
    let mut rays = Worst::default();
    let mut layers = Worst::default();
    let (mut ray_checks, mut ray_misses, mut layer_checks, mut placed_first) = (0, 0, 0, 0);
    let mut collected = Vec::new();
    for row in 0..samples {
        for column in 0..samples {
            let unit = Vec2::new(column as f32, row as f32) / (samples - 1) as f32 * 2.0 - 1.0;
            let point = center + unit * half_extent;
            let Some(sample) = terrains.sample(point.x, point.y) else {
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
            // Collision holds level-0 tiles around focus points only. Rays
            // meeting placed objects first (decision 064) count, and the
            // ground beneath them compares.
            if collision {
                let mut hits = Vec::new();
                world
                    .resource::<CollisionWorld>()
                    .raycast_all(
                        CollisionRay {
                            origin: Vec3::new(point.x, sample.height + 1000.0, point.y),
                            direction: Vec3::NEG_Y,
                            max_distance: 2000.0,
                        },
                        QueryFilter::default(),
                        &mut hits,
                        PLACED_RAY_HITS,
                    )
                    .map_err(crate::collision::native_error)?;
                placed_first += usize::from(hits.first().is_some_and(|hit| hit.placed.is_some()));
                let hit = hits.into_iter().find(|hit| hit.placed.is_none());
                match hit.filter(|hit| hit.entity == sample.terrain) {
                    Some(hit) => {
                        ray_checks += 1;
                        rays.add((hit.point.y - sample.height).abs(), point);
                    }
                    None => ray_misses += 1,
                }
            }
            // The layers tiles generate from, their hydrology baked.
            if let Some(terrain) = world.get::<Terrain>(sample.terrain)
                && let Some(stack) = world.resource::<Terrains>().layers(sample.terrain)
            {
                let local = point - Vec2::new(terrain.origin.x, terrain.origin.z);
                let height = terrain.origin.y + stack.height_at(&terrain.pyramid, local);
                layer_checks += 1;
                layers.add((height - sample.height).abs(), point);
            }
            if collected.len() < 8 {
                collected.push(json!({"at": point.to_array(), "height": sample.height}));
            }
        }
    }
    Ok(json!({
        "points": samples * samples,
        "resident": queried,
        "level_zero": level_zero,
        "holes": holes,
        "collision": collision.then(|| json!({
            "compared": ray_checks,
            "missed": ray_misses,
            "placed_first": placed_first,
            "worst": rays.value(),
        })),
        "layers": json!({"compared": layer_checks, "worst": layers.value()}),
        "examples": collected,
    }))
}
