//! Read-only terrain queries: heights at points, a sweep comparing what
//! terrain queries, collision rays and generated terrains' layers say the
//! ground is, routes for paths, walks and tracks.

use engine_collision::{CollisionRay, CollisionWorld, QueryFilter, refresh_collision_world};
use engine_core::{
    Deformation, Engine, Feature, PresentedTime, RouteOptions, Terrain, TerrainSource, Terrains,
    Tracks,
    bevy_ecs::prelude::{Entity, World},
    glam::{Vec2, Vec3},
    has_feature,
};
use engine_scene::SceneEntity;
use engine_settings::ErrorCode;
use serde_json::{Value, json};

use super::{coordinate, feature, missing};
use crate::protocol::TerrainQuery;
use crate::protocol::bounds::{self, MAX_POINTS};
use crate::{DevError, bounded};

/// Hits a sweep ray may meet: placed objects above the
/// ground, and the ground itself.
const PLACED_RAY_HITS: usize = 16;

/// A terrain's authored ID, or its entity bits outside scenes.
fn terrain_id(world: &World, entity: engine_core::bevy_ecs::prelude::Entity) -> Value {
    world.get::<SceneEntity>(entity).map_or_else(
        || json!(entity.to_bits()),
        |source| json!(source.authored_id),
    )
}

/// Runs terrain query `query`: its result.
pub(crate) fn execute(engine: &mut Engine, query: TerrainQuery) -> Result<Value, DevError> {
    match query {
        TerrainQuery::TerrainHeight { points } => heights(engine, &points),
        TerrainQuery::NavPath { from, to, options } => walk(engine, from, to, options),
        TerrainQuery::DeformationQuery { points, seconds } => tracks(engine, &points, seconds),
        TerrainQuery::TerrainSweep {
            center,
            half_extent,
            samples,
        } => {
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
            sweep(engine, Vec2::new(cx, cz), half_extent as f32, samples)
        }
        TerrainQuery::TerrainRoute {
            from,
            to,
            level,
            max_grade,
            grade_cost,
            water_cost,
            turn_cost,
        } => {
            let from = Vec2::new(
                coordinate(from[0], "$.from[0]")?,
                coordinate(from[1], "$.from[1]")?,
            );
            let to = Vec2::new(coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?);
            let defaults = RouteOptions::default();
            let options = RouteOptions {
                level: level.unwrap_or(defaults.level),
                max_grade: max_grade.map_or(defaults.max_grade, |value| value as f32),
                grade_cost: grade_cost.map_or(defaults.grade_cost, |value| value as f32),
                water_cost: water_cost.map_or(defaults.water_cost, |value| value as f32),
                turn_cost: turn_cost.map_or(defaults.turn_cost, |value| value as f32),
            };
            options.validate().map_err(|invalid| {
                DevError::new(
                    ErrorCode::OutOfRange,
                    format!("$.{}", invalid.field),
                    invalid.rule,
                )
            })?;
            let world = engine.app.world();
            let terrains = feature::<Terrains>(world, Feature::Terrain)?;
            let Some(sample) = terrains.sample(from.x, from.y) else {
                return Err(DevError::new(
                    ErrorCode::InvalidRequest,
                    "$.from",
                    "no resident terrain lies under from",
                ));
            };
            let Some(terrain) = terrains.terrain(sample.terrain) else {
                return Err(DevError::new(
                    ErrorCode::InvalidRequest,
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
    }
}

/// What the resident tiles say of the ground at `points`.
fn heights(engine: &mut Engine, points: &[[f64; 2]]) -> Result<Value, DevError> {
    if points.is_empty() || points.len() > MAX_POINTS {
        return Err(DevError::new(
            ErrorCode::OutOfRange,
            "$.points",
            format!("name 1 to {MAX_POINTS} points"),
        ));
    }
    let world = engine.app.world();
    let terrains = feature::<Terrains>(world, Feature::Terrain)?;
    let samples = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let x = coordinate(point[0], &format!("$.points[{index}][0]"))?;
            let z = coordinate(point[1], &format!("$.points[{index}][1]"))?;
            Ok(terrains.sample(x, z).map_or(Value::Null, |sample| {
                // What its hydrology bake found there,
                // as the finest level sees it, and the water standing
                // there: its surface (world height)
                // and depth.
                let bake = terrains.layers(sample.terrain).and_then(|layers| {
                    let terrain = terrains.terrain(sample.terrain)?;
                    let water = layers.hydrology()?;
                    let local = (x - terrain.origin.x, z - terrain.origin.z);
                    // Above the ground as tiles hold it, which layers
                    // after the bake may raise over the water.
                    let standing = water
                        .water_at(local.0, local.1)
                        .map(|(surface, _)| terrain.origin.y + surface)
                        .filter(|&surface| surface > sample.height)
                        .map(
                            |surface| json!({"surface": surface, "depth": surface - sample.height}),
                        );
                    Some((
                        water.at(local.0, local.1, terrain.pyramid.spacing(0)),
                        standing,
                    ))
                });
                // Meters from the nearest road's edge,
                // where a path's masks reach.
                let path = terrains.layers(sample.terrain).and_then(|layers| {
                    let terrain = terrains.terrain(sample.terrain)?;
                    let road = layers.path_at(x - terrain.origin.x, z - terrain.origin.z);
                    (road.distance < engine_core::terrain_data::PATH_FAR).then_some(road.edge())
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

/// A navigation walk over the first generated terrain from `from` to `to`.
fn walk(
    engine: &mut Engine,
    from: [f64; 2],
    to: [f64; 2],
    options: Option<engine_core::terrain_data::NavOptions>,
) -> Result<Value, DevError> {
    let from = Vec2::new(
        coordinate(from[0], "$.from[0]")?,
        coordinate(from[1], "$.from[1]")?,
    );
    let to = Vec2::new(coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?);
    let options = options.unwrap_or_default();
    options.validate().map_err(|invalid| {
        DevError::new(
            ErrorCode::OutOfRange,
            format!("$.options.{}", invalid.field),
            invalid.rule,
        )
    })?;
    let world = engine.app.world_mut();
    let entity = world
        .query::<(Entity, &Terrain)>()
        .iter(world)
        .find(|(_, terrain)| matches!(terrain.source, TerrainSource::Generated(_)))
        .map(|(entity, _)| entity)
        .ok_or_else(|| DevError::new(ErrorCode::UnknownTerrain, "$", "no generated terrain"))?;
    let walk = world
        .get_resource::<Terrains>()
        .filter(|_| has_feature(world, Feature::Terrain))
        .ok_or_else(|| missing(Feature::Terrain))?
        .walk(entity, from, to, &options);
    let round = |value: f32| (f64::from(value) * 100.0).round() / 100.0;
    Ok(json!({
        "terrain": terrain_id(world, entity),
        "options": options,
        "walk": walk.map(|walk| json!({
            "points": walk.points.iter().map(|point| point.map(round)).collect::<Vec<_>>(),
            "length": walk.length,
            "level": walk.level,
            "searched": walk.searched,
        })),
    }))
}

/// The tracks pressed at `points`, refilled to `seconds`.
fn tracks(
    engine: &mut Engine,
    points: &[[f64; 2]],
    seconds: Option<f64>,
) -> Result<Value, DevError> {
    if points.is_empty() || points.len() > MAX_POINTS {
        return Err(DevError::new(
            ErrorCode::OutOfRange,
            "$.points",
            format!("name 1 to {MAX_POINTS} points"),
        ));
    }
    let world = engine.app.world();
    let seconds = match seconds {
        Some(seconds) if seconds.is_finite() && seconds.abs() <= 1.0e9 => seconds,
        Some(_) => {
            return Err(DevError::new(
                ErrorCode::OutOfRange,
                "$.seconds",
                "seconds must be finite and within +/-1e9",
            ));
        }
        None => world
            .get_resource::<PresentedTime>()
            .map_or(0.0, |time| time.seconds),
    };
    let settings = world
        .get_resource::<Deformation>()
        .copied()
        .unwrap_or_default()
        .sanitized();
    let tracks = feature::<Tracks>(world, Feature::Deformation)?;
    let terrains = feature::<Terrains>(world, Feature::Terrain)?;
    // Tenths of millimeters keep responses small.
    let round = |value: f32| (f64::from(value) * 1.0e4).round() / 1.0e4;
    let samples = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let x = coordinate(point[0], &format!("$.points[{index}][0]"))?;
            let z = coordinate(point[1], &format!("$.points[{index}][1]"))?;
            let depth = tracks.at(Vec2::new(x, z), seconds, &settings);
            let deform = terrains.deform_at(x, z);
            Ok(json!({
                "track": round(depth.track),
                "rim": round(depth.rim),
                "deform": deform.map(round),
                "depth": deform.map(|deform| round(engine_core::math::min(depth.track, deform))),
                "ground": terrains.height_at(x, z).map(round),
            }))
        })
        .collect::<Result<Vec<_>, DevError>>()?;
    Ok(json!({
        "seconds": seconds,
        "samples": samples,
        "stamps": tracks.len(),
        "dropped": tracks.dropped(),
    }))
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
    let terrains = feature::<Terrains>(world, Feature::Terrain)?;
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
            // meeting placed objects first count, and the
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
                && let Some(stack) = terrains.layers(sample.terrain)
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
