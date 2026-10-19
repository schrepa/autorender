//! Terrain's queries: heights at points, a sweep comparing what terrain
//! queries, collision rays and generated terrains' layers say the ground
//! is, and routes for paths.

use engine_collision::{CollisionRay, CollisionWorld, QueryFilter, refresh_collision_world};
use engine_core::{
    Engine, Feature, RouteOptions, Terrain, Terrains, WaterKind,
    bevy_ecs::prelude::World,
    glam::{Vec2, Vec3},
};
use engine_scene::{SceneEntity, SceneInstance};
use engine_settings::ErrorCode;
#[cfg(not(target_arch = "wasm32"))]
use engine_settings::stated;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::results::{
    HeightSample, RouteFound, StandingWater, SweepExample, SweepLayers, SweepRays,
    TerrainHeightResult, TerrainRouteResult, TerrainSweepResult, Worst,
};
use crate::features::{EngineQuery, feature};
use crate::protocol::bounds::{self, MAX_POINTS};
use crate::results::result;
use crate::{DevError, bounded, coordinate};

/// Hits a sweep ray may meet: placed objects above the
/// ground, and the ground itself.
const PLACED_RAY_HITS: usize = 16;

/// A terrain as replies name it: its authored ID, or its entity's bits
/// outside scenes.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum TerrainKey {
    /// Its authored ID.
    Authored(String),
    /// Its entity's bits, outside scenes.
    Entity(u64),
}

/// The terrain `entity` as replies name it.
pub(crate) fn terrain_key(
    world: &World,
    entity: engine_core::bevy_ecs::prelude::Entity,
) -> TerrainKey {
    world.get::<SceneEntity>(entity).map_or_else(
        || TerrainKey::Entity(entity.to_bits()),
        |source| TerrainKey::Authored(source.authored_id.clone()),
    )
}

/// Terrain surfaces under world points `[x, z]`: each one's terrain, height, normal, hole and the level of the
/// resident tile answering, what its hydrology bake found there and the
/// water standing there, and the meters from the
/// nearest road's edge (negative on roads, null beyond
/// their masks' reach), or null where no tile is resident.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainHeightQuery {
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = MAX_POINTS)))]
    pub points: Vec<[f64; 2]>,
}

impl EngineQuery for TerrainHeightQuery {
    fn answer(&self, engine: &mut Engine, _: Option<&SceneInstance>) -> Result<Value, DevError> {
        heights(engine, &self.points)
    }
}

/// Compares, over `samples` by `samples` points across `half_extent`
/// meters of world `center` `[x, z]`, what terrain queries (the tiles
/// renderers draw), collision rays and generated terrains' layers say
/// the ground is: the largest differences and where.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainSweepQuery {
    pub center: [f64; 2],
    #[serde(default = "default_half_extent")]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::SWEEP_EXTENT)))]
    pub half_extent: f64,
    #[serde(default = "default_sweep_samples")]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(bounds::SWEEP_SAMPLES)))]
    pub samples: u32,
}

impl EngineQuery for TerrainSweepQuery {
    fn answer(&self, engine: &mut Engine, _: Option<&SceneInstance>) -> Result<Value, DevError> {
        let &Self {
            center,
            half_extent,
            samples,
        } = self;
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
}

fn default_half_extent() -> f64 {
    64.0
}

fn default_sweep_samples() -> u32 {
    16
}

/// The cheapest route over the generated terrain under world `from`
/// `[x, z]` to world `to`, over its layers with their
/// bakes, as a path layer would follow it: walking between the samples
/// of pyramid `level`, each step's cost growing by `grade_cost` times
/// its squared grade and by `water_cost` where water stands, each turn
/// costing `turn_cost` meters per unit of one less its angle's cosine,
/// refusing grades steeper than `max_grade` (rise per meter).
/// Returns the terrain's origin `[x, z]` and the route's points
/// terrain-local (a path layer's `points`; add the origin for world
/// points), its length, its steepest grade within the limit and the
/// turning steps of its hairpins crossing steeper ground (up to 45
/// degrees), or a null route where none keeps within the limits. Its
/// work grows with the square of the ends' distance.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainRouteQuery {
    pub from: [f64; 2],
    pub to: [f64; 2],
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("level")))]
    pub level: Option<u32>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("max_grade")))]
    pub max_grade: Option<f64>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("grade_cost")))]
    pub grade_cost: Option<f64>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("water_cost")))]
    pub water_cost: Option<f64>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = route_rule("turn_cost")))]
    pub turn_cost: Option<f64>,
}

impl EngineQuery for TerrainRouteQuery {
    fn answer(&self, engine: &mut Engine, _: Option<&SceneInstance>) -> Result<Value, DevError> {
        let &Self {
            from,
            to,
            level,
            max_grade,
            grade_cost,
            water_cost,
            turn_cost,
        } = self;
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
        Ok(result::terrain_route(TerrainRouteResult {
            terrain: terrain_key(world, sample.terrain),
            options,
            origin: origin.to_array(),
            // Centimeters are plenty, and keep long routes within the
            // response limit.
            route: route.map(|route| RouteFound {
                points: route
                    .points
                    .iter()
                    .map(|point| point.map(|value| (f64::from(value) * 100.0).round() / 100.0))
                    .collect(),
                length: route.length,
                steepest: route.steepest,
                hairpin_steps: route.hairpin_steps,
            }),
        }))
    }
}

/// The schema transform stating route option `field`'s rule on the request
/// field of that name.
#[cfg(not(target_arch = "wasm32"))]
fn route_rule(field: &'static str) -> impl Fn(&mut schemars::Schema) {
    let state =
        stated(engine_core::RouteOptions::rule_of(field).expect("route options declare the field"));
    let default = serde_json::to_value(engine_core::RouteOptions::default())
        .expect("route options serialize")[field]
        .clone();
    move |schema| {
        state(schema);
        schema.insert("default".into(), default.clone());
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
            Ok(terrains.sample(x, z).map(|sample| {
                // What its hydrology bake found there, as the finest level
                // sees it.
                let found = terrains.layers(sample.terrain).and_then(|layers| {
                    let terrain = terrains.terrain(sample.terrain)?;
                    let water = layers.hydrology()?;
                    let local = (x - terrain.origin.x, z - terrain.origin.z);
                    Some(water.at(local.0, local.1, terrain.pyramid.spacing(0)))
                });
                // The water standing there as the tiles hold it, over the
                // ground they hold: its surface (world height) and depth.
                let standing = terrains
                    .water_at(Vec3::new(x, sample.height, z))
                    .filter(|water| water.kind != WaterKind::Fall)
                    .map(|water| StandingWater {
                        surface: water.surface,
                        depth: water.depth,
                    });
                // Meters from the nearest road's edge,
                // where a path's masks reach.
                let path = terrains.layers(sample.terrain).and_then(|layers| {
                    let terrain = terrains.terrain(sample.terrain)?;
                    let road = layers.path_at(x - terrain.origin.x, z - terrain.origin.z);
                    (road.distance < engine_core::terrain_data::PATH_FAR).then_some(road.edge())
                });
                HeightSample {
                    terrain: terrain_key(world, sample.terrain),
                    height: sample.height,
                    normal: sample.normal.to_array(),
                    hole: sample.hole,
                    level: sample.level,
                    water: found,
                    standing_water: standing,
                    path,
                }
            }))
        })
        .collect::<Result<Vec<_>, DevError>>()?;
    Ok(result::terrain_height(TerrainHeightResult {
        samples,
        stats: terrains.stats(),
    }))
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
        refresh_collision_world(engine.app.world_mut())
            .map_err(crate::features::collision::native_error)?;
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
                    .map_err(crate::features::collision::native_error)?;
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
                collected.push(SweepExample {
                    at: point.to_array(),
                    height: sample.height,
                });
            }
        }
    }
    Ok(result::terrain_sweep(TerrainSweepResult {
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
        layers: SweepLayers {
            compared: layer_checks,
            worst: layers,
        },
        examples: collected,
    }))
}
