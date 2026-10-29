use super::operations::{coordinate, reply, terrains};
use super::*;
use crate::{HydrologySample, PATH_FAR};
use engine_core::bounds::MAX_POINTS;
use engine_core::{Error, ErrorCode, Operation, WaterKind};
#[cfg(feature = "schema")]
use engine_settings::stated;
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainHeightQuery {
    #[cfg_attr(feature = "schema", schemars(length(min = 1, max = MAX_POINTS)))]
    pub points: Vec<[f64; 2]>,
}

impl Operation for TerrainHeightQuery {
    const NAME: &'static str = "terrain_height";
    const CONTROL: bool = false;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        heights(world, &request.points)
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }

    #[cfg(feature = "schema")]
    fn result_schema() -> Option<schemars::Schema> {
        Some(schemars::schema_for!(TerrainHeightResult))
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainRouteQuery {
    pub from: [f64; 2],
    pub to: [f64; 2],
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = route_rule("level")))]
    pub level: Option<u32>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = route_rule("max_grade")))]
    pub max_grade: Option<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = route_rule("grade_cost")))]
    pub grade_cost: Option<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = route_rule("water_cost")))]
    pub water_cost: Option<f64>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = route_rule("turn_cost")))]
    pub turn_cost: Option<f64>,
}

impl Operation for TerrainRouteQuery {
    const NAME: &'static str = "terrain_route";
    const CONTROL: bool = false;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        let Self {
            from,
            to,
            level,
            max_grade,
            grade_cost,
            water_cost,
            turn_cost,
        } = request;
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
            Error::new(
                ErrorCode::OutOfRange,
                format!("$.{}", invalid.field),
                invalid.rule,
            )
        })?;
        let world = &*world;
        let terrains = terrains(world)?;
        let Some(sample) = terrains.sample(from.x, from.y) else {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                "$.from",
                "no resident terrain lies under from",
            ));
        };
        let Some(terrain) = terrains.terrain(sample.terrain) else {
            return Err(Error::new(
                ErrorCode::InvalidRequest,
                "$.from",
                "no terrain lies under from",
            ));
        };
        let origin = Vec2::new(terrain.origin.x, terrain.origin.z);
        let route = terrains.route(sample.terrain, from, to, &options);
        Ok(reply(TerrainRouteResult {
            terrain: terrain_key(world, sample.terrain),
            options,
            origin: origin.to_array(),
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

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }

    #[cfg(feature = "schema")]
    fn result_schema() -> Option<schemars::Schema> {
        Some(schemars::schema_for!(TerrainRouteResult))
    }
}

#[cfg(feature = "schema")]
fn route_rule(field: &'static str) -> impl Fn(&mut schemars::Schema) {
    let state = stated(RouteOptions::rule_of(field).expect("route options declare the field"));
    let default = serde_json::to_value(RouteOptions::default())
        .expect("route options serialize")[field]
        .clone();
    move |schema| {
        state(schema);
        schema.insert("default".into(), default.clone());
    }
}

fn heights(world: &World, points: &[[f64; 2]]) -> Result<Value, Error> {
    if points.is_empty() || points.len() > MAX_POINTS {
        return Err(Error::new(
            ErrorCode::OutOfRange,
            "$.points",
            format!("name 1 to {MAX_POINTS} points"),
        ));
    }
    let terrains = terrains(world)?;
    let samples = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let x = coordinate(point[0], &format!("$.points[{index}][0]"))?;
            let z = coordinate(point[1], &format!("$.points[{index}][1]"))?;
            Ok(terrains.sample(x, z).map(|sample| {
                let found = terrains.layers(sample.terrain).and_then(|layers| {
                    let terrain = terrains.terrain(sample.terrain)?;
                    let water = layers.hydrology()?;
                    let local = (x - terrain.origin.x, z - terrain.origin.z);
                    Some(water.at(local.0, local.1, terrain.pyramid.spacing(0)))
                });
                let standing = terrains
                    .water_at(Vec3::new(x, sample.height, z))
                    .filter(|water| water.kind != WaterKind::Fall)
                    .map(|water| StandingWater {
                        surface: water.surface,
                        depth: water.depth,
                    });
                let path = terrains.layers(sample.terrain).and_then(|layers| {
                    let terrain = terrains.terrain(sample.terrain)?;
                    let road = layers.path_at(x - terrain.origin.x, z - terrain.origin.z);
                    (road.distance < PATH_FAR).then_some(road.edge())
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
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(reply(TerrainHeightResult {
        samples,
        stats: terrains.stats(),
    }))
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainHeightResult {
    pub samples: Vec<Option<HeightSample>>,
    pub stats: TerrainStats,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct HeightSample {
    pub terrain: TerrainKey,
    pub height: f32,
    pub normal: [f32; 3],
    pub hole: bool,
    pub level: u32,
    pub water: Option<HydrologySample>,
    pub standing_water: Option<StandingWater>,
    pub path: Option<f32>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct StandingWater {
    pub surface: f32,
    pub depth: f32,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainRouteResult {
    pub terrain: TerrainKey,
    pub options: RouteOptions,
    pub origin: [f32; 2],
    pub route: Option<RouteFound>,
}

#[derive(Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RouteFound {
    pub points: Vec<[f64; 2]>,
    pub length: f32,
    pub steepest: f32,
    pub hairpin_steps: usize,
}
