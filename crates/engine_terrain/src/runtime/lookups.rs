use super::operations::{coordinate, reply};
use super::*;
use crate::{HydrologySample, PATH_FAR};
use engine_core::Operation;
use engine_core::bounds::MAX_POINTS;
use engine_places::water::{WaterBody, WaterKind, WaterSample};
use serde_json::Value;

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainHeightQuery {
    #[schemars(length(min = 1, max = MAX_POINTS))]
    pub points: Vec<[f64; 2]>,
}

impl Operation for TerrainHeightQuery {
    const NAME: &'static str = "terrain_height";
    type Params = (
        Res<'static, Terrains>,
        Res<'static, TerrainStats>,
        Query<'static, 'static, &'static WaterBody>,
    );
    type Request = Self;

    fn run(
        (terrains, stats, bodies): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        heights((&terrains, &stats, &bodies), &request.points)
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainRouteQuery {
    pub from: [f64; 2],
    pub to: [f64; 2],
    #[serde(default)]
    pub level: Option<u32>,
    #[serde(default)]
    pub max_grade: Option<f64>,
    #[serde(default)]
    pub grade_cost: Option<f64>,
    #[serde(default)]
    pub water_cost: Option<f64>,
    #[serde(default)]
    pub turn_cost: Option<f64>,
}

impl Operation for TerrainRouteQuery {
    const NAME: &'static str = "terrain_route";
    type Params = Res<'static, Terrains>;
    type Request = Self;

    fn run(terrains: SystemParamItem<'_, '_, Self::Params>, request: Self) -> Result<Value, Error> {
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
        options.validate().map_err(|invalid| Error {
            code: ErrorCode::OutOfRange,
            ..Error::from(invalid)
        })?;
        let Some(sample) = terrains.sample(from.x, from.y) else {
            return Err(Error::new(ErrorCode::InvalidRequest, "$.from").got(from.to_array()));
        };
        let Some(terrain) = terrains.terrain(sample.terrain) else {
            return Err(Error::new(ErrorCode::InvalidRequest, "$.from").got(from.to_array()));
        };
        let origin = Vec2::new(terrain.origin.x, terrain.origin.z);
        let route = terrains.route(sample.terrain, from, to, &options);
        Ok(reply(TerrainRouteResult {
            terrain: terrain_key(sample.terrain),
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
}

fn water_at(bodies: &Query<'_, '_, &WaterBody>, point: Vec3) -> Option<WaterSample> {
    let mut held: Vec<&WaterBody> = bodies.iter().collect();
    held.sort_by_key(|body| std::cmp::Reverse(body.precedence));
    held.iter().find_map(|body| body.shape.water(point))
}

fn heights(
    (terrains, stats, bodies): (&Terrains, &TerrainStats, &Query<'_, '_, &WaterBody>),
    points: &[[f64; 2]],
) -> Result<Value, Error> {
    if points.is_empty() || points.len() > MAX_POINTS {
        return Err(Error::new(ErrorCode::OutOfRange, "$.points")
            .range(1.0, MAX_POINTS as f64)
            .got(points.len()));
    }
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
                let standing = water_at(bodies, Vec3::new(x, sample.height, z))
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
                    terrain: terrain_key(sample.terrain),
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
        stats: stats.clone(),
    }))
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct TerrainHeightResult {
    pub samples: Vec<Option<HeightSample>>,
    pub stats: TerrainStats,
}

#[derive(Serialize, schemars::JsonSchema)]
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

#[derive(Serialize, schemars::JsonSchema)]
pub struct StandingWater {
    pub surface: f32,
    pub depth: f32,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct TerrainRouteResult {
    pub terrain: TerrainKey,
    pub options: RouteOptions,
    pub origin: [f32; 2],
    pub route: Option<RouteFound>,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct RouteFound {
    pub points: Vec<[f64; 2]>,
    pub length: f32,
    pub steepest: f32,
    pub hairpin_steps: usize,
}
