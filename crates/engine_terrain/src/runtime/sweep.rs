use super::operations::{coordinate, reply, within};
use super::*;
use engine_core::{Filled, Operation, stated};
use engine_places_simulation::physics::{Collision, CollisionRay, QueryFilter};
use serde_json::Value;

const PLACED_RAY_HITS: usize = 16;
const SWEEP_EXTENT: Rule = Rule::Range(0.0, 4096.0);
const SWEEP_SAMPLES: Rule = Rule::Range(2.0, 64.0);

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TerrainSweepQuery {
    pub center: [f64; 2],
    #[serde(default = "default_half_extent")]
    #[schemars(transform = stated(SWEEP_EXTENT))]
    pub half_extent: f64,
    #[serde(default = "default_sweep_samples")]
    #[schemars(transform = stated(SWEEP_SAMPLES))]
    pub samples: u32,
}

fn default_half_extent() -> f64 {
    64.0
}

fn default_sweep_samples() -> u32 {
    16
}

impl Operation for TerrainSweepQuery {
    const NAME: &'static str = "terrain_sweep";
    type Params = (
        Res<'static, Terrains>,
        Res<'static, Collision>,
        Filled<'static, Collision>,
    );
    type Request = Self;

    fn run(
        (terrains, collision, collides): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let center = Vec2::new(
            coordinate(request.center[0], "$.center[0]")?,
            coordinate(request.center[1], "$.center[1]")?,
        );
        let half_extent = within(request.half_extent, SWEEP_EXTENT, "$.half_extent")?;
        within(
            f64::from(request.samples),
            SWEEP_SAMPLES,
            "$.samples",
        )?;
        sweep(
            &terrains,
            collides.on().then_some(&*collision),
            (center, half_extent as f32),
            request.samples,
        )
        .map(reply)
    }

    fn schema() -> Value {
        engine_heightfield::schema::<Self>()
    }

    fn result_schema() -> Option<Value> {
        Some(engine_heightfield::schema::<TerrainSweepResult>())
    }
}

fn sweep(
    terrains: &Terrains,
    collision: Option<&Collision>,
    (center, half_extent): (Vec2, f32),
    samples: u32,
) -> Result<TerrainSweepResult, Error> {
    let grounds: Vec<Entity> = terrains
        .entries
        .iter()
        .filter(|state| state.around.is_none())
        .map(|state| state.entity)
        .collect();
    let mut queried = 0;
    let mut level_zero = 0;
    let mut holes = 0;
    let mut rays = Worst::default();
    let (mut ray_checks, mut ray_misses, mut placed_first) = (0, 0, 0);
    let mut collected = Vec::new();
    let mut hits = Vec::new();
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
            if sample.level != 0 {
                continue;
            }
            level_zero += 1;
            if let Some(collision) = collision {
                collision.0.raycast_all(
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
                let hit = hits.iter().find(|hit| hit.placed.is_none());
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
    Ok(TerrainSweepResult {
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
    })
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
