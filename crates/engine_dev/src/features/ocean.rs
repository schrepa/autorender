use engine_core::{Engine, Floating, Ground, Ocean, PresentedTime};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::EngineQuery;
use crate::protocol::bounds::MAX_POINTS;
use crate::results::result;
use crate::{DevError, DevTools, InspectorDescriptor, coordinate};

pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools
        .register_component::<Floating>(
            InspectorDescriptor::generated(schemas, "Floating")
                .with_default(json!(Floating::default())),
            |floating| json!(floating),
        )
        .expect("built-in descriptor is bounded");
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert("Floating".into(), super::schema::<Floating>());
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct WaterQuery {
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = MAX_POINTS)))]
    pub points: Vec<[f64; 2]>,
    #[serde(default)]
    pub seconds: Option<f64>,
}

impl EngineQuery for WaterQuery {
    fn answer(
        &self,
        engine: &mut Engine,
        _scene: Option<&SceneInstance>,
    ) -> Result<Value, DevError> {
        let points = &self.points;
        if points.is_empty() || points.len() > MAX_POINTS {
            return Err(DevError::new(
                ErrorCode::OutOfRange,
                "$.points",
                format!("name 1 to {MAX_POINTS} points"),
            ));
        }
        let world = engine.app.world();
        let seconds = match self.seconds {
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
        let waves = world
            .get_resource::<Ocean>()
            .filter(|ocean| ocean.enabled)
            .map(|ocean| ocean.spectrum());
        let samples = points
            .iter()
            .enumerate()
            .map(|(index, point)| {
                let x = coordinate(point[0], &format!("$.points[{index}][0]"))?;
                let z = coordinate(point[1], &format!("$.points[{index}][1]"))?;
                let surface = waves
                    .as_ref()
                    .map(|waves| waves.surface(point[0], point[1], seconds));
                let ground = Ground::sample_in(world, x, z).map(|sample| sample.height);
                let round = |value: f64, step: f64| (value / step).round() * step;
                let water = engine_core::WaterSources::sample_in(
                    world,
                    engine_core::glam::Vec3::new(x, ground.unwrap_or(0.0), z),
                )
                .map(|water| WorldWater {
                    kind: water.kind,
                    surface: round(f64::from(water.surface), 1.0e-3),
                    depth: round(f64::from(water.depth), 1.0e-3),
                    velocity: water.velocity.map(|value| round(f64::from(value), 1.0e-3)),
                    foam: round(f64::from(water.foam), 1.0e-3),
                });
                Ok(WaterSample {
                    ocean: surface.map(|surface| SeaSurface {
                        height: round(f64::from(surface.height), 1.0e-3),
                        normal: surface.normal.map(|value| round(f64::from(value), 1.0e-4)),
                        compression: round(f64::from(surface.compression), 1.0e-4),
                        still: surface.still.map(|value| round(value, 1.0e-3)),
                    }),
                    ground: ground.map(|ground| round(f64::from(ground), 1.0e-3)),
                    depth: surface
                        .zip(ground)
                        .map(|(surface, ground)| round(f64::from(surface.height - ground), 1.0e-3)),
                    water,
                })
            })
            .collect::<Result<Vec<_>, DevError>>()?;
        Ok(result::water_query(WaterQueryResult {
            seconds,
            samples,
            level: waves.as_ref().map(|waves| waves.level),
            amplitude: waves.as_ref().map(|waves| waves.amplitude()),
            waves: waves.as_ref().map(|waves| waves.waves().to_vec()),
        }))
    }
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WaterQueryResult {
    pub seconds: f64,
    pub samples: Vec<WaterSample>,
    pub level: Option<f32>,
    pub amplitude: Option<f32>,
    pub waves: Option<Vec<engine_core::OceanWave>>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WaterSample {
    pub ocean: Option<SeaSurface>,
    pub ground: Option<f64>,
    pub depth: Option<f64>,
    pub water: Option<WorldWater>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WorldWater {
    pub kind: engine_core::WaterKind,
    pub surface: f64,
    pub depth: f64,
    pub velocity: [f64; 3],
    pub foam: f64,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SeaSurface {
    pub height: f64,
    pub normal: [f64; 3],
    pub compression: f64,
    pub still: [f64; 2],
}
