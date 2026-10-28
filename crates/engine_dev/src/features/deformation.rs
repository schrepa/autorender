use engine_core::{
    Deformation, Engine, Feature, FixedTime, PresentedTime, Terrains, Track, Tracks, glam::Vec2,
};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
#[cfg(not(target_arch = "wasm32"))]
use engine_settings::{rule::same, stated};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{EngineControl, EngineQuery, feature, feature_mut};
use crate::protocol::bounds::MAX_POINTS;
use crate::results::result;
use crate::{DevError, DevTools, InspectorDescriptor, bounded, coordinate};

pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools
        .register_component::<engine_core::Deformer>(
            InspectorDescriptor::generated(schemas, "Deformer")
                .with_default(json!(engine_core::Deformer::default())),
            |deformer| json!(deformer),
        )
        .expect("built-in descriptor is bounded");
    tools
        .register_resource::<engine_core::Tracks>(
            InspectorDescriptor::generated(schemas, "Tracks"),
            |tracks| {
                json!(crate::views::TracksView {
                    stamps: tracks.len(),
                    pressed: tracks.pressed(),
                    dropped: tracks.dropped(),
                    latest: tracks.latest().copied(),
                })
            },
        )
        .expect("built-in descriptor is bounded");
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert("Deformer".into(), super::schema::<engine_core::Deformer>());
    schemas.insert("Tracks".into(), super::schema::<crate::views::TracksView>());
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DeformControl {
    pub from: [f64; 2],
    #[serde(default)]
    pub to: Option<[f64; 2]>,
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::Deformer::rule_of("radius")))))]
    pub radius: f64,
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::Deformer::rule_of("depth")))))]
    pub depth: f64,
    #[serde(default = "default_flatness")]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::Deformer::rule_of("flatness")))))]
    pub flatness: f64,
    #[serde(default)]
    pub clear: bool,
}

impl EngineControl for DeformControl {
    fn changed(&self) -> &'static str {
        "track pressed"
    }

    fn apply(&self, engine: &mut Engine, _: Option<&SceneInstance>) -> Result<Value, DevError> {
        press_track(engine, self)
    }
}

fn default_flatness() -> f64 {
    0.5
}

fn press_track(engine: &mut Engine, press: &DeformControl) -> Result<Value, DevError> {
    let from = [
        coordinate(press.from[0], "$.from[0]")?,
        coordinate(press.from[1], "$.from[1]")?,
    ];
    let to = match press.to {
        Some(to) => [coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?],
        None => from,
    };
    for (value, field) in [
        (press.radius, "radius"),
        (press.depth, "depth"),
        (press.flatness, "flatness"),
    ] {
        let rule = engine_core::Deformer::rule_of(field).expect("deformers declare the field");
        bounded(value, rule, ErrorCode::OutOfRange, &format!("$.{field}"))?;
    }
    let world = engine.app.world_mut();
    let seconds = world.resource::<FixedTime>().elapsed_seconds;
    let resource = world.get_resource::<Deformation>().copied();
    let settings = resource.unwrap_or_default().sanitized();
    let track = Track {
        from,
        to,
        radius: press.radius as f32,
        depth: press.depth as f32,
        flatness: press.flatness as f32,
        seconds,
    };
    let mut tracks = feature_mut::<Tracks>(world, Feature::Deformation)?;
    if press.clear {
        tracks.clear();
    }
    tracks.press(track, &settings);
    Ok(result::deform(DeformResult {
        temporary: true,
        track,
        stamps: tracks.len(),
        dropped: tracks.dropped(),
        enabled: resource.is_some_and(|settings| settings.enabled),
    }))
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DeformResult {
    pub temporary: bool,
    pub track: Track,
    pub stamps: usize,
    pub dropped: u64,
    pub enabled: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DeformationQueryResult {
    pub seconds: f64,
    pub samples: Vec<DepthSample>,
    pub stamps: usize,
    pub dropped: u64,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DepthSample {
    pub track: f64,
    pub rim: f64,
    pub deform: Option<f64>,
    pub depth: Option<f64>,
    pub ground: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DeformationQuery {
    #[cfg_attr(not(target_arch = "wasm32"), schemars(length(min = 1, max = MAX_POINTS)))]
    pub points: Vec<[f64; 2]>,
    #[serde(default)]
    pub seconds: Option<f64>,
}

impl EngineQuery for DeformationQuery {
    fn answer(&self, engine: &mut Engine, _: Option<&SceneInstance>) -> Result<Value, DevError> {
        tracks(engine, &self.points, self.seconds)
    }
}

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
    let round = |value: f32| (f64::from(value) * 1.0e4).round() / 1.0e4;
    let samples = points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let x = coordinate(point[0], &format!("$.points[{index}][0]"))?;
            let z = coordinate(point[1], &format!("$.points[{index}][1]"))?;
            let depth = tracks.at(Vec2::new(x, z), seconds, &settings);
            let deform = terrains.deform_at(x, z);
            Ok(DepthSample {
                track: round(depth.track),
                rim: round(depth.rim),
                deform: deform.map(round),
                depth: deform.map(|deform| round(engine_core::math::min(depth.track, deform))),
                ground: terrains.height_at(x, z).map(round),
            })
        })
        .collect::<Result<Vec<_>, DevError>>()?;
    Ok(result::deformation_query(DeformationQueryResult {
        seconds,
        samples,
        stamps: tracks.len(),
        dropped: tracks.dropped(),
    }))
}
