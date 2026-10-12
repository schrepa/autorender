//! Deformation's development tools: the inspectors of deformers and the
//! tracks they press.

use engine_core::{Deformation, Engine, Feature, FixedTime, Track, Tracks};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
#[cfg(not(target_arch = "wasm32"))]
use engine_settings::{rule::same, stated};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{EngineControl, feature_mut};
use crate::{DevError, DevTools, InspectorDescriptor, bounded, coordinate};

/// Registers deformation's inspectors.
pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools.register_component::<engine_core::Deformer>(
        InspectorDescriptor::generated(schemas, "Deformer")
            .with_default(json!(engine_core::Deformer::default())),
        |deformer| json!(deformer),
    )
    .expect("built-in descriptor is bounded");
    tools.register_resource::<engine_core::Tracks>(
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

/// Adds the schemas of deformation's inspectors.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert("Deformer".into(), super::schema::<engine_core::Deformer>());
    schemas.insert("Tracks".into(), super::schema::<crate::views::TracksView>());
}

/// Press a track as a deformer would, now, and render
/// while paused: a capsule `radius` meters across from world `from`
/// `[x, z]` to `to` (a footprint at `from` when omitted), sinking ground
/// as deformable as that `depth` meters along its middle, round
/// (`flatness` 0) to flat (1). `clear`
/// forgets every track first. Not a source change: the log and the map
/// keep it until cleared or refilled. Returns the stamp, the stamps the
/// log holds and whether deformation is enabled.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DeformControl {
    pub version: u32,
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

/// Presses `press` into the log as a deformer would at the fixed clock's
/// time; its result.
fn press_track(engine: &mut Engine, press: &DeformControl) -> Result<Value, DevError> {
    let from = [
        coordinate(press.from[0], "$.from[0]")?,
        coordinate(press.from[1], "$.from[1]")?,
    ];
    let to = match press.to {
        Some(to) => [coordinate(to[0], "$.to[0]")?, coordinate(to[1], "$.to[1]")?],
        None => from,
    };
    // A press keeps a deformer's rules.
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
    Ok(json!({
        "temporary": true,
        "track": track,
        "stamps": tracks.len(),
        "dropped": tracks.dropped(),
        "enabled": resource.is_some_and(|settings| settings.enabled),
    }))
}
