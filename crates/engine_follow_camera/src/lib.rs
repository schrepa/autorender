#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod camera;
mod smoke;

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Runs, SceneComponents, Schedule, Smoke, Source, settings,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

settings! {
    #[section("follow_camera")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneFollowCamera")]
    pub struct FollowCamera {
        pub enabled: bool = true => any,
        pub distance: f32 = 6.0 => range(0.1, 1000.0),
        pub height: f32 = 1.5 => range(-100.0, 100.0),
        pub pitch_min: f32 = -1.2 => range(-1.5, 1.5),
        pub pitch_max: f32 = 0.8 => range(-1.5, 1.5),
        pub smoothing: f32 = 0.1 => range(0.0, 10.0),
        pub collision_radius: f32 = 0.25 => range(0.01, 10.0),
    }
}

#[derive(
    Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[schemars(rename = "SceneCameraTarget")]
#[serde(deny_unknown_fields)]
pub struct CameraTarget {}

pub struct FollowCameraBrick;

impl Brick for FollowCameraBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "follow_camera",
        runs: Runs::Gpu,
        run: smoke::run,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|follow: Res<FollowCamera>| follow.enabled)
            .settings::<FollowCamera>()
            .resource::<camera::Orbit>()
            .resource::<camera::Shake>()
            .join::<SceneComponents>(
                ComponentKind::of::<CameraTarget>("camera_target")
                    .schema(target_schema)
                    .example(target_example),
                None,
            )
            .systems(Schedule::Presentation, camera::follow)
            .operation::<camera::CameraShake>();
    }
}

fn target_schema() -> Value {
    serde_json::to_value(schemars::schema_for!(CameraTarget)).unwrap_or(Value::Null)
}

fn target_example() -> Value {
    json!({})
}
