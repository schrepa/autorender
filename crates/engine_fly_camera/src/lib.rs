#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod camera;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source, Verbs, settings};

settings! {
    #[section("fly_camera")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneFlyCamera")]
    pub struct FlyCamera {
        pub enabled: bool = true => any,
        pub speed: f32 = 22.0 => range(0.0, 1000.0),
        pub look_sensitivity: f32 = 0.003 => range(0.0, 1.0),
        pub turn_speed: f32 = 1.8 => range(0.0, 100.0),
    }
}

pub struct FlyCameraBrick;

impl Brick for FlyCameraBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|fly: Res<FlyCamera>| fly.enabled)
            .settings::<FlyCamera>()
            .resource::<camera::Rig>()
            .join::<Verbs>(camera::MOVE, None)
            .join::<Verbs>(camera::RISE, None)
            .join::<Verbs>(camera::LOOK, None)
            .join::<Verbs>(camera::TURN, None)
            .systems(Schedule::Presentation, camera::fly);
    }
}
