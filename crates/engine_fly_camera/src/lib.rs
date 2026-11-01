include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod camera;
mod input;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Runs, Schedule, Smoke, Source, settings};

settings! {
    #[section("fly_camera")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneFlyCamera")]
    pub struct FlyCamera {
        pub enabled: bool = true => any,
        pub speed: f32 = 22.0 => range(0.0, 1000.0),
        pub steered_only: bool = false => any,
    }
}

pub struct FlyCameraBrick;

impl Brick for FlyCameraBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "input_camera",
        runs: Runs::Gpu,
        run: input::input_camera,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|fly: Res<FlyCamera>| fly.enabled)
            .settings::<FlyCamera>()
            .systems(Schedule::Presentation, camera::fly);
    }
}
