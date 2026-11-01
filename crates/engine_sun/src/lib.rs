#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod light;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source};
use engine_places::sun::{Sun, SunPath};

pub use light::Sunlight;

pub struct SunBrick;

impl Brick for SunBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<Sunlight>()
            .piece::<Sun>()
            .fill::<Sun>(Some(shaders::sun::faces::sun::MODULE))
            .systems(Schedule::ViewSync, publish_sun);
    }
}

fn publish_sun(light: Res<Sunlight>, path: Res<SunPath>, mut sun: ResMut<Sun>) {
    let values = light.values();
    let direction = path.direction.unwrap_or(values.sun_direction).normalize();
    sun.set_if_neq(Sun {
        direction: direction.extend(0.0).to_array(),
        color: values.sun_color.extend(0.0).to_array(),
    });
}
