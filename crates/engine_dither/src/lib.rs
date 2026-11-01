#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::Res;
use engine_core::{Brick, Bricks, Source, settings};
use engine_render::PostDisplay;

settings! {
    #[section("dither")]
    #[derive(Copy, Eq)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneDither")]
    pub struct Dither {
        pub enabled: bool = true => any,
    }
}

pub struct DitherBrick;

impl Brick for DitherBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|dither: Res<Dither>| dither.enabled)
            .settings::<Dither>()
            .join::<PostDisplay>((), Some(shaders::dither::apply::MODULE));
    }
}
