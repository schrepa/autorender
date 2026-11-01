#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod brick_smokes;

use bevy_ecs::prelude::Res;
use engine_core::{Brick, Bricks, Runs, Smoke, Source, settings};
use engine_places::dither::DitherPlace;
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
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "dither_settings",
        runs: Runs::Gpu,
        run: brick_smokes::dither,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|dither: Res<Dither>| dither.enabled)
            .settings::<Dither>()
            .fill::<DitherPlace>(Some(shaders::dither::faces::dither::MODULE))
            .join::<PostDisplay>((), Some(shaders::dither::apply::MODULE));
    }
}
