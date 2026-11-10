#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::Res;
use engine_core::{
    AttachPoint, Brick, Bricks, Pass, PassName, PassPurpose, PassViews, SlotSet, Source, Writes,
    never, settings,
};
use engine_render::RenderBricks;

settings! {
    #[section("dither")]
    #[derive(Copy, Eq)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneDither")]
    pub struct Dither {
        pub enabled: bool = true => any,
    }
}

pub(crate) struct DitherPass;

impl PassName for DitherPass {
    const NAME: &'static str = "dither";
    const PASS: Pass = Pass {
        writes: Writes::NOTHING,
        reads: SlotSet::NONE,
        views: PassViews::Every,
        at: AttachPoint::Display,
        purpose: PassPurpose::Scene,
        takes: never,
    };
}

pub struct DitherBrick;

impl Brick for DitherBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|dither: Res<Dither>| dither.enabled)
            .settings::<Dither>()
            .applies::<DitherPass>(shaders::dither::apply::MODULE);
    }
}
