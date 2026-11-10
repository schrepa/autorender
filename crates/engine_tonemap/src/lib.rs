#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod tonemap;

use engine_core::{
    AttachPoint, Brick, Bricks, Pass, PassName, PassPurpose, PassViews, Schedule, SlotSet, Source,
    Writes, never,
};
use engine_places_rendering::tonemap::Display;
use engine_render::RenderBricks;

pub use tonemap::Grading;

pub(crate) struct TonemapPass;

impl PassName for TonemapPass {
    const NAME: &'static str = "tonemap";
    const PASS: Pass = Pass {
        writes: Writes::NOTHING,
        reads: SlotSet::NONE,
        views: PassViews::Every,
        at: AttachPoint::Display,
        purpose: PassPurpose::Scene,
        takes: never,
    };
}

pub struct TonemapBrick;

impl Brick for TonemapBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<Grading>()
            .piece::<Display>()
            .fill::<Display>(Some(shaders::tonemap::faces::tonemap::MODULE))
            .applies::<TonemapPass>(shaders::tonemap::tone::MODULE)
            .systems(Schedule::ViewSync, tonemap::publish_display);
    }
}
