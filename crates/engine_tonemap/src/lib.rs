#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod light;
mod tonemap;

use engine_core::{Brick, Bricks, Schedule, Source};
use engine_places::tonemap::Display;

pub use light::Grading;

pub struct TonemapBrick;

impl Brick for TonemapBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<Grading>()
            .piece::<Display>()
            .fill::<Display>(Some(shaders::tonemap::faces::tonemap::MODULE))
            .systems(Schedule::ViewSync, tonemap::publish_display);
    }
}
