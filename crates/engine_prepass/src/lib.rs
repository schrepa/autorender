#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use engine_core::{
    AttachPoint, Brick, Bricks, Pass, PassName, PassViews, SlotSet, Source, Writes, every,
};
use engine_render::RenderBricks;

pub(crate) struct PrepassPass;

impl PassName for PrepassPass {
    const NAME: &'static str = "prepass";
    const PASS: Pass = Pass {
        writes: Writes::DEPTH,
        reads: SlotSet::MASK,
        views: PassViews::Main,
        at: AttachPoint::Depth,
        takes: every,
    };
}

pub struct PrepassBrick;

impl Brick for PrepassBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks.pass::<PrepassPass>();
    }
}
