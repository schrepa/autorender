#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod antialiasing;
mod previous;
mod resolve;
mod temporal;

use engine_core::{Brick, Bricks, Jitter, Schedule, Source};
use engine_render::{RenderBricks, Resolve};

pub use antialiasing::{TemporalAntialiasing, TemporalAntialiasingStats};

pub struct TaaBrick;

impl Brick for TaaBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .switch::<TemporalAntialiasing>()
            .stats::<TemporalAntialiasingStats>()
            .piece::<temporal::TaaPiece>()
            .resource::<temporal::TaaSettling>()
            .generated(temporal::GENERATED)
            .fill::<Resolve>(Some(shaders::taa::resolved::MODULE))
            .fill::<Jitter>(None)
            .view_target(previous::MOTION)
            .systems(Schedule::WorldSync, temporal::jitter)
            .systems(Schedule::ViewSync, temporal::publish)
            .part::<resolve::ResolveGpu>();
    }
}
