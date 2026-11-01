#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod antialiasing;
mod brick_smokes;
mod previous;
mod temporal;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Runs, Schedule, Smoke, Source};
use engine_render::{RenderBricks, Resolve, ViewOutputs};

pub use antialiasing::{TemporalAntialiasing, TemporalAntialiasingStats};

pub struct TaaBrick;

impl Brick for TaaBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "temporal_antialiasing",
        runs: Runs::Gpu,
        run: brick_smokes::temporal_antialiasing,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|taa: Res<TemporalAntialiasing>| taa.enabled)
            .settings::<TemporalAntialiasing>()
            .stats::<TemporalAntialiasingStats>()
            .resource::<previous::History>()
            .piece::<temporal::TaaPiece>()
            .fill::<Resolve>(Some(shaders::taa::resolve::MODULE))
            .join::<ViewOutputs>(previous::MOTION, Some(shaders::taa::motion::MODULE))
            .join::<ViewOutputs>(temporal::HISTORY, Some(shaders::taa::history::MODULE))
            .systems(Schedule::ViewSync, temporal::publish)
            .off(previous::forget)
            .resolve_pass(temporal::PASS);
    }
}
