#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod antialiasing;
mod brick_smokes;
mod lit;
mod previous;
mod temporal;

use bevy_ecs::prelude::*;
use engine_core::{
    AttachPoint, Brick, Bricks, Pass, PassName, PassViews, Runs, Schedule, SlotSet, Smoke, Source,
    Writes, never,
};
use engine_render::{RenderBricks, Resolve};

pub use antialiasing::{TemporalAntialiasing, TemporalAntialiasingStats};

pub(crate) struct TemporalPass;

impl PassName for TemporalPass {
    const NAME: &'static str = "temporal_lit";
    const PASS: Pass = Pass {
        writes: Writes::NOTHING,
        reads: SlotSet::NONE,
        views: PassViews::Main,
        at: AttachPoint::Post,
        takes: never,
    };
}

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
            .piece::<temporal::TaaPiece>()
            .fill::<Resolve>(Some(shaders::taa::resolve::MODULE))
            .view_target(previous::MOTION)
            .view_target(temporal::HISTORY)
            .pass::<TemporalPass>()
            .systems(Schedule::ViewSync, temporal::publish)
            .resolve_pass(temporal::PASS)
            .part::<lit::LitGpu>();
    }
}
