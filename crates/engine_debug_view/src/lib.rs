#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod gpu;
mod modes;
mod probe;
mod settings;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Schedule, Source};
use engine_render::{DebugModes, DebugShown, RenderBricks};

pub use settings::DebugView;

use gpu::DebugViewPiece;
use probe::{Probe, ProbeGpu, ProbePixel, ProbedPixel};

pub struct DebugViewBrick;

impl Brick for DebugViewBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(shows)
            .settings::<DebugView>()
            .resource::<Probe>()
            .piece::<DebugViewPiece>()
            .fill::<DebugShown>(Some(shaders::debug_view::shown::MODULE));
        for (index, (name, _)) in modes::OWN.into_iter().enumerate() {
            let wgsl = (index == 0).then_some(shaders::debug_view::modes::MODULE);
            bricks.join::<DebugModes>(modes::member(name), wgsl);
        }
        bricks
            .systems(Schedule::ViewSync, (probe::advance, modes::publish))
            .operation::<ProbePixel>()
            .observation::<ProbedPixel>()
            .part::<ProbeGpu>();
    }
}

fn shows(view: Res<DebugView>, probe: Res<Probe>) -> bool {
    view.showing() || probe.armed()
}
