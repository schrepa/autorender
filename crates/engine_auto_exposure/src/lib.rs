#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants,
    clippy::iter_over_hash_type,
    clippy::expect_used
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod gpu_layouts;
mod meter;
mod settings;

use std::mem::offset_of;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Readback, Schedule, Source};
use engine_places::exposure::Exposure;
use engine_render::{DebugShown, RenderBricks};

use crate::gpu_layouts::ExposureState;

pub use settings::{AutoExposure, ExposureStats};

pub struct AutoExposureBrick;

impl Brick for AutoExposureBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|exposure: Res<AutoExposure>| exposure.enabled)
            .settings::<AutoExposure>()
            .stats::<ExposureStats>()
            .resource::<ExposureFeedback>()
            .fill::<Exposure>(Some(shaders::auto_exposure::faces::exposure::MODULE))
            .systems(Schedule::ViewSync, exposure_stats)
            .off(forget_exposure)
            .part::<meter::MeterGpu>();
    }
}

#[derive(Resource, Default)]
pub(crate) struct ExposureFeedback {
    read: Option<[f32; 3]>,
}

impl Readback for ExposureFeedback {
    fn arrived(&mut self, _: u64, _: u32, bytes: &[u8]) {
        let word = |offset: usize| {
            bytes
                .get(offset..offset + 4)
                .and_then(|word| <[u8; 4]>::try_from(word).ok())
                .map_or(0.0, f32::from_le_bytes)
        };
        self.read = Some(
            [
                offset_of!(ExposureState, stops),
                offset_of!(ExposureState, metered),
                offset_of!(ExposureState, view),
            ]
            .map(word),
        );
    }
}

fn hundredths(stops: f32) -> i32 {
    (stops * 100.0).round() as i32
}

fn exposure_stats(
    feedback: Res<ExposureFeedback>,
    shown: Res<DebugShown>,
    mut stats: ResMut<ExposureStats>,
) {
    let active = shown.owner.is_none();
    let [stops, metered, view] = feedback.read.filter(|_| active).unwrap_or_default();
    stats.set_if_neq(ExposureStats {
        active,
        stops: hundredths(stops),
        metered: hundredths(metered),
        view: hundredths(view),
    });
}

fn forget_exposure(mut feedback: ResMut<ExposureFeedback>, mut stats: ResMut<ExposureStats>) {
    feedback.read = None;
    stats.set_if_neq(ExposureStats::default());
}
