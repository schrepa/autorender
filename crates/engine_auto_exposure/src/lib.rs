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
use engine_core::{
    AttachPoint, Brick, Bricks, GpuStruct, Pass, PassName, PassPurpose, PassViews, Readable,
    Readback, Schedule, Settling, SlotSet, Source, Writes, never,
};
use engine_places_rendering::exposure::Exposure;
use engine_render::RenderBricks;

use crate::gpu_layouts::{ExposureState, ExposureUniform};

pub use settings::{AutoExposure, ExposureStats};

const GENERATED: &[&str] = &[gpu_layouts::WGSL, ExposureUniform::WGSL, ExposureState::WGSL];

pub(crate) struct ExposurePass;

impl PassName for ExposurePass {
    const NAME: &'static str = "exposure";
    const PASS: Pass = Pass {
        writes: Writes::NOTHING,
        reads: SlotSet::NONE,
        views: PassViews::Main,
        at: AttachPoint::Display,
        purpose: PassPurpose::Scene,
        takes: never,
    };
}

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
            .pass::<ExposurePass>()
            .systems(Schedule::ViewSync, exposure_stats)
            .generated(GENERATED)
            .part::<meter::MeterGpu>();
    }
}

#[derive(Resource, Default)]
pub(crate) struct ExposureFeedback {
    read: Option<[f32; 5]>,
    settling: Settling,
}

impl Readable for ExposureFeedback {
    fn arrived(&mut self, _: u32, read: Readback<&[u8]>) {
        let bytes = read.value;
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
                offset_of!(ExposureState, wanted),
                offset_of!(ExposureState, moving),
            ]
            .map(word),
        );
    }
}

fn hundredths(stops: f32) -> i32 {
    (stops * 100.0).round() as i32
}

fn exposure_stats(feedback: Res<ExposureFeedback>, mut stats: ResMut<ExposureStats>) {
    let unread = feedback.read.is_none();
    let [stops, metered, viewed, wanted, moving] = feedback.read.unwrap_or_default();
    let (stops, wanted) = (hundredths(stops), hundredths(wanted));
    feedback
        .settling
        .left(u32::from(unread || (moving != 0.0 && stops != wanted)));
    stats.set_if_neq(ExposureStats {
        active: true,
        stops,
        metered: hundredths(metered),
        view: hundredths(viewed),
        wanted,
    });
}
