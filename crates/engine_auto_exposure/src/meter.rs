use bevy_ecs::prelude::*;
use engine_core::{Error, FrameTime, PassName, PrimaryView};
use engine_places_rendering::sky::SkyLight;
use engine_places_rendering::tonemap::Display;
use engine_render::{
    AfterRead, Allocation, Bind, Buffer, ComputePipeline, Count, Encode, GpuPart, Install, Lifetime,
    Pass, Prepare, TargetName,
};

use crate::gpu_layouts::{
    EXPOSURE_BINS, EXPOSURE_GRID, EXPOSURE_GROUP, ExposureState, ExposureUniform,
};
use crate::{AutoExposure, ExposureFeedback, ExposurePass, shaders};

const STATE_BYTES: u64 = size_of::<ExposureState>() as u64;
const UNIFORM_BYTES: u64 = size_of::<ExposureUniform>() as u64;
const BIN_BYTES: u64 = EXPOSURE_BINS as u64 * 4;
const STATE_READ: u32 = 0;
const NOON_LIGHT: f32 = 1.1;
const ADAPTATION_POWER: f32 = 0.75;
const ADAPTATION_MOST: f32 = 16.0;

pub(crate) struct MeterGpu {
    meter: Pass<Self>,
    metering: ComputePipeline<Self>,
    adapting: ComputePipeline<Self>,
    uniform: Buffer<Self>,
    bins: Buffer<Self>,
    state: Buffer<Self>,
}

impl ExposureUniform {
    fn of(
        settings: &AutoExposure,
        sky: &SkyLight,
        authored: f32,
        elapsed: f32,
        drawn: [u32; 2],
    ) -> Self {
        let noon = if sky.on {
            sky.sun_intensity * NOON_LIGHT
        } else {
            0.0
        };
        Self {
            frame: [authored, elapsed, settings.strength, settings.compensation],
            reach: [
                settings.brighten,
                settings.darken,
                settings.to_darker,
                settings.to_brighter,
            ],
            meter: [
                settings.skip_dark,
                settings.skip_bright,
                settings.center,
                settings.sky,
            ],
            day: [
                noon,
                sky.adaptation * ADAPTATION_POWER,
                ADAPTATION_MOST.log2(),
                0.0,
            ],
            drawn: [drawn[0] as f32, drawn[1] as f32, 0.0, 0.0],
        }
    }
}

impl GpuPart for MeterGpu {
    type Reads = (
        Res<'static, AutoExposure>,
        Res<'static, SkyLight>,
        Res<'static, Display>,
        Res<'static, FrameTime>,
        Res<'static, PrimaryView>,
    );
    const NAME: &'static str = "exposure";
    const LAZY: bool = true;
    const LABELS: &'static [&'static str] = &["engine/exposure"];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::auto_exposure::{faces::exposure as face, meter};
        let state = cx.buffer("exposure_state", Lifetime::Previous);
        cx.frame(face::EXPOSURE_STATE, Bind::Buffer(state))?;
        let uniform = cx.buffer("exposure_uniform", Lifetime::Kept);
        let bins = cx.buffer("exposure_bins", Lifetime::Kept);
        let pass = cx
            .pass(ExposurePass::NAME)
            .at(ExposurePass::PASS.at)
            .label("engine/exposure")
            .span("auto_exposure")
            .id();
        let metering = cx.compute_pipeline(
            pass,
            meter::METER_EXPOSURE,
            &[
                (meter::SETTINGS, Bind::Buffer(uniform)),
                (meter::SCENE_COLOR, Bind::Target(TargetName::COLOR)),
                (meter::SCENE_DEPTH, Bind::Target(TargetName::DEPTH)),
                (meter::BINS, Bind::Buffer(bins)),
            ],
        )?;
        let adapting = cx.compute_pipeline(
            pass,
            meter::ADAPT_EXPOSURE,
            &[
                (meter::SETTINGS, Bind::Buffer(uniform)),
                (meter::BINS, Bind::Buffer(bins)),
                (meter::STATE, Bind::Buffer(state)),
            ],
        )?;
        Ok(Self {
            meter: pass,
            metering,
            adapting,
            uniform,
            bins,
            state,
        })
    }

    fn prepare(&mut self, reads: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        let (settings, sky, display, time, view) = reads;
        cx.allocate_buffer(self.uniform, UNIFORM_BYTES);
        cx.allocate_buffer(self.bins, BIN_BYTES);
        let authored = display.exposure[0];
        if cx.allocate_buffer(self.state, STATE_BYTES) == Allocation::Fresh {
            let state = ExposureState {
                exposure: authored,
                ..ExposureState::default()
            };
            cx.write(self.state, 0, &[state]);
        }
        if view.debug() {
            return;
        }
        let elapsed = if view.capture() || !view.history().valid() {
            0.0
        } else if time.real_seconds > 0.0 {
            time.real_seconds as f32
        } else {
            -1.0
        };
        let drawn = view.render_size().to_array();
        let values = ExposureUniform::of(&settings, &sky, authored, elapsed, drawn);
        cx.write(self.uniform, 0, &[values]);
        cx.run(self.meter);
        cx.read_back::<ExposureFeedback>(STATE_READ, self.state, 0..STATE_BYTES, AfterRead::Keep);
    }

    fn encode(&mut self, _: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        cx.clear(self.bins);
        let groups = EXPOSURE_GRID / EXPOSURE_GROUP;
        cx.dispatch(&self.metering, Count::Groups([groups, groups, 1]));
        cx.dispatch(&self.adapting, Count::Groups([1, 1, 1]));
        0
    }
}
