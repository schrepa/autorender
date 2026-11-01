use std::mem::offset_of;

use bevy_ecs::prelude::*;
use engine_core::{Error, FrameTime, View};
use engine_math::{clamp, max};
use engine_places::sky::{ADAPTATION_MOST, ADAPTATION_POWER, NOON_LIGHT, SkyLight};
use engine_places::sun::Sun;
use engine_places::tonemap::Display;
use engine_render::{
    AfterRead, Bind, Buffer, ComputePipeline, Count, Encode, GpuPart, Install, Lifetime, Pass,
    Prepare, ViewTarget,
};

use crate::gpu_layouts::{
    EXPOSURE_BINS, EXPOSURE_GRID, EXPOSURE_GROUP, ExposureState, ExposureUniform,
};
use crate::{AutoExposure, ExposureFeedback, shaders};

const STATE_BYTES: u64 = size_of::<ExposureState>() as u64;
const READ_BYTES: u64 = offset_of!(ExposureState, spare) as u64;
const UNIFORM_BYTES: u64 = size_of::<ExposureUniform>() as u64;
const BIN_BYTES: u64 = EXPOSURE_BINS as u64 * 4;
const STATE_READ: u32 = 0;

const DUSK: f32 = 0.069_756;
const NIGHT: f32 = 0.241_922;

pub(crate) struct MeterGpu {
    meter: Pass<Self>,
    metering: ComputePipeline<Self>,
    adapting: ComputePipeline<Self>,
    uniform: Buffer<Self>,
    bins: Buffer<Self>,
    state: Buffer<Self>,
    written: Option<ExposureUniform>,
    last_seconds: Option<f64>,
    last_camera: [u32; 16],
}

impl ExposureUniform {
    fn of(
        settings: &AutoExposure,
        sun: &Sun,
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
        let below = -sun.direction[1];
        let night = clamp((below - DUSK) / (NIGHT - DUSK), 0.0, 1.0);
        let night = night * night * (3.0 - 2.0 * night);
        let most = ADAPTATION_MOST.log2();
        let most = most + (settings.night - most) * night;
        Self {
            frame: [authored, elapsed, settings.strength, settings.compensation],
            reach: [
                settings.brighten * (1.0 - night),
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
            day: [noon, sky.adaptation * ADAPTATION_POWER, most, 0.0],
            drawn: [drawn[0] as f32, drawn[1] as f32, 0.0, 0.0],
        }
    }
}

impl GpuPart for MeterGpu {
    type Reads = (
        Res<'static, AutoExposure>,
        Res<'static, Sun>,
        Res<'static, SkyLight>,
        Res<'static, Display>,
        Res<'static, FrameTime>,
        Res<'static, View>,
    );

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        use shaders::auto_exposure::{faces::exposure as face, meter};
        let state = cx.buffer("exposure_state", Lifetime::Previous);
        cx.frame(face::EXPOSURE_STATE, Bind::Buffer(state))?;
        let uniform = cx.buffer("exposure_uniform", Lifetime::Kept);
        let bins = cx.buffer("exposure_bins", Lifetime::Kept);
        let pass = cx
            .pass("exposure")
            .label("engine/exposure")
            .span("auto_exposure")
            .id();
        let metering = cx.compute_pipeline(
            pass,
            meter::METER_EXPOSURE,
            &[
                (meter::SETTINGS, Bind::Buffer(uniform)),
                (meter::SCENE_COLOR, Bind::Target(ViewTarget::Scene)),
                (meter::SCENE_DEPTH, Bind::Target(ViewTarget::Depth)),
                (meter::BINS, Bind::Buffer(bins)),
                (meter::STATE, Bind::Buffer(state)),
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
            written: None,
            last_seconds: None,
            last_camera: [0; 16],
        })
    }

    fn prepare(
        &mut self,
        reads: SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        let (settings, sun, sky, display, time, view) = reads;
        cx.allocate_buffer(self.uniform, UNIFORM_BYTES);
        cx.allocate_buffer(self.bins, BIN_BYTES);
        cx.allocate_buffer(self.state, STATE_BYTES);
        let authored = display.exposure[0];
        if cx.fresh() {
            let state = ExposureState {
                exposure: authored,
                ..ExposureState::default()
            };
            cx.write(self.state, 0, &[state]);
            self.written = None;
            self.last_seconds = None;
            self.last_camera = [0; 16];
        }
        if view.debug() {
            return;
        }
        let seconds = time.presented_seconds;
        let camera = view.view_projection().to_cols_array().map(f32::to_bits);
        let moved = std::mem::replace(&mut self.last_camera, camera) != camera;
        let elapsed = self
            .last_seconds
            .replace(seconds)
            .map_or(0.0, |last| max((seconds - last) as f32, 0.0));
        let elapsed = if elapsed <= 0.0 && moved {
            -1.0
        } else {
            elapsed
        };
        let drawn = view.render_size().to_array();
        let values = ExposureUniform::of(&settings, &sun, &sky, authored, elapsed, drawn);
        if self.written != Some(values) {
            cx.write(self.uniform, 0, &[values]);
            self.written = Some(values);
        }
        cx.run(self.meter);
        cx.read_back::<ExposureFeedback>(STATE_READ, self.state, 0..READ_BYTES, AfterRead::Keep);
    }

    fn encode(&mut self, _: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        cx.clear(self.bins);
        let groups = EXPOSURE_GRID / EXPOSURE_GROUP;
        cx.dispatch(&self.metering, Count::Groups([groups, groups, 1]));
        cx.dispatch(&self.adapting, Count::Groups([1, 1, 1]));
        0
    }
}
