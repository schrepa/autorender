use bevy_ecs::prelude::{Res, ResMut, Resource};
use bevy_ecs::system::SystemParamItem;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{AssetBudget, Error, Operation, settings};

pub const MAX_TIME_SCALE: f32 = 8.0;

#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub fixed_hz: f64,
    pub asset_budget: AssetBudget,
    pub max_transform_nodes: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            fixed_hz: 60.0,
            asset_budget: AssetBudget::default(),
            max_transform_nodes: crate::MAX_TRANSFORM_NODES,
        }
    }
}

settings! {
    #[section("frame_limits")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneFrameLimits")]
    pub struct FrameLimits {
        pub max_fixed_steps: u32 = 5 => range(1, 1000),
        pub max_frame_seconds: f64 = 0.25 => above(0.0),
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
#[component(immutable)]
pub struct FrameTime {
    pub delta_seconds: f32,
    pub elapsed_seconds: f64,
    pub game_delta_seconds: f32,
    pub game_elapsed_seconds: f64,
    pub presented_seconds: f64,
    pub frame: u64,
    pub between_frames_micros: u32,
}

impl FrameTime {
    pub fn real_presented_seconds(&self) -> f64 {
        self.presented_seconds + (self.elapsed_seconds - self.game_elapsed_seconds)
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct TimeControl {
    pub paused: bool,
    pub scale: f32,
}

impl Default for TimeControl {
    fn default() -> Self {
        Self {
            paused: false,
            scale: 1.0,
        }
    }
}

impl TimeControl {
    pub fn kept_scale(self) -> f32 {
        if self.scale.is_nan() {
            1.0
        } else {
            crate::math::clamp(self.scale, 0.0, MAX_TIME_SCALE)
        }
    }

    pub fn rate(self) -> f64 {
        if self.paused {
            0.0
        } else {
            f64::from(self.kept_scale())
        }
    }

    fn view(self) -> Value {
        json!({"time_paused": self.paused, "time_scale": self.kept_scale()})
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimeRequest {
    #[serde(default)]
    paused: Option<bool>,
    #[serde(default)]
    scale: Option<f32>,
}

pub(crate) struct TimeOperation;

impl Operation for TimeOperation {
    const NAME: &'static str = "time";
    type Params = ResMut<'static, TimeControl>;
    type Request = TimeRequest;

    fn run(
        mut control: SystemParamItem<'_, '_, Self::Params>,
        request: TimeRequest,
    ) -> Result<Value, Error> {
        if let Some(paused) = request.paused {
            control.paused = paused;
        }
        if let Some(scale) = request.scale {
            control.scale = scale;
            control.scale = control.kept_scale();
        }
        Ok(control.view())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimeStateRequest {}

pub(crate) struct TimeState;

impl Operation for TimeState {
    const NAME: &'static str = "time_state";
    const CONTROL: bool = false;
    type Params = Res<'static, TimeControl>;
    type Request = TimeStateRequest;

    fn run(
        control: SystemParamItem<'_, '_, Self::Params>,
        _: TimeStateRequest,
    ) -> Result<Value, Error> {
        Ok(control.view())
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
#[component(immutable)]
pub struct FixedTime {
    pub delta_seconds: f32,
    pub elapsed_seconds: f64,
    pub tick: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameReport {
    pub fixed_steps: u32,
    pub dropped_seconds: f64,
    pub cut_seconds: f64,
    pub interpolation: f32,
}

pub(crate) fn sanitize_delta(delta_seconds: f64) -> f64 {
    if delta_seconds.is_finite() && delta_seconds > 0.0 {
        delta_seconds
    } else {
        0.0
    }
}

pub(crate) struct FixedClock {
    step_seconds: f64,
    accumulator_seconds: f64,
}

impl FixedClock {
    pub(crate) fn new(config: EngineConfig) -> Self {
        let hz = if config.fixed_hz.is_finite() && config.fixed_hz > 0.0 {
            config.fixed_hz.clamp(1.0, 1000.0)
        } else {
            EngineConfig::default().fixed_hz
        };
        Self {
            step_seconds: 1.0 / hz,
            accumulator_seconds: 0.0,
        }
    }

    pub(crate) fn step_seconds(&self) -> f64 {
        self.step_seconds
    }

    pub(crate) fn reset(&mut self) {
        self.accumulator_seconds = 0.0;
    }

    pub(crate) fn advance(&mut self, delta_seconds: f64, max_fixed_steps: u32) -> FrameReport {
        let previous_remainder = self.accumulator_seconds;
        let incoming_steps = delta_seconds / self.step_seconds;
        let fractional_seconds = if incoming_steps.is_finite() {
            incoming_steps.fract() * self.step_seconds
        } else {
            delta_seconds % self.step_seconds
        };
        let accumulated_fraction = previous_remainder + fractional_seconds;
        let available_steps =
            incoming_steps.floor() + (accumulated_fraction / self.step_seconds).floor();
        let fixed_steps = available_steps.min(f64::from(max_fixed_steps)) as u32;
        self.accumulator_seconds = accumulated_fraction % self.step_seconds;
        let dropped_seconds = if available_steps > f64::from(fixed_steps) {
            (delta_seconds - f64::from(fixed_steps) * self.step_seconds + previous_remainder
                - self.accumulator_seconds)
                .max(0.0)
        } else {
            0.0
        };
        let interpolation = (self.accumulator_seconds / self.step_seconds) as f32;
        FrameReport {
            fixed_steps,
            dropped_seconds,
            cut_seconds: 0.0,
            interpolation: crate::math::min(interpolation, f32::from_bits(1.0_f32.to_bits() - 1)),
        }
    }
}
