use bevy_ecs::prelude::Resource;

use crate::AssetBudget;

#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    pub fixed_hz: f64,
    pub max_fixed_steps: u32,
    pub max_render_instances: usize,
    pub asset_budget: AssetBudget,
    pub max_transform_nodes: usize,
    pub frustum_culling: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            fixed_hz: 60.0,
            max_fixed_steps: 5,
            max_render_instances: usize::MAX,
            asset_budget: AssetBudget::default(),
            max_transform_nodes: crate::MAX_TRANSFORM_NODES,
            frustum_culling: true,
        }
    }
}

#[derive(Resource, Debug, Clone, Copy)]
pub struct FixedTime {
    pub delta_seconds: f32,
    pub elapsed_seconds: f64,
    pub tick: u64,
}

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct FrameTime {
    pub delta_seconds: f32,
    pub elapsed_seconds: f64,
    pub frame: u64,
}

#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct PresentedTime {
    pub seconds: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FrameReport {
    pub fixed_steps: u32,
    pub dropped_seconds: f64,
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
    max_fixed_steps: u32,
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
            max_fixed_steps: config.max_fixed_steps.clamp(1, 1000),
        }
    }

    pub(crate) fn step_seconds(&self) -> f64 {
        self.step_seconds
    }

    pub(crate) fn accumulator_seconds(&self) -> f64 {
        self.accumulator_seconds
    }

    pub(crate) fn reset(&mut self) {
        self.accumulator_seconds = 0.0;
    }

    pub(crate) fn advance(&mut self, delta_seconds: f64) -> FrameReport {
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
        let fixed_steps = available_steps.min(f64::from(self.max_fixed_steps)) as u32;
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
            interpolation: crate::math::min(interpolation, f32::from_bits(1.0_f32.to_bits() - 1)),
        }
    }
}
