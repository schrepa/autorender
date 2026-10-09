use bevy_ecs::prelude::Resource;

use crate::AssetBudget;

/// Bounds gameplay work and render extraction independently.
#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    /// Simulation frequency; finite values clamp to 1–1000 Hz. Invalid values
    /// use 60 Hz. This makes the fixed delta representable as `f32`.
    pub fixed_hz: f64,
    /// Maximum fixed ticks per frame, clamped to 1–1000.
    pub max_fixed_steps: u32,
    /// Maximum visible instances copied per snapshot; beyond it the rest are
    /// dropped and counted. `usize::MAX`, the default, sets no budget (the
    /// GPU's buffers bound it). Zero emits no instances; counters and
    /// retention of culled sources still run.
    pub max_render_instances: usize,
    /// Admission limits of the engine's asset library; the default admits
    /// everything.
    pub asset_budget: AssetBudget,
    /// Transformed entities propagation admits ([`crate::TransformNodeLimit`]);
    /// the default sets no budget.
    pub max_transform_nodes: usize,
    /// Conservatively omit boxes outside the camera frustum during extraction.
    /// Disable for repeatable comparisons against unculled extraction.
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

/// Simulation time, advanced immediately before each fixed schedule run.
#[derive(Resource, Debug, Clone, Copy)]
pub struct FixedTime {
    pub delta_seconds: f32,
    pub elapsed_seconds: f64,
    pub tick: u64,
}

/// Presentation time, updated once before a frame's fixed and presentation work.
/// Elapsed time sums the bounded deltas, excluding dropped wall-clock time.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct FrameTime {
    /// Valid nonnegative input delta, capped at 0.25 seconds.
    pub delta_seconds: f32,
    pub elapsed_seconds: f64,
    pub frame: u64,
}

/// The simulation time the last extracted frame presents:
/// the fixed clock where its poses stand, between ticks for interpolated
/// frames. What moves with time and must line up with simulated objects,
/// such as the sea's waves, follows it.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct PresentedTime {
    pub seconds: f64,
}

/// Work performed by one call to `Engine::advance`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameReport {
    pub fixed_steps: u32,
    /// Whole fixed-step time discarded to enforce the catch-up bound.
    pub dropped_seconds: f64,
    /// Fraction remaining until the next fixed tick, in the range [0, 1).
    /// Pass to `PresentationMode::Interpolated` for opted-in local transforms.
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

    /// Unconsumed time since the last due tick, in `[0, step)`.
    pub(crate) fn accumulator_seconds(&self) -> f64 {
        self.accumulator_seconds
    }

    pub(crate) fn reset(&mut self) {
        self.accumulator_seconds = 0.0;
    }

    pub(crate) fn advance(&mut self, delta_seconds: f64) -> FrameReport {
        let previous_remainder = self.accumulator_seconds;
        // Split off whole ticks first so even a huge finite host delta cannot
        // overflow the accumulator or erase its existing fractional remainder.
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
        // A fraction just below one can round up when converted to f32.
        let interpolation = (self.accumulator_seconds / self.step_seconds) as f32;
        FrameReport {
            fixed_steps,
            dropped_seconds,
            interpolation: crate::math::min(interpolation, f32::from_bits(1.0_f32.to_bits() - 1)),
        }
    }
}
