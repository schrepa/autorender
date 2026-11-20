use std::collections::BTreeSet;

use bevy_ecs::prelude::Resource;
use serde::Serialize;

use crate::{Error, ErrorCode, Expected};

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[component(immutable)]
pub struct FrameTime {
    pub frame: u64,
    pub ticks: u32,
    pub alpha: f32,
    pub seconds: f64,
    pub real_seconds: f64,
    pub elapsed: f64,
    pub real_elapsed: f64,
    pub cut_seconds: f64,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Pause {
    reasons: BTreeSet<&'static str>,
}

impl Pause {
    pub fn add(&mut self, reason: &'static str) -> bool {
        self.reasons.insert(reason)
    }

    pub fn remove(&mut self, reason: &'static str) -> bool {
        self.reasons.remove(reason)
    }

    pub fn holds(&self, reason: &'static str) -> bool {
        self.reasons.contains(reason)
    }

    pub fn paused(&self) -> bool {
        !self.reasons.is_empty()
    }

    pub fn reasons(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.reasons.iter().copied()
    }
}

#[derive(Resource, Clone, Copy, Debug)]
#[component(immutable)]
pub struct FixedTime {
    pub delta_seconds: f32,
    pub elapsed_seconds: f64,
    pub tick: u64,
    pub rate: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Interval {
    Fresh,
    Since(f64),
}

impl Interval {
    pub(crate) fn next(self, now: f64) -> (Self, f64) {
        match self {
            Self::Fresh => (Self::Since(now), 0.0),
            Self::Since(last) => {
                let delta = now - last;
                let delta = if delta.is_finite() && delta > 0.0 {
                    delta
                } else {
                    0.0
                };
                (Self::Since(now), delta)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Advanced {
    pub(crate) ticks: u32,
    pub(crate) alpha: f32,
}

pub(crate) struct FixedClock {
    step_seconds: f64,
    catch_up_seconds: f64,
    accumulator_seconds: f64,
}

impl FixedClock {
    pub(crate) fn new(ticks_per_second: u32, catch_up_seconds: f64) -> Result<Self, Error> {
        if ticks_per_second == 0 {
            return Err(Error::new(ErrorCode::OutOfRange, "$.ticks_per_second")
                .at_least(1)
                .got(ticks_per_second));
        }
        if !catch_up_seconds.is_finite() || catch_up_seconds <= 0.0 {
            return Err(Error::new(ErrorCode::OutOfRange, "$.catch_up_seconds")
                .expected(Expected::Above)
                .range(0.0, f64::MAX)
                .got(catch_up_seconds));
        }
        Ok(Self {
            step_seconds: 1.0 / f64::from(ticks_per_second),
            catch_up_seconds,
            accumulator_seconds: 0.0,
        })
    }

    pub(crate) fn step_seconds(&self) -> f64 {
        self.step_seconds
    }

    pub(crate) fn capped(&self, real_seconds: f64) -> f64 {
        real_seconds.min(self.catch_up_seconds)
    }

    pub(crate) fn reset(&mut self) {
        self.accumulator_seconds = 0.0;
    }

    pub(crate) fn advance(&mut self, delta_seconds: f64) -> Advanced {
        let incoming_steps = delta_seconds / self.step_seconds;
        let fractional_seconds = if incoming_steps.is_finite() {
            incoming_steps.fract() * self.step_seconds
        } else {
            delta_seconds % self.step_seconds
        };
        let accumulated_fraction = self.accumulator_seconds + fractional_seconds;
        let ticks =
            (incoming_steps.floor() + (accumulated_fraction / self.step_seconds).floor()) as u32;
        self.accumulator_seconds = accumulated_fraction % self.step_seconds;
        let alpha = (self.accumulator_seconds / self.step_seconds) as f32;
        Advanced {
            ticks,
            alpha: crate::math::min(alpha, f32::from_bits(1.0_f32.to_bits() - 1)),
        }
    }
}
