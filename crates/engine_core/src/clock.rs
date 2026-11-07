use std::collections::BTreeSet;

use bevy_ecs::prelude::{ResMut, Resource};
use bevy_ecs::system::SystemParamItem;
use bevy_ecs::world::World;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{AssetBudget, Clock, Error, ErrorCode, List, Operation, settings};

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

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[component(immutable)]
pub struct FrameTime {
    pub frame: u64,
    pub ticks: u32,
    pub alpha: f32,
    pub seconds: f64,
    pub real_seconds: f64,
    pub elapsed: f64,
    pub real_elapsed: f64,
    pub dropped_seconds: f64,
    pub cut_seconds: f64,
    pub between_frames_micros: u32,
}

impl FrameTime {
    pub fn seconds_on(&self, clock: Clock) -> f64 {
        match clock {
            Clock::Game => self.seconds,
            Clock::Real => self.real_seconds,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum PauseReason {
    Agent,
    Game,
    Focus,
    Hidden,
    Suspended,
    #[serde(untagged)]
    Brick(&'static str),
}

impl PauseReason {
    pub fn stops_frames(self) -> bool {
        match self {
            Self::Hidden | Self::Suspended => true,
            Self::Agent | Self::Game | Self::Focus | Self::Brick(_) => false,
        }
    }
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Pause {
    reasons: BTreeSet<PauseReason>,
}

impl List for Pause {
    const NAME: &'static str = "pause";
    type Member = PauseReason;

    fn dropped(world: &mut World, member: &PauseReason) {
        if world
            .get_resource::<Self>()
            .is_some_and(|pause| pause.holds(*member))
        {
            world.resource_mut::<Self>().remove(*member);
        }
    }
}

impl Pause {
    pub fn add(&mut self, reason: PauseReason) -> bool {
        self.reasons.insert(reason)
    }

    pub fn remove(&mut self, reason: PauseReason) -> bool {
        self.reasons.remove(&reason)
    }

    pub fn holds(&self, reason: PauseReason) -> bool {
        self.reasons.contains(&reason)
    }

    pub fn paused(&self) -> bool {
        !self.reasons.is_empty()
    }

    pub fn reasons(&self) -> impl Iterator<Item = PauseReason> + '_ {
        self.reasons.iter().copied()
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TimeScale(pub f32);

impl Default for TimeScale {
    fn default() -> Self {
        Self(1.0)
    }
}

impl TimeScale {
    pub fn check(self) -> Result<Self, Error> {
        if self.0.is_finite() && self.0 >= 0.0 {
            Ok(self)
        } else {
            Err(Error::new(ErrorCode::InvalidValue, "$.scale")
                .range(0.0, f64::from(f32::MAX))
                .got(self.0))
        }
    }
}

pub(crate) fn game_rate(pause: &Pause, scale: TimeScale) -> f64 {
    if pause.paused() || !(scale.0.is_finite() && scale.0 >= 0.0) {
        0.0
    } else {
        f64::from(scale.0)
    }
}

fn time_view(pause: &Pause, scale: TimeScale) -> Value {
    let reasons: Vec<PauseReason> = pause.reasons().collect();
    json!({"time_paused": pause.paused(), "pause_reasons": reasons, "time_scale": scale.0})
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
    type Params = (ResMut<'static, Pause>, ResMut<'static, TimeScale>);
    type Request = TimeRequest;

    fn run(
        (mut pause, mut scale): SystemParamItem<'_, '_, Self::Params>,
        request: TimeRequest,
    ) -> Result<Value, Error> {
        let wanted = request.scale.map(TimeScale).map(TimeScale::check).transpose()?;
        if let Some(wanted) = wanted {
            *scale = wanted;
        }
        match request.paused {
            Some(true) if !pause.holds(PauseReason::Agent) => {
                pause.add(PauseReason::Agent);
            }
            Some(false) if pause.holds(PauseReason::Agent) => {
                pause.remove(PauseReason::Agent);
            }
            Some(_) | None => {}
        }
        Ok(time_view(&pause, *scale))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TimeStateRequest {}

pub(crate) struct TimeState;

impl Operation for TimeState {
    const NAME: &'static str = "time_state";
    const CONTROL: bool = false;
    type Params = (
        bevy_ecs::prelude::Res<'static, Pause>,
        bevy_ecs::prelude::Res<'static, TimeScale>,
    );
    type Request = TimeStateRequest;

    fn run(
        (pause, scale): SystemParamItem<'_, '_, Self::Params>,
        _: TimeStateRequest,
    ) -> Result<Value, Error> {
        Ok(time_view(&pause, *scale))
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
#[component(immutable)]
pub struct FixedTime {
    pub delta_seconds: f32,
    pub elapsed_seconds: f64,
    pub tick: u64,
}

pub(crate) fn sanitize_delta(delta_seconds: f64) -> f64 {
    if delta_seconds.is_finite() && delta_seconds > 0.0 {
        delta_seconds
    } else {
        0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Interval {
    Fresh,
    Since(f64),
}

impl Interval {
    pub(crate) fn next(self, timestamp: Option<f64>) -> (Self, f64) {
        match (self, timestamp) {
            (Self::Fresh, Some(now)) => (Self::Since(now), 0.0),
            (Self::Since(last), Some(now)) => (Self::Since(now), sanitize_delta(now - last)),
            (Self::Fresh | Self::Since(_), None) => (Self::Fresh, 0.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Advanced {
    pub(crate) ticks: u32,
    pub(crate) dropped_seconds: f64,
    pub(crate) alpha: f32,
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

    pub(crate) fn advance(&mut self, delta_seconds: f64, max_fixed_steps: u32) -> Advanced {
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
        let ticks = available_steps.min(f64::from(max_fixed_steps)) as u32;
        self.accumulator_seconds = accumulated_fraction % self.step_seconds;
        let dropped_seconds = if available_steps > f64::from(ticks) {
            (delta_seconds - f64::from(ticks) * self.step_seconds + previous_remainder
                - self.accumulator_seconds)
                .max(0.0)
        } else {
            0.0
        };
        let alpha = (self.accumulator_seconds / self.step_seconds) as f32;
        Advanced {
            ticks,
            dropped_seconds,
            alpha: crate::math::min(alpha, f32::from_bits(1.0_f32.to_bits() - 1)),
        }
    }
}
