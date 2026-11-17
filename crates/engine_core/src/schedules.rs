use bevy_ecs::schedule::{InternedScheduleLabel, ScheduleLabel};

use serde::Serialize;

use crate::Schedule;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunPace {
    FixedStep,
    Frame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
pub struct RunPoint {
    pub name: &'static str,
    pub pace: RunPace,
}

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FixedSimulation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Presentation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TransformPropagation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FrameStart;

impl Schedule {
    pub const ALL: [Self; 3] = [
        Self::FrameStart,
        Self::FixedSimulation,
        Self::Presentation,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::FrameStart => "frame_start",
            Self::FixedSimulation => "fixed_simulation",
            Self::Presentation => "presentation",
        }
    }

    pub const fn pace(self) -> RunPace {
        match self {
            Self::FixedSimulation => RunPace::FixedStep,
            Self::FrameStart | Self::Presentation => RunPace::Frame,
        }
    }

    pub fn point(self) -> RunPoint {
        RunPoint {
            name: self.name(),
            pace: self.pace(),
        }
    }

    #[doc(hidden)]
    pub fn label(self) -> InternedScheduleLabel {
        match self {
            Self::FrameStart => FrameStart.intern(),
            Self::FixedSimulation => FixedSimulation.intern(),
            Self::Presentation => Presentation.intern(),
        }
    }
}
