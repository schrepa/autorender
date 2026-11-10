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
pub(crate) struct FixedPreSimulation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FixedMove;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FixedSimulation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FixedPostSimulation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Presentation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TransformPropagation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct WorldSync;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ViewSync;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FrameStart;

impl Schedule {
    pub const ALL: [Self; 8] = [
        Self::FrameStart,
        Self::FixedPreSimulation,
        Self::FixedMove,
        Self::FixedSimulation,
        Self::FixedPostSimulation,
        Self::Presentation,
        Self::WorldSync,
        Self::ViewSync,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::FrameStart => "frame_start",
            Self::FixedPreSimulation => "fixed_pre_simulation",
            Self::FixedMove => "fixed_move",
            Self::FixedSimulation => "fixed_simulation",
            Self::FixedPostSimulation => "fixed_post_simulation",
            Self::Presentation => "presentation",
            Self::WorldSync => "world_sync",
            Self::ViewSync => "view_sync",
        }
    }

    pub const fn pace(self) -> RunPace {
        match self {
            Self::FixedPreSimulation
            | Self::FixedMove
            | Self::FixedSimulation
            | Self::FixedPostSimulation => RunPace::FixedStep,
            Self::FrameStart | Self::Presentation | Self::WorldSync | Self::ViewSync => {
                RunPace::Frame
            }
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
            Self::FixedPreSimulation => FixedPreSimulation.intern(),
            Self::FixedMove => FixedMove.intern(),
            Self::FixedSimulation => FixedSimulation.intern(),
            Self::FixedPostSimulation => FixedPostSimulation.intern(),
            Self::Presentation => Presentation.intern(),
            Self::WorldSync => WorldSync.intern(),
            Self::ViewSync => ViewSync.intern(),
        }
    }
}
