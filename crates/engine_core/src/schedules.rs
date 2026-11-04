use bevy_ecs::schedule::{InternedScheduleLabel, ScheduleLabel};

use crate::Schedule;

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
pub(crate) struct BetweenFrames;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Waiting;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Stream;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FrameStart;

impl Schedule {
    pub(crate) const ALL: [Self; 11] = [
        Self::FrameStart,
        Self::FixedPreSimulation,
        Self::FixedMove,
        Self::FixedSimulation,
        Self::FixedPostSimulation,
        Self::Presentation,
        Self::WorldSync,
        Self::ViewSync,
        Self::BetweenFrames,
        Self::Waiting,
        Self::Stream,
    ];

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
            Self::BetweenFrames => BetweenFrames.intern(),
            Self::Waiting => Waiting.intern(),
            Self::Stream => Stream.intern(),
        }
    }
}
