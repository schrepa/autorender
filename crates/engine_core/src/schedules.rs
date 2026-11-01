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

impl Schedule {
    pub(crate) const ALL: [Self; 8] = [
        Self::FixedPreSimulation,
        Self::FixedMove,
        Self::FixedSimulation,
        Self::FixedPostSimulation,
        Self::Presentation,
        Self::WorldSync,
        Self::ViewSync,
        Self::BetweenFrames,
    ];

    #[doc(hidden)]
    pub fn label(self) -> InternedScheduleLabel {
        match self {
            Self::FixedPreSimulation => FixedPreSimulation.intern(),
            Self::FixedMove => FixedMove.intern(),
            Self::FixedSimulation => FixedSimulation.intern(),
            Self::FixedPostSimulation => FixedPostSimulation.intern(),
            Self::Presentation => Presentation.intern(),
            Self::WorldSync => WorldSync.intern(),
            Self::ViewSync => ViewSync.intern(),
            Self::BetweenFrames => BetweenFrames.intern(),
        }
    }
}
