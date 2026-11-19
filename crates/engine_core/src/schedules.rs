use bevy_ecs::schedule::{InternedScheduleLabel, ScheduleLabel};

use crate::Schedule;

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

    #[doc(hidden)]
    pub fn label(self) -> InternedScheduleLabel {
        match self {
            Self::FrameStart => FrameStart.intern(),
            Self::FixedSimulation => FixedSimulation.intern(),
            Self::Presentation => Presentation.intern(),
        }
    }
}
