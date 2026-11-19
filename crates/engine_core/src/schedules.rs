use bevy_ecs::schedule::{InternedScheduleLabel, ScheduleLabel};

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BeforeTicks;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct FixedSimulation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Presentation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct TransformPropagation;

#[derive(ScheduleLabel, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BeforeDraw;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Schedule {
    BeforeTicks,
    FixedSimulation,
    Presentation,
    BeforeDraw,
}

impl Schedule {
    pub const ALL: [Self; 4] = [
        Self::BeforeTicks,
        Self::FixedSimulation,
        Self::Presentation,
        Self::BeforeDraw,
    ];

    #[doc(hidden)]
    pub fn label(self) -> InternedScheduleLabel {
        match self {
            Self::BeforeTicks => BeforeTicks.intern(),
            Self::FixedSimulation => FixedSimulation.intern(),
            Self::Presentation => Presentation.intern(),
            Self::BeforeDraw => BeforeDraw.intern(),
        }
    }
}
