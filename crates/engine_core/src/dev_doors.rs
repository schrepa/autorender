#[cfg(not(feature = "dev"))]
use std::marker::PhantomData;

use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::{ScheduleSystem, SystemParam};
#[cfg(feature = "dev")]
use bevy_ecs::change_detection::{DetectChanges, DetectChangesMut};
#[cfg(feature = "dev")]
use bevy_ecs::system::{Res, ResMut};
use engine_registry::StatsEntry;
use serde::{Deserialize, Serialize};

use crate::bricks::dev_settings_entry;
use crate::{Bricks, Schedule, Section};

#[doc(hidden)]
#[derive(Resource, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(transparent)]
pub struct DevOnly<S>(pub S);

impl<S> From<S> for DevOnly<S> {
    fn from(section: S) -> Self {
        Self(section)
    }
}

#[cfg(feature = "dev")]
#[derive(SystemParam)]
pub struct Stats<'w, S: Section> {
    shown: ResMut<'w, DevOnly<S>>,
}

#[cfg(not(feature = "dev"))]
#[derive(SystemParam)]
pub struct Stats<'w, S: Section> {
    off: PhantomData<fn() -> &'w S>,
}

#[cfg(feature = "dev")]
impl<S: Section> Stats<'_, S> {
    pub fn show(&mut self, made: impl FnOnce() -> S) {
        self.shown.set_if_neq(DevOnly(made()));
    }

    pub fn edit(&mut self, change: impl FnOnce(&mut S)) {
        change(&mut self.shown.0);
    }
}

#[cfg(not(feature = "dev"))]
impl<S: Section> Stats<'_, S> {
    pub fn show(&mut self, made: impl FnOnce() -> S) {
        drop(made);
    }

    pub fn edit(&mut self, change: impl FnOnce(&mut S)) {
        drop(change);
    }
}

#[cfg(feature = "dev")]
#[derive(SystemParam)]
pub struct DevSetting<'w, S: Section> {
    held: Option<Res<'w, DevOnly<S>>>,
}

#[cfg(not(feature = "dev"))]
#[derive(SystemParam)]
pub struct DevSetting<'w, S: Section> {
    off: PhantomData<fn() -> &'w S>,
}

#[cfg(feature = "dev")]
impl<S: Section> DevSetting<'_, S> {
    pub fn get(&self) -> Option<&S> {
        self.held.as_deref().map(|held| &held.0)
    }

    pub fn changed(&self) -> Option<&S> {
        self.held
            .as_ref()
            .filter(|held| held.is_changed())
            .map(|held| &held.0)
    }
}

#[cfg(not(feature = "dev"))]
impl<S: Section> DevSetting<'_, S> {
    pub fn get(&self) -> Option<&S> {
        None
    }

    pub fn changed(&self) -> Option<&S> {
        None
    }
}

pub struct DevBricks<'a, 'r>(&'a mut Bricks<'r>);

impl DevBricks<'_, '_> {
    #[track_caller]
    pub fn settings<S: Section>(&mut self) -> &mut Self {
        self.0.record.settings(dev_settings_entry::<S>());
        self
    }

    #[track_caller]
    pub fn stats<S: Section>(&mut self) -> &mut Self {
        self.0.record.stats(StatsEntry::of::<DevOnly<S>>(S::KEY));
        self
    }

    pub fn systems<M>(
        &mut self,
        schedule: Schedule,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        self.0.systems(schedule, systems);
        self
    }
}

impl Bricks<'_> {
    pub fn dev(&mut self, register: impl FnOnce(&mut DevBricks<'_, '_>)) -> &mut Self {
        #[cfg(feature = "dev")]
        register(&mut DevBricks(self));
        #[cfg(not(feature = "dev"))]
        drop(register);
        self
    }
}
