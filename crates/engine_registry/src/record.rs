use std::any::{Any, TypeId};
use std::sync::Arc;

use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{InternedScheduleLabel, IntoScheduleConfigs, ScheduleConfigs};
use bevy_ecs::system::ScheduleSystem;
use engine_place::{Error, ErrorCode, List};

use crate::entries::{JoinEntry, Owned, SettingsEntry, StatsEntry};
use crate::state::SwitchFn;

pub trait RecordExt {
    fn record(&mut self) -> &mut BrickRecord;
}

#[derive(Default)]
pub struct BrickSystems {
    pub(crate) calls: Vec<(InternedScheduleLabel, ScheduleConfigs<ScheduleSystem>)>,
}

impl BrickSystems {
    pub fn add<M>(
        &mut self,
        schedule: InternedScheduleLabel,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) {
        self.calls.push((schedule, systems.chain()));
    }
}

pub struct BrickRecord {
    pub package: &'static str,
    game: bool,
    pub entries: Vec<Box<dyn Any + Send + Sync>>,
    pub render: Vec<Arc<dyn Any + Send + Sync>>,
    prefix: Option<&'static str>,
    switch: Option<(&'static str, SwitchFn)>,
    pub(crate) joins: Vec<JoinEntry>,
    owned: Vec<Owned>,
    settings: Vec<SettingsEntry>,
    stats: Vec<StatsEntry>,
    problems: Vec<Error>,
}

impl BrickRecord {
    pub fn new(package: &'static str) -> Self {
        Self {
            package,
            game: false,
            entries: Vec::new(),
            render: Vec::new(),
            prefix: None,
            switch: None,
            joins: Vec::new(),
            owned: Vec::new(),
            settings: Vec::new(),
            stats: Vec::new(),
            problems: Vec::new(),
        }
    }

    pub fn set_game(&mut self, game: bool) {
        self.game = game;
    }

    pub fn game(&self) -> bool {
        self.game
    }

    pub fn prefix(&self) -> Option<&'static str> {
        self.prefix
    }

    pub fn switch_key(&self) -> Option<&'static str> {
        self.switch.map(|(key, _)| key)
    }

    pub(crate) fn switch_fn(&self) -> Option<SwitchFn> {
        self.switch.map(|(_, on)| on)
    }

    pub fn owned(&self) -> &[Owned] {
        &self.owned
    }

    pub fn settings_entries(&self) -> &[SettingsEntry] {
        &self.settings
    }

    pub fn stats_entries(&self) -> &[StatsEntry] {
        &self.stats
    }

    pub fn entries<T: Any>(&self) -> impl Iterator<Item = &T> + '_ {
        self.entries
            .iter()
            .filter_map(|entry| entry.downcast_ref::<T>())
    }

    pub fn render_entries<T: Any>(&self) -> impl Iterator<Item = &T> + '_ {
        self.render
            .iter()
            .filter_map(|entry| entry.downcast_ref::<T>())
    }

    pub fn owns(&self, type_id: TypeId) -> bool {
        self.owned.iter().any(|owned| owned.type_id == type_id)
    }

    pub fn switch(&mut self, key: &'static str, on: SwitchFn) {
        if self.switch.is_some() {
            let path = format!("$.bricks.{}.switch", self.package);
            self.problems
                .push(Error::new(ErrorCode::NameTwice, path).got(key));
            return;
        }
        self.switch = Some((key, on));
    }

    pub fn shader_prefix(&mut self, prefix: &'static str) {
        self.prefix = Some(prefix);
    }

    pub fn join<L: List>(&mut self, member: L::Member) {
        self.joins.push(JoinEntry::of::<L>(member));
    }

    pub fn resource<R: Resource + Default>(&mut self) {
        self.own(Owned::of::<R>());
    }

    pub fn settings(&mut self, mut entry: SettingsEntry) {
        entry.brick = self.package;
        if self.section_taken(entry.key) {
            return;
        }
        self.own(entry.owned());
        self.settings.push(entry);
    }

    pub fn stats(&mut self, mut entry: StatsEntry) {
        entry.brick = self.package;
        if self.section_taken(entry.key) {
            return;
        }
        self.own(entry.owned());
        self.stats.push(entry);
    }

    pub(crate) fn take_problems(&mut self) -> Vec<Error> {
        std::mem::take(&mut self.problems)
    }

    pub(crate) fn dropped(&self) -> Vec<Owned> {
        self.owned
            .iter()
            .filter(|owned| !owned.survives)
            .copied()
            .collect()
    }

    pub(crate) fn kept(&self) -> Vec<TypeId> {
        self.owned
            .iter()
            .filter(|owned| owned.survives)
            .map(|owned| owned.type_id)
            .collect()
    }

    fn section_taken(&mut self, key: &'static str) -> bool {
        let taken = self.settings.iter().any(|entry| entry.key == key)
            || self.stats.iter().any(|entry| entry.key == key);
        if taken {
            let path = format!("$.sections.{key}");
            self.problems
                .push(Error::new(ErrorCode::NameTwice, path).got(self.package));
        }
        taken
    }

    fn own(&mut self, owned: Owned) {
        if !self.owns(owned.type_id) {
            self.owned.push(owned);
        }
    }
}
