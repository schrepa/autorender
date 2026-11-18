use std::any::{Any, TypeId};
use std::sync::Arc;

use bevy_ecs::component::ComponentId;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{InternedScheduleLabel, IntoScheduleConfigs, ScheduleConfigs};
use bevy_ecs::system::ScheduleSystem;
use bevy_ecs::world::World;
use engine_place::{Error, ErrorCode, List};

use crate::entries::{JoinEntry, Kept, Owned, SettingsEntry, StatsEntry};
use crate::jobs::JobMember;
use crate::scene::SpawnedBy;
use crate::state::SwitchFn;

pub trait RecordExt {
    fn record(&mut self) -> &mut BrickRecord;
}

type MakeSystems = Box<dyn FnOnce() -> ScheduleConfigs<ScheduleSystem> + Send + Sync>;

type Turn = fn(&mut World);

pub(crate) struct SystemsCall {
    pub(crate) schedule: InternedScheduleLabel,
    pub(crate) make: MakeSystems,
}

pub struct BrickRecord {
    pub package: &'static str,
    game: bool,
    pub entries: Vec<Box<dyn Any + Send + Sync>>,
    pub render: Vec<Arc<dyn Any + Send + Sync>>,
    brick: TypeId,
    crate_name: &'static str,
    source: [u8; 32],
    prefix: Option<&'static str>,
    corpora: Vec<&'static (dyn Any + Send + Sync)>,
    switch: Option<(&'static str, SwitchFn)>,
    pub(crate) spawned: fn(&mut World) -> ComponentId,
    pub(crate) systems: Vec<SystemsCall>,
    pub(crate) joins: Vec<JoinEntry>,
    owned: Vec<Owned>,
    settings: Vec<SettingsEntry>,
    stats: Vec<StatsEntry>,
    problems: Vec<Error>,
}

impl BrickRecord {
    pub fn new<B: 'static>(
        package: &'static str,
        source: [u8; 32],
        corpora: Vec<&'static (dyn Any + Send + Sync)>,
    ) -> Self {
        let type_name = std::any::type_name::<B>();
        Self {
            package,
            game: false,
            entries: Vec::new(),
            render: Vec::new(),
            brick: TypeId::of::<B>(),
            crate_name: type_name.split("::").next().unwrap_or(type_name),
            source,
            prefix: None,
            corpora,
            switch: None,
            spawned: spawned_id::<B>,
            systems: Vec::new(),
            joins: Vec::new(),
            owned: Vec::new(),
            settings: Vec::new(),
            stats: Vec::new(),
            problems: Vec::new(),
        }
    }

    pub fn brick(&self) -> TypeId {
        self.brick
    }

    pub fn set_game(&mut self, game: bool) {
        self.game = game;
    }

    pub fn game(&self) -> bool {
        self.game
    }

    pub fn source(&self) -> [u8; 32] {
        self.source
    }

    pub fn prefix(&self) -> Option<&'static str> {
        self.prefix
    }

    pub(crate) fn crate_name(&self) -> &'static str {
        self.crate_name
    }

    pub fn corpora<C: Any>(&self) -> impl Iterator<Item = &'static C> + '_ {
        self.corpora
            .iter()
            .filter_map(|&corpus| corpus.downcast_ref::<C>())
    }

    pub fn switch_key(&self) -> Option<&'static str> {
        self.switch.map(|(key, _)| key)
    }

    pub(crate) fn switch_fn(&self) -> Option<SwitchFn> {
        self.switch.map(|(_, on)| on)
    }

    pub fn joins(&self) -> &[JoinEntry] {
        &self.joins
    }

    pub fn take_members(&mut self) -> Vec<JobMember> {
        self.joins
            .iter_mut()
            .filter_map(|joined| Some((joined.list, joined.take_member()?)))
            .collect()
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

    pub fn systems<M: 'static>(
        &mut self,
        schedule: InternedScheduleLabel,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M> + Send + Sync + 'static,
    ) {
        self.systems.push(SystemsCall {
            schedule,
            make: Box::new(move || systems.chain()),
        });
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

    pub(crate) fn dropped(&self) -> Vec<(Turn, Turn)> {
        self.owned
            .iter()
            .filter_map(|owned| match owned.kept {
                Kept::Dropped(remove) => Some((owned.init, remove)),
                Kept::Survives(_) => None,
            })
            .collect()
    }

    pub(crate) fn kept(&self) -> Vec<Turn> {
        self.owned
            .iter()
            .filter_map(|owned| match owned.kept {
                Kept::Survives(touch) => Some(touch),
                Kept::Dropped(_) => None,
            })
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

fn spawned_id<B: 'static>(world: &mut World) -> ComponentId {
    world.register_component::<SpawnedBy<B>>()
}
