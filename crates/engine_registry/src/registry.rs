use std::any::{Any, TypeId};
use std::collections::BTreeMap;

use bevy_ecs::change_detection::Res;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{
    Chain, InternedScheduleLabel, IntoScheduleConfigs, ScheduleBuildError, ScheduleConfigs,
    ScheduleGraph, Schedules, SystemSet,
};
use bevy_ecs::system::ScheduleSystem;
use bevy_ecs::world::World;
use engine_place::{Board, CORE_ROW, Callable, Error, ErrorCode};
use engine_settings::Expected;
use serde_json::Value;

use crate::entries::{OperationEntry, SettingsEntry, StatsEntry};
use crate::jobs::JobQueue;
use crate::record::BrickRecord;
use crate::state::{BrickStates, Despawn, Switches};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BrickId(pub(crate) u32);

impl BrickId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct BrickSet(&'static str);

fn unregistered(name: &str) -> Error {
    Error::new(ErrorCode::UnknownVariant, "$.name")
        .expected(Expected::Present)
        .got(name)
}

pub fn run_operation(world: &mut World, name: &str, request: Value) -> Result<Value, Error> {
    let operation = world
        .get_resource::<Registry>()
        .and_then(|registry| registry.operation(name))
        .copied()
        .ok_or_else(|| unregistered(name))?;
    Board::owned_by(operation.brick, || operation.run(world, request))
        .map_err(|error| error.within("$.args"))
}

pub fn call<O: Callable>(world: &mut World, request: O::Request) -> Result<O::Reply, Error> {
    let Some(brick) = world
        .get_resource::<Registry>()
        .and_then(|registry| registry.operation(O::NAME))
        .filter(|entry| entry.declares::<O>())
        .map(|entry| entry.brick)
    else {
        return Err(unregistered(O::NAME));
    };
    Board::owned_by(brick, || O::answer(world, request)).map_err(|error| error.within("$.args"))
}

#[derive(Resource, Default)]
pub struct Registry {
    bricks: Vec<BrickRecord>,
    core_sections: Vec<SettingsEntry>,
    core_operations: Vec<OperationEntry>,
    pub(crate) despawn: Option<Despawn>,
    problems: Vec<Error>,
    notices: Vec<Error>,
}

#[derive(Default)]
pub struct RegistryBuilder {
    registry: Registry,
}

impl RegistryBuilder {
    pub fn add(&mut self, mut record: BrickRecord) {
        let registry = &mut self.registry;
        registry.problems.append(&mut record.take_problems());
        let package = record.package;
        if registry.has(package) {
            let path = format!("$.bricks.{package}");
            registry
                .problems
                .push(Error::new(ErrorCode::NameTwice, path).got(package));
            return;
        }
        registry.check_names(&record);
        registry.bricks.push(record);
    }

    pub fn core_settings(&mut self, entry: SettingsEntry) {
        self.registry.core_settings(entry);
    }

    pub fn core_operation(&mut self, entry: OperationEntry) {
        self.registry.core_operation(entry);
    }

    pub fn despawn_rule(&mut self, despawn: Despawn) {
        self.registry.despawn = Some(despawn);
    }

    pub fn install(self, world: &mut World, schedules: &[InternedScheduleLabel]) -> Switches {
        let Self { mut registry } = self;
        registry.check_owned();
        for entry in &registry.core_sections {
            entry.owned().init(world);
        }
        for record in &registry.bricks {
            Board::owned_by(record.package, || {
                for owned in record.owned() {
                    owned.init(world);
                }
            });
        }
        registry.join_members(world);
        world.insert_resource(BrickStates::new(registry.bricks.len()));
        if !world.contains_resource::<JobQueue>() {
            world.init_resource::<JobQueue>();
        }
        registry.add_systems(world);
        let mut refused = build(world, schedules);
        registry.problems.append(&mut refused);
        let switches = registry.bricks.iter().map(BrickRecord::switch_fn).collect();
        world.insert_resource(registry);
        Switches::new(switches)
    }
}

impl Registry {
    pub fn core_settings(&mut self, mut entry: SettingsEntry) {
        entry.brick = CORE_ROW;
        let owner = self
            .section(entry.key)
            .map(|known| known.brick)
            .or_else(|| self.stats_section(entry.key).map(|known| known.brick));
        if let Some(owner) = owner {
            let path = format!("$.sections.{}", entry.key);
            self.problems.push(
                Error::new(ErrorCode::NameTwice, path)
                    .got(owner)
                    .nearest(CORE_ROW),
            );
            return;
        }
        self.core_sections.push(entry);
    }

    pub fn core_operation(&mut self, mut entry: OperationEntry) {
        entry.brick = CORE_ROW;
        if let Some(owner) = self.operation(entry.name).map(|known| known.brick) {
            let path = format!("$.operations.{}", entry.name);
            self.problems.push(
                Error::new(ErrorCode::NameTwice, path)
                    .got(owner)
                    .nearest(CORE_ROW),
            );
            return;
        }
        self.core_operations.push(entry);
    }

    pub fn report(&mut self, error: Error) {
        self.problems.push(error);
    }

    pub fn report_notice(&mut self, error: Error) {
        if !self.notices.contains(&error) {
            self.notices.push(error);
        }
    }

    pub fn notices(&self) -> &[Error] {
        &self.notices
    }

    pub fn problems(&self) -> &[Error] {
        &self.problems
    }

    pub fn bricks(&self) -> &[BrickRecord] {
        &self.bricks
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (BrickId, &BrickRecord)> + '_ {
        self.bricks
            .iter()
            .enumerate()
            .map(|(index, record)| (BrickId(index as u32), record))
    }

    pub fn names(&self) -> impl ExactSizeIterator<Item = &'static str> + '_ {
        self.bricks.iter().map(|record| record.package)
    }

    pub fn id(&self, package: &str) -> Option<BrickId> {
        self.bricks
            .iter()
            .position(|known| known.package == package)
            .map(|index| BrickId(index as u32))
    }

    pub fn has(&self, package: &str) -> bool {
        self.id(package).is_some()
    }

    pub fn brick(&self, package: &str) -> Option<&BrickRecord> {
        self.record(self.id(package)?)
    }

    pub fn record(&self, brick: BrickId) -> Option<&BrickRecord> {
        self.bricks.get(brick.index())
    }

    pub fn render_entries<T: Any>(&self) -> impl Iterator<Item = (BrickId, &T)> + '_ {
        self.iter()
            .flat_map(|(id, record)| record.render_entries::<T>().map(move |entry| (id, entry)))
    }

    pub fn sections(&self) -> impl Iterator<Item = &SettingsEntry> + '_ {
        self.core_sections.iter().chain(
            self.bricks
                .iter()
                .flat_map(|record| record.settings_entries()),
        )
    }

    pub fn section(&self, key: &str) -> Option<&SettingsEntry> {
        self.sections().find(|entry| entry.key == key)
    }

    pub fn stats(&self) -> impl Iterator<Item = &StatsEntry> + '_ {
        self.bricks.iter().flat_map(|record| record.stats_entries())
    }

    pub fn stats_section(&self, key: &str) -> Option<&StatsEntry> {
        self.stats().find(|entry| entry.key == key)
    }

    pub fn operations(&self) -> impl Iterator<Item = &OperationEntry> + '_ {
        self.core_operations
            .iter()
            .chain(self.bricks.iter().flat_map(|record| record.operations()))
    }

    pub fn operation(&self, name: &str) -> Option<&OperationEntry> {
        self.operations().find(|entry| entry.name == name)
    }

    pub fn check_section(&self, key: &str, value: &Value, path: &str) -> Result<(), Error> {
        self.section(key)
            .map_or(Ok(()), |entry| entry.check(value, path))
    }

    pub fn section_schema(&self, key: &str) -> Option<schemars::Schema> {
        Some(self.section(key)?.schema())
    }

    pub fn corpora<C: Any>(&self) -> impl Iterator<Item = (&'static str, &'static C)> + '_ {
        self.bricks.iter().flat_map(|record| {
            let owner = record.crate_name();
            record.corpora::<C>().map(move |corpus| (owner, corpus))
        })
    }

    pub fn frame_stats(&self, world: &World) -> BTreeMap<&'static str, Value> {
        self.stats()
            .map(|entry| (entry.key, entry.read(world)))
            .collect()
    }

    fn check_names(&mut self, record: &BrickRecord) {
        let package = record.package;
        let keys = record
            .settings_entries()
            .iter()
            .map(|entry| entry.key)
            .chain(record.stats_entries().iter().map(|entry| entry.key));
        for key in keys {
            let owner = self
                .section(key)
                .map(|entry| entry.brick)
                .or_else(|| self.stats_section(key).map(|entry| entry.brick));
            if let Some(owner) = owner {
                let path = format!("$.sections.{key}");
                self.problems.push(
                    Error::new(ErrorCode::NameTwice, path)
                        .got(owner)
                        .nearest(package),
                );
            }
        }
        if let Some(prefix) = record.prefix()
            && let Some(owner) = self
                .bricks
                .iter()
                .find(|known| known.prefix() == Some(prefix))
        {
            let path = format!("$.modules.{prefix}");
            self.problems.push(
                Error::new(ErrorCode::NameTwice, path)
                    .got(owner.package)
                    .nearest(package),
            );
        }
        for entry in record.operations() {
            if let Some(owner) = self.operation(entry.name).map(|known| known.brick) {
                let path = format!("$.operations.{}", entry.name);
                self.problems.push(
                    Error::new(ErrorCode::NameTwice, path)
                        .got(owner)
                        .nearest(package),
                );
            }
        }
    }

    fn check_owned(&mut self) {
        let mut seen: Vec<(TypeId, &'static str)> = Vec::new();
        for record in &self.bricks {
            for owned in record.owned() {
                match seen.iter().find(|(known, _)| *known == owned.type_id) {
                    Some(&(_, first)) => {
                        let path = format!("$.bricks.{}.registers.{}", record.package, owned.name);
                        self.problems.push(
                            Error::new(ErrorCode::NameTwice, path)
                                .got(first)
                                .nearest(record.package),
                        );
                    }
                    None => seen.push((owned.type_id, record.package)),
                }
            }
        }
    }

    fn join_members(&mut self, world: &mut World) {
        for (index, record) in self.bricks.iter_mut().enumerate() {
            let owner = (record.package, record.brick());
            for joined in &mut record.joins {
                (joined.init)(world);
                if let Some(member) = joined.member.take() {
                    (joined.add)(world, member, BrickId(index as u32), owner);
                }
            }
        }
    }

    fn add_systems(&mut self, world: &mut World) {
        let mut all = world.get_resource_or_init::<Schedules>();
        let mut last: Vec<(InternedScheduleLabel, BrickSet)> = Vec::new();
        for (index, record) in self.bricks.iter_mut().enumerate() {
            let (package, brick) = (record.package, BrickId(index as u32));
            let mut groups: Vec<Group> = Vec::new();
            for call in std::mem::take(&mut record.systems) {
                let configs = (call.make)();
                match groups
                    .iter_mut()
                    .find(|group| group.schedule == call.schedule)
                {
                    Some(group) => group.configs.push(configs),
                    None => groups.push(Group {
                        schedule: call.schedule,
                        configs: vec![configs],
                    }),
                }
            }
            for group in groups {
                let schedule = group.schedule;
                let mut chain = Chain::Unchained;
                chain.set_chained();
                let chained = ScheduleConfigs::Configs {
                    configs: group.configs,
                    collective_conditions: Vec::new(),
                    metadata: chain,
                };
                let before = crate::attributed::keys(all.entry(schedule));
                all.add_systems(schedule, chained.in_set(BrickSet(package)));
                crate::attributed::wrap(all.entry(schedule), &before, package);
                let set =
                    BrickSet(package).run_if(move |states: Res<BrickStates>| states.on(brick));
                match last.iter_mut().find(|(known, _)| *known == schedule) {
                    Some((_, previous)) => {
                        all.configure_sets(schedule, set.after(*previous));
                        *previous = BrickSet(package);
                    }
                    None => {
                        all.configure_sets(schedule, set);
                        last.push((schedule, BrickSet(package)));
                    }
                }
            }
        }
    }
}

struct Group {
    schedule: InternedScheduleLabel,
    configs: Vec<ScheduleConfigs<ScheduleSystem>>,
}

fn build(world: &mut World, labels: &[InternedScheduleLabel]) -> Vec<Error> {
    let mut refused: Vec<Error> = Vec::new();
    for &label in labels {
        let built =
            world.try_schedule_scope(label, |world, schedule| match schedule.initialize(world) {
                Ok(_) => None,
                Err(error) => Some(refusal(label, &error, schedule.graph(), world)),
            });
        if let Ok(Some(error)) = built {
            refused.push(error);
        }
    }
    refused
}

fn refusal(
    label: InternedScheduleLabel,
    error: &ScheduleBuildError,
    graph: &ScheduleGraph,
    world: &World,
) -> Error {
    let code = if matches!(
        error,
        ScheduleBuildError::DependencySort(_) | ScheduleBuildError::FlatDependencySort(_)
    ) {
        ErrorCode::PlaceCycle
    } else {
        ErrorCode::InvalidValue
    };
    let text = error.to_string(graph, world);
    let lines = text.lines().collect::<Vec<_>>();
    Error::new(code, format!("$.schedules.{label:?}")).got(lines)
}

pub fn settings(world: &World, key: &str) -> Result<SettingsEntry, Error> {
    world
        .get_resource::<Registry>()
        .and_then(|registry| registry.section(key).copied())
        .ok_or_else(|| Error::new(ErrorCode::UnknownValue, format!("$.settings.{key}")).got(key))
}
