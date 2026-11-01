use std::any::{Any, TypeId};
use std::collections::BTreeMap;

use bevy_ecs::component::ComponentId;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{
    Chain, InternedScheduleLabel, IntoScheduleConfigs, ScheduleConfigs, Schedules, SystemSet,
};
use bevy_ecs::system::{IntoSystem, ScheduleSystem, System};
use bevy_ecs::world::World;
use engine_place::{Error, ErrorCode, PlaceDecl, PlaceKind, Slot, Smoke};
use serde_json::Value;

use crate::access::AccessRule;
use crate::attribution::CORE_ROW;
use crate::entries::{OperationEntry, PieceEntry, SettingsEntry, StatsEntry};
use crate::gates::{Gate, GateDriver, GateId, Gates, Off, Scope, gate_on};
use crate::jobs::JobQueue;
use crate::order::{self, PlaceRules};
use crate::record::{BrickRecord, crate_segment, is_package};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BrickId(pub(crate) u32);

impl BrickId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BrickSet(pub &'static str);

#[derive(Resource, Default)]
pub struct Registry {
    bricks: Vec<BrickRecord>,
    catalog: Vec<PlaceDecl>,
    core_fills: Vec<PlaceDecl>,
    bases: Vec<u32>,
    listed: Vec<&'static str>,
    left_out: Vec<&'static str>,
    rule: AccessRule,
    problems: Vec<Error>,
}

impl Registry {
    pub fn add(&mut self, mut record: BrickRecord) {
        self.problems.append(&mut record.take_problems());
        let package = record.package;
        if !is_package(crate_segment(record.type_name()), package) {
            let path = format!("$.bricks.{package}");
            self.problems
                .push(Error::new(ErrorCode::ForeignAccess, path).got(record.type_name()));
        }
        let at = match self
            .bricks
            .binary_search_by(|known| known.package.cmp(package))
        {
            Ok(_) => {
                let path = format!("$.bricks.{package}");
                self.problems
                    .push(Error::new(ErrorCode::NameTwice, path).got(package));
                return;
            }
            Err(at) => at,
        };
        self.check_names(&record);
        self.left_out.retain(|&left| left != package);
        self.listed.push(package);
        self.bricks.insert(at, record);
    }

    pub fn places(&mut self, places: &'static [PlaceDecl]) {
        for decl in places {
            self.catalog_add(*decl);
        }
    }

    pub fn fill_core<S: Slot>(&mut self) {
        let slot = TypeId::of::<S>();
        if let Some(filler) = self.filler(slot) {
            let path = format!("$.places.{}", S::NAME);
            self.problems
                .push(Error::new(ErrorCode::SlotFilledTwice, path).got(filler));
            return;
        }
        let decl = PlaceDecl::slot::<S>();
        self.core_fills.push(decl);
        self.catalog_add(decl);
    }

    pub fn leave_out(&mut self, package: &'static str) {
        if !self.left_out.contains(&package) && !self.has(package) {
            self.left_out.push(package);
        }
    }

    pub fn left_out(&self) -> &[&'static str] {
        &self.left_out
    }

    pub fn report(&mut self, error: Error) {
        self.problems.push(error);
    }

    pub fn problems(&self) -> &[Error] {
        &self.problems
    }

    pub fn take_problems(&mut self) -> Vec<Error> {
        std::mem::take(&mut self.problems)
    }

    pub fn access_rule(&self) -> &AccessRule {
        &self.rule
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
            .binary_search_by(|known| known.package.cmp(package))
            .ok()
            .map(|index| BrickId(index as u32))
    }

    pub fn has(&self, package: &str) -> bool {
        self.id(package).is_some()
    }

    pub fn brick(&self, package: &str) -> Option<&BrickRecord> {
        self.record(self.id(package)?)
    }

    pub fn brick_mut(&mut self, package: &str) -> Option<&mut BrickRecord> {
        let id = self.id(package)?;
        self.bricks.get_mut(id.index())
    }

    pub fn record(&self, brick: BrickId) -> Option<&BrickRecord> {
        self.bricks.get(brick.index())
    }

    pub fn of_type(&self, brick: TypeId) -> Option<(BrickId, &BrickRecord)> {
        self.iter().find(|(_, record)| record.brick() == brick)
    }

    pub fn render_entries<T: Any>(&self) -> impl Iterator<Item = (BrickId, &T)> + '_ {
        self.iter()
            .flat_map(|(id, record)| record.render_entries::<T>().map(move |entry| (id, entry)))
    }

    pub fn system_set(&self, brick: BrickId) -> Option<BrickSet> {
        self.record(brick).map(|record| BrickSet(record.package))
    }

    pub fn gate(&self, brick: BrickId, scope: Scope) -> Option<GateId> {
        let base = *self.bases.get(brick.index())?;
        let record = self.record(brick)?;
        (scope.index() < record.scopes()).then_some(GateId(base + scope.0))
    }

    pub fn brick_gate(&self, brick: BrickId) -> Option<GateId> {
        self.gate(brick, Scope::BRICK)
    }

    pub fn catalog(&self) -> &[PlaceDecl] {
        &self.catalog
    }

    pub fn place(&self, name: &str) -> Option<&PlaceDecl> {
        self.catalog.iter().find(|decl| decl.name() == name)
    }

    pub fn filler(&self, slot: TypeId) -> Option<&'static str> {
        if self.core_fills.iter().any(|decl| decl.type_id() == slot) {
            return Some(CORE_ROW);
        }
        self.bricks
            .iter()
            .find(|record| record.fills_slot(slot))
            .map(|record| record.package)
    }

    pub fn filler_id(&self, slot: TypeId) -> Option<BrickId> {
        self.iter()
            .find(|(_, record)| record.fills_slot(slot))
            .map(|(id, _)| id)
    }

    pub fn filler_named(&self, name: &str) -> Option<&'static str> {
        let decl = self.place(name)?;
        (decl.kind() == PlaceKind::Slot)
            .then(|| self.filler(decl.type_id()))
            .flatten()
    }

    pub fn members(&self, list: TypeId) -> impl Iterator<Item = &'static str> + '_ {
        self.bricks
            .iter()
            .filter(move |record| record.joins().iter().any(|joined| joined.list == list))
            .map(|record| record.package)
    }

    pub fn sections(&self) -> impl Iterator<Item = &SettingsEntry> + '_ {
        self.bricks
            .iter()
            .flat_map(|record| record.settings_entries())
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
        self.bricks.iter().flat_map(|record| record.operations())
    }

    pub fn operation(&self, name: &str) -> Option<&OperationEntry> {
        self.operations().find(|entry| entry.name == name)
    }

    pub fn smokes(&self) -> impl Iterator<Item = (&'static str, &'static Smoke)> + '_ {
        let bricks = self.bricks.iter().flat_map(|record| {
            let package = record.package;
            record.smokes().iter().map(move |smoke| (package, smoke))
        });
        let places = self.catalog.iter().flat_map(|decl| {
            let name = decl.name();
            decl.smokes().iter().map(move |smoke| (name, smoke))
        });
        bricks.chain(places)
    }

    pub fn smoke(&self, name: &str) -> Option<(&'static str, &'static Smoke)> {
        self.smokes().find(|(_, smoke)| smoke.name == name)
    }

    pub fn pieces(&self) -> impl Iterator<Item = (BrickId, &PieceEntry)> + '_ {
        self.iter()
            .filter_map(|(id, record)| record.piece_entry().map(|piece| (id, piece)))
    }

    pub fn frame_stats(&self, world: &World) -> BTreeMap<&'static str, Value> {
        self.stats()
            .map(|entry| (entry.key, entry.read(world)))
            .collect()
    }

    pub fn pending(&self, world: &World) -> u32 {
        self.stats()
            .fold(0_u32, |sum, entry| sum.saturating_add(entry.pending(world)))
    }

    pub fn install(mut self, world: &mut World, schedules: &[InternedScheduleLabel]) -> GateDriver {
        self.number();
        self.catalog_records();
        self.check_smokes();
        self.rule = self.rule_of();
        self.check_owned();
        for decl in &self.catalog {
            decl.insert(world);
        }
        for record in &self.bricks {
            for owned in record.owned() {
                owned.init(world);
            }
        }
        let refresh = self.join_members(world);
        world.insert_resource(self.gates());
        if !world.contains_resource::<JobQueue>() {
            world.init_resource::<JobQueue>();
        }
        self.order_schedules(world, schedules);
        self.add_systems(world);
        let driver = self.driver(world, refresh);
        self.check_operations(world);
        let mut refused = order::build(world, schedules);
        self.problems.append(&mut refused);
        world.insert_resource(self);
        driver
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
        for filled in record.fills() {
            if let Some(owner) = self.filler(filled.slot) {
                let path = format!("$.places.{}", filled.place.name());
                self.problems.push(
                    Error::new(ErrorCode::SlotFilledTwice, path)
                        .got(owner)
                        .nearest(package),
                );
            }
        }
    }

    fn catalog_add(&mut self, decl: PlaceDecl) {
        match self
            .catalog
            .iter()
            .find(|known| known.name() == decl.name())
        {
            Some(known) if known.type_id() == decl.type_id() => {}
            Some(known) => {
                let path = format!("$.places.{}", decl.name());
                let error = Error::new(ErrorCode::NameTwice, path).got(known.name());
                self.problems.push(error);
            }
            None => self.catalog.push(decl),
        }
    }

    fn number(&mut self) {
        self.bases.clear();
        let mut next = 0_u32;
        for record in &self.bricks {
            self.bases.push(next);
            next += record.scopes() as u32;
        }
    }

    fn catalog_records(&mut self) {
        let decls: Vec<PlaceDecl> = self
            .bricks
            .iter()
            .flat_map(|record| record.places().copied())
            .collect();
        for decl in decls {
            self.catalog_add(decl);
        }
        self.catalog.sort_by(|a, b| a.name().cmp(b.name()));
    }

    fn check_smokes(&mut self) {
        let mut seen: Vec<(&'static str, &'static str)> = Vec::new();
        let mut twice: Vec<Error> = Vec::new();
        for (owner, smoke) in self.smokes() {
            match seen.iter().find(|(name, _)| *name == smoke.name) {
                Some(&(_, first)) => {
                    let path = format!("$.smokes.{}", smoke.name);
                    twice.push(
                        Error::new(ErrorCode::NameTwice, path)
                            .got(first)
                            .nearest(owner),
                    );
                }
                None => seen.push((smoke.name, owner)),
            }
        }
        self.problems.append(&mut twice);
    }

    fn rule_of(&self) -> AccessRule {
        let mut rule = AccessRule::new(self.names().collect());
        for decl in &self.catalog {
            let filler = match decl.kind() {
                PlaceKind::Slot => self.filler_id(decl.type_id()),
                PlaceKind::List | PlaceKind::Component => None,
            };
            rule.place(decl.type_id(), decl.kind(), filler);
        }
        rule
    }

    fn check_owned(&mut self) {
        for (index, record) in self.bricks.iter().enumerate() {
            let brick = BrickId(index as u32);
            let piece = record.piece_entry().map(|piece| piece.type_id);
            for owned in record.owned() {
                let filled_slot = piece == Some(owned.type_id);
                self.rule.check_type(
                    brick,
                    owned.type_id,
                    owned.name,
                    filled_slot,
                    &mut self.problems,
                );
            }
        }
    }

    fn join_members(&mut self, world: &mut World) -> Vec<fn(&mut World)> {
        let mut lists: Vec<(TypeId, fn(&mut World))> = Vec::new();
        for (index, record) in self.bricks.iter_mut().enumerate() {
            let base = self.bases[index];
            let package = record.package;
            for joined in &mut record.joins {
                (joined.init)(world);
                if let Some(member) = joined.member.take() {
                    (joined.add)(world, member, GateId(base + joined.scope.0), package);
                }
                if !lists.iter().any(|(list, _)| *list == joined.list) {
                    lists.push((joined.list, joined.refresh));
                }
            }
        }
        lists.into_iter().map(|(_, refresh)| refresh).collect()
    }

    fn gates(&self) -> Gates {
        let mut parents: Vec<Option<GateId>> = Vec::new();
        let mut bricks: Vec<BrickId> = Vec::new();
        for (index, record) in self.bricks.iter().enumerate() {
            let base = self.bases[index];
            for local in 0..record.scopes() {
                let scope = Scope(local as u32);
                parents.push(record.parent(scope).map(|parent| GateId(base + parent.0)));
                bricks.push(BrickId(index as u32));
            }
        }
        Gates::new(parents, bricks)
    }

    fn order_schedules(&self, world: &mut World, schedules: &[InternedScheduleLabel]) {
        let bricks: Box<[_]> = self
            .iter()
            .map(|(id, record)| (BrickSet(record.package).intern(), id))
            .collect();
        let slots: Box<[(ComponentId, Option<BrickId>)]> = self
            .catalog
            .iter()
            .filter(|decl| decl.kind() == PlaceKind::Slot)
            .filter_map(|decl| {
                let id = world.components().get_id(decl.type_id())?;
                Some((id, self.filler_id(decl.type_id())))
            })
            .collect();
        let mut all = world.get_resource_or_init::<Schedules>();
        for &schedule in schedules {
            let rules = PlaceRules {
                schedule,
                bricks: bricks.clone(),
                slots: slots.clone(),
                rule: self.rule.clone(),
            };
            order::order(all.entry(schedule), rules);
        }
    }

    fn add_systems(&mut self, world: &mut World) {
        let mut all = world.get_resource_or_init::<Schedules>();
        for index in 0..self.listed.len() {
            let package = self.listed[index];
            let Some(brick) = self.id(package) else {
                continue;
            };
            let calls = std::mem::take(&mut self.bricks[brick.index()].systems);
            let mut groups: Vec<Group> = Vec::new();
            for call in calls {
                let mut configs = (call.make)();
                if self.bricks[brick.index()].gated(call.scope)
                    && let Some(gate) = self.gate(brick, call.scope)
                {
                    configs.run_if_dyn(Box::new(IntoSystem::into_system(gate_on(gate))));
                }
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
                let mut chain = Chain::Unchained;
                chain.set_chained();
                let chained = ScheduleConfigs::Configs {
                    configs: group.configs,
                    collective_conditions: Vec::new(),
                    metadata: chain,
                };
                all.add_systems(group.schedule, chained.in_set(BrickSet(package)));
            }
        }
    }

    fn driver(&mut self, world: &mut World, refresh: Vec<fn(&mut World)>) -> GateDriver {
        let mut gates: Vec<Gate> = Vec::new();
        let mut offs: Vec<Off> = Vec::new();
        let mut resets: Vec<(GateId, fn(&mut World))> = Vec::new();
        for (index, record) in self.bricks.iter_mut().enumerate() {
            let brick = BrickId(index as u32);
            let base = self.bases[index];
            let package = record.package;
            for (local, scope) in record.scopes.iter_mut().enumerate() {
                let Some(mut condition) = scope.condition.take() else {
                    continue;
                };
                let access = condition.initialize(world);
                let name = condition.name();
                self.rule
                    .check_into(brick, &name, &access, world, &mut self.problems);
                gates.push(Gate {
                    id: GateId(base + local as u32),
                    brick: package,
                    condition,
                    on: true,
                });
            }
            for mut system in record.offs.drain(..) {
                let access = system.initialize(world);
                let name = system.name();
                self.rule
                    .check_into(brick, &name, &access, world, &mut self.problems);
                offs.push(Off {
                    root: GateId(base),
                    brick: package,
                    system,
                });
            }
            for filled in &record.fills {
                resets.push((GateId(base + filled.scope.0), filled.reset));
            }
        }
        GateDriver::new(gates, offs, resets, refresh)
    }

    fn check_operations(&mut self, world: &mut World) {
        let operations: Vec<(BrickId, OperationEntry)> = self
            .iter()
            .flat_map(|(id, record)| record.operations().iter().map(move |entry| (id, *entry)))
            .collect();
        for (brick, entry) in operations {
            let access = entry.access(world);
            self.rule
                .check_into(brick, entry.name, &access, world, &mut self.problems);
        }
    }
}

struct Group {
    schedule: InternedScheduleLabel,
    configs: Vec<ScheduleConfigs<ScheduleSystem>>,
}

pub fn settings(world: &World, key: &str) -> Result<SettingsEntry, Error> {
    world
        .get_resource::<Registry>()
        .and_then(|registry| registry.section(key).copied())
        .ok_or_else(|| Error::new(ErrorCode::UnknownValue, format!("$.settings.{key}")).got(key))
}
