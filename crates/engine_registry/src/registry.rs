use std::any::{Any, TypeId};
use std::collections::BTreeMap;

use bevy_ecs::component::ComponentId;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{
    Chain, InternedScheduleLabel, IntoScheduleConfigs, ScheduleConfigs, Schedules, SystemSet,
};
use bevy_ecs::system::{IntoSystem, ScheduleSystem};
use bevy_ecs::world::World;
use engine_place::{Error, ErrorCode, PlaceDecl, PlaceKind, Slot, Smoke};
use serde_json::Value;

use crate::access::{AccessRule, MoverRule};
use crate::attribution::CORE_ROW;
use crate::entries::{Kept, OperationEntry, PieceEntry, SettingsEntry, StatsEntry};
use crate::explain::ExplainEntry;
use crate::hooks::{Named, add_removal, add_settle};
use crate::jobs::JobQueue;
use crate::order::{self, PlaceRules};
use crate::owners::{Shared, Used, ValueKind, Values, note, ownership};
use crate::record::BrickRecord;
use crate::state::{
    BrickStates, Despawn, GateDriver, GateFn, GateId, Reset, Scope, Turning, state_on,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BrickId(pub(crate) u32);

impl BrickId {
    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn profile_index(self) -> u16 {
        u16::try_from(self.0 + 1).unwrap_or(u16::MAX)
    }
}

#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BrickSet(pub &'static str);

#[derive(Resource, Default)]
pub struct Registry {
    bricks: Vec<BrickRecord>,
    catalog: Vec<PlaceDecl>,
    core_fills: Vec<PlaceDecl>,
    core_sections: Vec<SettingsEntry>,
    core_operations: Vec<OperationEntry>,
    bases: Vec<u32>,
    left_out: Vec<&'static str>,
    rule: AccessRule,
    named: Vec<Named>,
    eachers: Vec<(TypeId, MemberEach)>,
    owners: Vec<Shared>,
    private: Vec<Vec<(ComponentId, Reset)>>,
    movers: Option<MoverRule>,
    despawn: Option<Despawn>,
    problems: Vec<Error>,
}

#[derive(Default)]
pub struct RegistryBuilder {
    registry: Registry,
    listed: Vec<&'static str>,
    left_owned: Vec<(TypeId, &'static str)>,
    firsts: Vec<TypeId>,
    orders: Vec<(&'static str, &'static str)>,
}

pub(crate) type MemberEach = fn(&World, &mut dyn FnMut(&dyn Any));

type Orders = Vec<(&'static str, &'static str)>;

impl RegistryBuilder {
    pub fn add(&mut self, mut record: BrickRecord) {
        let registry = &mut self.registry;
        registry.problems.append(&mut record.take_problems());
        let package = record.package;
        if record.owner().is_none() {
            let path = format!("$.bricks.{package}");
            registry
                .problems
                .push(Error::new(ErrorCode::ForeignAccess, path).got(record.type_name()));
        }
        let at = match registry
            .bricks
            .binary_search_by(|known| known.package.cmp(package))
        {
            Ok(_) => {
                let path = format!("$.bricks.{package}");
                registry
                    .problems
                    .push(Error::new(ErrorCode::NameTwice, path).got(package));
                return;
            }
            Err(at) => at,
        };
        registry.check_names(&record);
        registry.left_out.retain(|&left| left != package);
        self.listed.push(package);
        registry.bricks.insert(at, record);
    }

    pub fn places(&mut self, places: &'static [PlaceDecl]) {
        for decl in places {
            self.registry.catalog_add(*decl);
        }
    }

    pub fn fill_core<S: Slot>(&mut self) {
        let registry = &mut self.registry;
        let slot = TypeId::of::<S>();
        if let Some(filler) = registry.filler(slot) {
            let path = format!("$.places.{}", S::NAME);
            registry
                .problems
                .push(Error::new(ErrorCode::SlotFilledTwice, path).got(filler));
            return;
        }
        let decl = PlaceDecl::slot::<S>();
        registry.core_fills.push(decl);
        registry.catalog_add(decl);
    }

    pub fn core_settings(&mut self, entry: SettingsEntry) {
        self.registry.core_settings(entry);
    }

    pub fn core_operation(&mut self, entry: OperationEntry) {
        self.registry.core_operation(entry);
    }

    pub fn mover_rule(&mut self, rule: MoverRule) {
        self.registry.movers = Some(rule);
    }

    pub fn despawn_rule(&mut self, despawn: Despawn) {
        self.registry.despawn = Some(despawn);
    }

    pub fn writers_first<T: 'static>(&mut self) {
        let type_id = TypeId::of::<T>();
        if !self.firsts.contains(&type_id) {
            self.firsts.push(type_id);
        }
    }

    pub fn leave_out(&mut self, record: &BrickRecord) {
        let package = record.package;
        if self.registry.has(package) {
            return;
        }
        if !self.registry.left_out.contains(&package) {
            self.registry.left_out.push(package);
        }
        self.left_owned
            .extend(record.owned().iter().map(|owned| (owned.type_id, package)));
    }

    pub fn install(self, world: &mut World, schedules: &[InternedScheduleLabel]) -> GateDriver {
        let Self {
            mut registry,
            listed,
            left_owned,
            firsts,
            orders,
        } = self;
        registry.number();
        registry.catalog_records();
        registry.check_smokes();
        registry.rule = registry.rule_of(&left_owned);
        registry.check_owned();
        for decl in &registry.catalog {
            decl.insert(world);
        }
        for entry in &registry.core_sections {
            entry.owned().init(world);
        }
        for record in &registry.bricks {
            for owned in record.owned() {
                owned.init(world);
            }
        }
        registry.join_members(world);
        registry.resolve_named(world);
        world.insert_resource(registry.states());
        if !world.contains_resource::<JobQueue>() {
            world.init_resource::<JobQueue>();
        }
        let orders = registry.order_games(&listed, orders);
        registry.order_schedules(world, schedules, &firsts, &orders);
        registry.add_systems(world, &listed);
        let mut uses: Vec<Used> = Vec::new();
        let conditions = registry.conditions(world, &mut uses);
        registry.add_hooks(world, &mut uses);
        registry.check_operations(world, &mut uses);
        let (mut refused, mut found) = order::build(world, schedules);
        registry.problems.append(&mut refused);
        uses.append(&mut found);
        let needs = registry.analyze(world, &uses);
        world.resource_mut::<BrickStates>().set_needs(needs);
        world.insert_resource(registry);
        GateDriver::new(conditions)
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

    pub fn profile_names(&self) -> Vec<&'static str> {
        std::iter::once(CORE_ROW).chain(self.names()).collect()
    }

    pub fn left_out(&self) -> &[&'static str] {
        &self.left_out
    }

    pub fn holds(&self, package: &str, type_id: TypeId) -> bool {
        let owner = self.brick(package).and_then(BrickRecord::owner);
        owner.is_some()
            && self
                .bricks
                .iter()
                .any(|record| record.owner() == owner && record.owns(type_id))
    }

    pub fn ownership(&self) -> &[Shared] {
        &self.owners
    }

    pub(crate) fn private(&self, brick: BrickId) -> &[(ComponentId, Reset)] {
        self.private.get(brick.index()).map_or(&[], Vec::as_slice)
    }

    pub fn keep_authored(&mut self, authored: &[ComponentId]) {
        for resets in &mut self.private {
            resets.retain(|(id, reset)| {
                !(authored.contains(id) && matches!(reset, Reset::Remove(_)))
            });
        }
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

    pub fn place_members(&self, place: &str) -> Vec<&'static str> {
        let named = |decl: &PlaceDecl| decl.name() == place;
        self.bricks
            .iter()
            .filter(|record| {
                record.fills().iter().any(|filled| named(&filled.place))
                    || record.joins().iter().any(|joined| named(&joined.place))
            })
            .map(|record| record.package)
            .collect()
    }

    pub fn each_slot(&self, world: &World, slot: TypeId, each: &mut dyn FnMut(&dyn Any)) {
        if let Some(decl) = self
            .core_fills
            .iter()
            .chain(&self.catalog)
            .find(|decl| decl.kind() == PlaceKind::Slot && PlaceDecl::type_id(decl) == slot)
        {
            decl.read(world, each);
        }
    }

    pub fn error_codes(&self) -> Vec<&'static str> {
        let mut codes = ErrorCode::NAMES.to_vec();
        for code in self.bricks.iter().flat_map(BrickRecord::error_codes) {
            if !codes.contains(&code.as_str()) {
                codes.push(code.as_str());
            }
        }
        codes
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

    pub fn explainers(&self) -> impl Iterator<Item = &ExplainEntry> + '_ {
        self.bricks.iter().flat_map(|record| record.explainers())
    }

    pub fn check_named(&self, key: &str, value: &Value, path: &str) -> Result<(), Error> {
        self.named
            .iter()
            .filter(|named| named.section == key)
            .try_for_each(|named| named.check(value, path))
    }

    pub fn check_section(&self, key: &str, value: &Value, path: &str) -> Result<(), Error> {
        let Some(entry) = self.section(key) else {
            return Ok(());
        };
        if self.named.iter().any(|named| named.section == key) {
            entry.parse(value, path)?;
            self.check_named(key, value, path)?;
        }
        entry.check(value, path)
    }

    pub fn section_schema(&self, key: &str) -> Option<schemars::Schema> {
        let mut schema = self.section(key)?.schema();
        for named in self.named.iter().filter(|named| named.section == key) {
            named.schema(&mut schema);
        }
        Some(schema)
    }

    pub fn slot_on(&self, states: &BrickStates, slot: TypeId) -> bool {
        if self.core_fills.iter().any(|decl| decl.type_id() == slot) {
            return true;
        }
        self.iter()
            .find_map(|(brick, record)| {
                record
                    .fills()
                    .iter()
                    .find(|filled| filled.slot == slot)
                    .map(|filled| (brick, filled.scope))
            })
            .and_then(|(brick, scope)| self.gate(brick, scope))
            .is_some_and(|gate| states.on(gate))
    }

    pub fn each_member(&self, world: &World, list: TypeId, each: &mut dyn FnMut(&dyn Any)) {
        if let Some((_, members)) = self.eachers.iter().find(|(known, _)| *known == list) {
            members(world, each);
        }
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

    pub(crate) fn turning(&self, brick: BrickId, gate: GateId) -> Turning {
        let (Some(record), Some(&base)) = (self.record(brick), self.bases.get(brick.index())) else {
            return Turning::default();
        };
        let scope = Scope(gate.0.saturating_sub(base));
        let slots = record
            .fills()
            .iter()
            .filter(|filled| filled.scope == scope)
            .map(|filled| filled.reset)
            .collect();
        let joins = record
            .joins()
            .iter()
            .filter(|joined| joined.scope == scope)
            .map(|joined| joined.left)
            .collect();
        if scope != Scope::BRICK {
            return Turning {
                slots,
                joins,
                ..Turning::default()
            };
        }
        let mut turning = Turning {
            slots,
            joins,
            private: self.private(brick).iter().map(|&(_, reset)| reset).collect(),
            spawned: Some(record.spawned),
            despawn: self.despawn,
            ..Turning::default()
        };
        for owned in record.owned() {
            match owned.kept {
                Kept::Survives(touch) => turning.settings.push(touch),
                Kept::Dropped(_) if self.rule.catalogued(owned.type_id) => {}
                Kept::Dropped(remove) => turning.resources.push((owned.init, remove)),
            }
        }
        turning
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

    fn rule_of(&self, left_owned: &[(TypeId, &'static str)]) -> AccessRule {
        let mut rule = AccessRule::new(
            self.bricks
                .iter()
                .map(|record| (record.package, record.owner()))
                .collect(),
            self.bricks.iter().map(BrickRecord::game).collect(),
            self.movers,
        );
        for decl in &self.catalog {
            let filler = match decl.kind() {
                PlaceKind::Slot => self.filler_id(decl.type_id()),
                PlaceKind::List | PlaceKind::Component | PlaceKind::Ledger => None,
            };
            rule.place(decl.type_id(), decl.kind(), filler);
        }
        for &(type_id, package) in left_owned {
            if !self.bricks.iter().any(|record| record.owns(type_id)) {
                rule.left_out(type_id, package);
            }
        }
        rule
    }

    fn check_owned(&mut self) {
        let mut seen: Vec<(TypeId, &'static str)> = Vec::new();
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
            let base = self.bases[index];
            let (package, brick) = (record.package, record.brick());
            for joined in &mut record.joins {
                (joined.init)(world);
                if let Some(member) = joined.member.take() {
                    let owner = (package, brick);
                    (joined.add)(world, member, GateId(base + joined.scope.0), owner);
                }
                if !self.eachers.iter().any(|(list, _)| *list == joined.list) {
                    self.eachers.push((joined.list, joined.each));
                }
            }
        }
    }

    fn resolve_named(&mut self, world: &World) {
        for record in &mut self.bricks {
            for named in std::mem::take(&mut record.named) {
                if !record
                    .settings_entries()
                    .iter()
                    .any(|entry| entry.key == named.section)
                {
                    let path = format!("$.bricks.{}.named.{}", record.package, named.section);
                    self.problems
                        .push(Error::new(ErrorCode::ForeignAccess, path).got(named.section));
                    continue;
                }
                let variants = named
                    .kept
                    .iter()
                    .copied()
                    .chain((named.names)(world))
                    .collect();
                self.named.push(Named {
                    section: named.section,
                    field: named.field,
                    variants,
                });
            }
        }
    }

    fn add_hooks(&mut self, world: &mut World, uses: &mut Vec<Used>) {
        for (index, record) in self.bricks.iter_mut().enumerate() {
            let brick = BrickId(index as u32);
            let root = GateId(self.bases[index]);
            let package = record.package;
            for mut removal in record.removals.drain(..) {
                let access = removal.system.initialize(world);
                let name = removal.system.name();
                self.rule
                    .check_into(brick, &name, &access, world, &mut self.problems);
                note(uses, brick, &access);
                if let Err(error) = add_removal(world, package, root, removal) {
                    self.problems.push(error);
                }
            }
            for mut settle in record.settles.drain(..) {
                let access = settle.system.initialize(world);
                let name = settle.system.name();
                self.rule
                    .check_into(brick, &name, &access, world, &mut self.problems);
                note(uses, brick, &access);
                add_settle(world, package, root, settle);
            }
        }
    }

    fn states(&self) -> BrickStates {
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
        let roots = self.bases.iter().map(|&base| GateId(base)).collect();
        let types = self.bricks.iter().map(BrickRecord::brick).collect();
        BrickStates::new(parents, bricks, roots, types)
    }

    fn order_games(&self, listed: &[&'static str], mut orders: Orders) -> Orders {
        let mut order = |first: &'static str, then: &'static str| {
            if !orders.contains(&(first, then)) {
                orders.push((first, then));
            }
        };
        let games: Vec<&'static str> = listed
            .iter()
            .copied()
            .filter(|&package| self.brick(package).is_some_and(BrickRecord::game))
            .collect();
        let engine: Vec<&'static str> = self
            .bricks
            .iter()
            .filter(|record| !record.game())
            .map(|record| record.package)
            .collect();
        for (at, &game) in games.iter().enumerate() {
            for &brick in &engine {
                order(brick, game);
            }
            for &later in &games[at + 1..] {
                order(game, later);
            }
        }
        orders
    }

    fn order_schedules(
        &self,
        world: &mut World,
        schedules: &[InternedScheduleLabel],
        firsts: &[TypeId],
        orders: &[(&'static str, &'static str)],
    ) {
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
        let firsts: Box<[ComponentId]> = firsts
            .iter()
            .filter_map(|&type_id| world.components().get_id(type_id))
            .collect();
        let orders: Box<[(BrickId, BrickId)]> = orders
            .iter()
            .filter_map(|&(first, then)| Some((self.id(first)?, self.id(then)?)))
            .collect();
        let shared: Box<[ComponentId]> = self
            .catalog
            .iter()
            .filter(|decl| decl.kind() != PlaceKind::Slot)
            .map(PlaceDecl::type_id)
            .chain(self.movers.map(|rule| rule.transform))
            .filter_map(|type_id| world.components().get_id(type_id))
            .collect();
        let mut all = world.get_resource_or_init::<Schedules>();
        for &schedule in schedules {
            let rules = PlaceRules {
                schedule,
                bricks: bricks.clone(),
                slots: slots.clone(),
                firsts: firsts.clone(),
                orders: orders.clone(),
                shared: shared.clone(),
                rule: self.rule.clone(),
            };
            order::order(all.entry(schedule), rules);
        }
    }

    fn add_systems(&mut self, world: &mut World, listed: &[&'static str]) {
        let mut all = world.get_resource_or_init::<Schedules>();
        for &package in listed {
            let Some(brick) = self.id(package) else {
                continue;
            };
            let calls = std::mem::take(&mut self.bricks[brick.index()].systems);
            let mut groups: Vec<Group> = Vec::new();
            for call in calls {
                let mut configs = (call.make)();
                if let Some(gate) = self.gate(brick, call.scope) {
                    configs.run_if_dyn(Box::new(IntoSystem::into_system(state_on(gate))));
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
                #[cfg(feature = "profiling")]
                let before = crate::measured::keys(all.entry(group.schedule));
                all.add_systems(group.schedule, chained.in_set(BrickSet(package)));
                #[cfg(feature = "profiling")]
                crate::measured::wrap(
                    all.entry(group.schedule),
                    &before,
                    (package, brick.profile_index()),
                );
            }
        }
    }

    fn conditions(&mut self, world: &mut World, uses: &mut Vec<Used>) -> Vec<Option<GateFn>> {
        let mut conditions: Vec<Option<GateFn>> = Vec::new();
        for (index, record) in self.bricks.iter_mut().enumerate() {
            let brick = BrickId(index as u32);
            for scope in &mut record.scopes {
                let condition = scope.condition.take().map(|mut condition| {
                    let access = condition.initialize(world);
                    let name = condition.name();
                    self.rule
                        .check_into(brick, &name, &access, world, &mut self.problems);
                    note(uses, brick, &access);
                    condition
                });
                conditions.push(condition);
            }
        }
        conditions
    }

    fn analyze(&mut self, world: &World, uses: &[Used]) -> Vec<Vec<BrickId>> {
        let rule = &self.rule;
        let catalog = &self.catalog;
        let catalogued = |type_id: TypeId| {
            rule.catalogued(type_id).then(|| {
                catalog
                    .iter()
                    .find(|decl| PlaceDecl::type_id(decl) == type_id)
                    .and_then(PlaceDecl::reset)
            })
        };
        let dropped: Vec<Vec<TypeId>> = self
            .bricks
            .iter()
            .map(|record| {
                record
                    .owned()
                    .iter()
                    .filter(|owned| !owned.survives() && !rule.catalogued(owned.type_id))
                    .map(|owned| owned.type_id)
                    .collect()
            })
            .collect();
        let values = Values {
            bricks: &self.bricks,
            catalogued: &catalogued,
            dropped: &dropped,
        };
        let held = ownership(world, &values, uses);
        for shared in &held.shared {
            let kind = match shared.kind {
                ValueKind::Resource => "resources",
                ValueKind::Component => "components",
            };
            self.problems.push(
                Error::new(ErrorCode::WrittenTwice, format!("$.{kind}.{}", shared.value))
                    .got(&shared.bricks),
            );
        }
        self.owners = held.shared;
        self.private = held.private;
        held.needs
    }

    fn check_operations(&mut self, world: &mut World, uses: &mut Vec<Used>) {
        let operations: Vec<(BrickId, OperationEntry)> = self
            .iter()
            .flat_map(|(id, record)| record.operations().iter().map(move |entry| (id, *entry)))
            .collect();
        for (brick, entry) in operations {
            let access = entry.access(world);
            self.rule
                .check_into(brick, entry.name, &access, world, &mut self.problems);
            note(uses, brick, &access);
        }
        let explainers: Vec<(BrickId, ExplainEntry)> = self
            .iter()
            .flat_map(|(id, record)| record.explainers().iter().map(move |entry| (id, *entry)))
            .collect();
        for (brick, entry) in explainers {
            let access = entry.access(world);
            self.rule
                .check_into(brick, "explain", &access, world, &mut self.problems);
            note(uses, brick, &access);
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

pub fn check_sections(world: &World) -> Result<(), Error> {
    let Some(registry) = world.get_resource::<Registry>() else {
        return Ok(());
    };
    registry
        .bricks
        .iter()
        .flat_map(|record| record.section_rules())
        .try_for_each(|rule| rule(world))
}
