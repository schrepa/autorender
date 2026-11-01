use std::any::{Any, TypeId};

use bevy_ecs::component::Mutable;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{InternedScheduleLabel, IntoScheduleConfigs, ScheduleConfigs};
use bevy_ecs::system::{BoxedSystem, ScheduleSystem};
use engine_place::{Error, ErrorCode, GpuStruct, List, PlaceDecl, Slot, Smoke};

use crate::entries::{
    Filled, JoinEntry, OperationEntry, Owned, PieceEntry, SettingsEntry, StatsEntry, init_of,
};
use crate::explain::ExplainEntry;
use crate::gates::{GateFn, Scope};
use crate::hooks::{NamedField, Removal, Settle};
use crate::jobs::JobMember;

pub trait RecordExt {
    fn record(&mut self) -> &mut BrickRecord;
}

pub(crate) struct ScopeEntry {
    pub(crate) parent: Option<Scope>,
    pub(crate) condition: Option<GateFn>,
    gated: bool,
}

type MakeSystems = Box<dyn FnOnce() -> ScheduleConfigs<ScheduleSystem> + Send + Sync>;

pub(crate) struct SystemsCall {
    pub(crate) scope: Scope,
    pub(crate) schedule: InternedScheduleLabel,
    pub(crate) make: MakeSystems,
}

pub struct BrickRecord {
    pub package: &'static str,
    pub entries: Vec<Box<dyn Any + Send + Sync>>,
    pub render: Vec<Box<dyn Any + Send + Sync>>,
    brick: TypeId,
    type_name: &'static str,
    source: [u8; 32],
    prefix: Option<&'static str>,
    smokes: &'static [Smoke],
    pub(crate) scopes: Vec<ScopeEntry>,
    current: Scope,
    pub(crate) offs: Vec<BoxedSystem>,
    pub(crate) systems: Vec<SystemsCall>,
    pub(crate) fills: Vec<Filled>,
    pub(crate) joins: Vec<JoinEntry>,
    owned: Vec<Owned>,
    settings: Vec<SettingsEntry>,
    stats: Vec<StatsEntry>,
    operations: Vec<OperationEntry>,
    explainers: Vec<ExplainEntry>,
    piece: Option<PieceEntry>,
    pub(crate) removals: Vec<Removal>,
    pub(crate) settles: Vec<Settle>,
    pub(crate) named: Vec<NamedField>,
    problems: Vec<Error>,
}

impl BrickRecord {
    pub fn new<B: 'static>(
        package: &'static str,
        source: [u8; 32],
        smokes: &'static [Smoke],
    ) -> Self {
        Self {
            package,
            entries: Vec::new(),
            render: Vec::new(),
            brick: TypeId::of::<B>(),
            type_name: std::any::type_name::<B>(),
            source,
            prefix: None,
            smokes,
            scopes: vec![ScopeEntry {
                parent: None,
                condition: None,
                gated: false,
            }],
            current: Scope::BRICK,
            offs: Vec::new(),
            systems: Vec::new(),
            fills: Vec::new(),
            joins: Vec::new(),
            owned: Vec::new(),
            settings: Vec::new(),
            stats: Vec::new(),
            operations: Vec::new(),
            explainers: Vec::new(),
            piece: None,
            removals: Vec::new(),
            settles: Vec::new(),
            named: Vec::new(),
            problems: Vec::new(),
        }
    }

    pub fn brick(&self) -> TypeId {
        self.brick
    }

    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    pub fn source(&self) -> [u8; 32] {
        self.source
    }

    pub fn prefix(&self) -> Option<&'static str> {
        self.prefix
    }

    pub fn smokes(&self) -> &'static [Smoke] {
        self.smokes
    }

    pub fn scope(&self) -> Scope {
        self.current
    }

    pub fn scopes(&self) -> usize {
        self.scopes.len()
    }

    pub fn parent(&self, scope: Scope) -> Option<Scope> {
        self.scopes
            .get(scope.index())
            .and_then(|entry| entry.parent)
    }

    pub fn gated(&self, scope: Scope) -> bool {
        let mut at = Some(scope);
        while let Some(current) = at {
            let Some(entry) = self.scopes.get(current.index()) else {
                return false;
            };
            if entry.gated {
                return true;
            }
            at = entry.parent;
        }
        false
    }

    pub fn fills(&self) -> &[Filled] {
        &self.fills
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

    pub fn operations(&self) -> &[OperationEntry] {
        &self.operations
    }

    pub fn explainers(&self) -> &[ExplainEntry] {
        &self.explainers
    }

    pub fn piece_entry(&self) -> Option<&PieceEntry> {
        self.piece.as_ref()
    }

    pub fn problems(&self) -> &[Error] {
        &self.problems
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

    pub fn places(&self) -> impl Iterator<Item = &PlaceDecl> + '_ {
        self.fills
            .iter()
            .map(|filled| &filled.place)
            .chain(self.joins.iter().map(|joined| &joined.place))
    }

    pub fn fills_slot(&self, slot: TypeId) -> bool {
        self.fills.iter().any(|filled| filled.slot == slot)
    }

    pub fn owns(&self, type_id: TypeId) -> bool {
        self.owned.iter().any(|owned| owned.type_id == type_id)
    }

    pub fn owns_path(&self, path: &str) -> bool {
        is_package(crate_segment(path), self.package)
    }

    pub fn when(&mut self, condition: GateFn) {
        if self.scopes[Scope::BRICK.index()].condition.is_some() {
            let path = format!("$.bricks.{}.when", self.package);
            self.problems
                .push(Error::new(ErrorCode::NameTwice, path).got(self.package));
            return;
        }
        let brick = &mut self.scopes[Scope::BRICK.index()];
        brick.condition = Some(condition);
        brick.gated = true;
    }

    pub fn enter(&mut self, condition: GateFn) -> Scope {
        let scope = Scope(self.scopes.len() as u32);
        self.scopes.push(ScopeEntry {
            parent: Some(self.current),
            condition: Some(condition),
            gated: true,
        });
        self.current = scope;
        scope
    }

    pub fn leave(&mut self) {
        self.current = self.parent(self.current).unwrap_or(Scope::BRICK);
    }

    pub fn off(&mut self, system: BoxedSystem) {
        self.offs.push(system);
    }

    pub fn systems<M: 'static>(
        &mut self,
        schedule: InternedScheduleLabel,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M> + Send + Sync + 'static,
    ) {
        self.systems.push(SystemsCall {
            scope: self.current,
            schedule,
            make: Box::new(move || systems.chain()),
        });
    }

    pub fn shader_prefix(&mut self, prefix: &'static str) {
        self.prefix = Some(prefix);
    }

    pub fn fill<S: Slot>(&mut self, wgsl: Option<Box<dyn Any + Send + Sync>>) {
        if self.fills_slot(TypeId::of::<S>()) {
            let path = format!("$.places.{}", S::NAME);
            self.problems
                .push(Error::new(ErrorCode::SlotFilledTwice, path).got(self.package));
            return;
        }
        self.fills.push(Filled::of::<S>(self.current, wgsl));
    }

    pub fn join<L: List>(&mut self, member: L::Member, wgsl: Option<Box<dyn Any + Send + Sync>>) {
        self.joins
            .push(JoinEntry::of::<L>(self.current, member, wgsl));
    }

    pub fn resource<R: Resource + Default>(&mut self) {
        self.own(Owned::of::<R>(init_of::<R>));
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

    pub fn piece<T: Resource<Mutability = Mutable> + Default + GpuStruct>(&mut self) {
        if self.piece.is_some() {
            let path = format!("$.bricks.{}.piece", self.package);
            self.problems
                .push(Error::new(ErrorCode::NameTwice, path).got(T::WGSL_NAME));
            return;
        }
        self.piece = Some(PieceEntry::of::<T>());
        self.own(Owned::of::<T>(init_of::<T>));
    }

    pub fn operation(&mut self, mut entry: OperationEntry) {
        entry.brick = self.package;
        if self.operations.iter().any(|known| known.name == entry.name) {
            let path = format!("$.operations.{}", entry.name);
            self.problems
                .push(Error::new(ErrorCode::NameTwice, path).got(self.package));
            return;
        }
        self.operations.push(entry);
    }

    pub fn explain(&mut self, mut entry: ExplainEntry) {
        entry.brick = self.package;
        self.explainers.push(entry);
    }

    pub fn on_remove(&mut self, removal: Removal) {
        self.removals.push(removal);
    }

    pub fn settle(&mut self, settle: Settle) {
        self.settles.push(settle);
    }

    pub fn named(&mut self, named: NamedField) {
        self.named.push(named);
    }

    pub(crate) fn take_problems(&mut self) -> Vec<Error> {
        std::mem::take(&mut self.problems)
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

pub(crate) fn crate_segment(path: &str) -> &str {
    path.split("::").next().unwrap_or(path)
}

pub(crate) fn is_package(segment: &str, package: &str) -> bool {
    segment.len() == package.len()
        && segment
            .bytes()
            .zip(package.bytes())
            .all(|(segment, package)| segment == package || (segment == b'_' && package == b'-'))
}
