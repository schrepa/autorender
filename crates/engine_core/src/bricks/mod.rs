use std::any::{Any, TypeId};
use std::marker::PhantomData;

use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::{Component, Immutable, Mutable};
use bevy_ecs::entity::Entity;
use bevy_ecs::query::{QueryFilter, ReadOnlyQueryData};
use bevy_ecs::resource::{IsResource, Resource};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::{
    Commands, In, IntoSystem, IsFunctionSystem, Query, ReadOnlySystemParam, Res, ScheduleSystem,
    SystemParam, SystemParamFunction, SystemState,
};
use bevy_ecs::world::World;
use engine_registry::{
    BrickRecord, ExplainAsk, ExplainEntry, ExplainRecord, GateFn, Gates, JobList, MembersOf,
    NamedField, OperationEntry, OperationScene, Registry, Removal, SceneEntity, SettingsEntry,
    Settle, SpawnedBy, StatsEntry,
};
use serde_json::Value;

use crate::{
    Authored, Error, ErrorCode, Explain, GpuStruct, JobKind, List, Module, Operation, PlaceDecl,
    SceneContext, SceneIds, Section, Slot, Smoke, Source,
};

pub trait Brick: 'static {
    const SOURCE: &'static Source;
    const JOBS: &'static [&'static JobKind] = &[];
    const SMOKES: &'static [Smoke] = &[];
    fn register(bricks: &mut Bricks<'_>);
}

pub struct Bricks<'r> {
    record: &'r mut BrickRecord,
}

impl engine_registry::RecordExt for Bricks<'_> {
    fn record(&mut self) -> &mut BrickRecord {
        &mut *self.record
    }
}

impl<'r> Bricks<'r> {
    #[doc(hidden)]
    pub fn recording(record: &'r mut BrickRecord) -> Self {
        Self { record }
    }
}

impl Bricks<'_> {
    pub fn when<M, F>(&mut self, gate: F) -> &mut Self
    where
        F: SystemParamFunction<M, In = (), Out = bool>,
        F::Param: Read,
        M: 'static,
    {
        self.record.when(gate_of(gate));
        self
    }

    pub fn within<M, F>(&mut self, gate: F, items: impl FnOnce(&mut Self)) -> &mut Self
    where
        F: SystemParamFunction<M, In = (), Out = bool>,
        F::Param: Read,
        M: 'static,
    {
        self.record.enter(gate_of(gate));
        items(self);
        self.record.leave();
        self
    }

    pub fn off<M>(
        &mut self,
        system: impl IntoSystem<(), (), M> + Send + Sync + 'static,
    ) -> &mut Self {
        self.record.off(Box::new(IntoSystem::into_system(system)));
        self
    }

    pub fn off_in<M>(
        &mut self,
        schedule: Schedule,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M> + Send + Sync + 'static,
    ) -> &mut Self
    where
        M: 'static,
    {
        self.record.off_in(schedule.label(), systems);
        self
    }

    pub fn fill<S: Slot>(&mut self, wgsl: Option<Module>) -> &mut Self {
        self.record.fill::<S>(wgsl.map(boxed_module));
        self
    }

    pub fn join<L: List>(
        &mut self,
        member: impl Into<L::Member>,
        wgsl: Option<Module>,
    ) -> &mut Self {
        self.record.join::<L>(member.into(), wgsl.map(boxed_module));
        self
    }

    pub fn systems<M>(
        &mut self,
        schedule: Schedule,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M> + Send + Sync + 'static,
    ) -> &mut Self
    where
        M: 'static,
    {
        self.record.systems(schedule.label(), systems);
        self
    }

    pub fn resource<R: Resource + Default>(&mut self) -> &mut Self {
        self.record.resource::<R>();
        self
    }

    #[track_caller]
    pub fn settings<S: Section + Resource<Mutability = Immutable>>(&mut self) -> &mut Self {
        self.record.settings(settings_entry::<S>());
        self
    }

    #[track_caller]
    pub fn stats<S: Section + Resource<Mutability = Mutable>>(&mut self) -> &mut Self {
        self.record
            .stats(StatsEntry::of::<S>(S::KEY, pending_of::<S>, S::schema));
        self
    }

    #[track_caller]
    pub fn settled_stats<S: Section + Resource<Mutability = Mutable>, M: 'static>(
        &mut self,
        settle: impl IntoSystem<(), (), M> + Send + Sync + 'static,
    ) -> &mut Self {
        self.record
            .stats(StatsEntry::of::<S>(S::KEY, pending_of::<S>, S::schema));
        self.record.settle(Settle::of(
            S::KEY,
            Box::new(IntoSystem::into_system(settle)),
        ));
        self
    }

    pub fn piece<T: Resource<Mutability = Mutable> + Default + GpuStruct>(&mut self) -> &mut Self {
        const {
            assert!(
                std::mem::size_of::<T>() % 16 == 0,
                "a piece is not a multiple of 16 bytes"
            );
        }
        self.record.piece::<T>();
        self
    }

    #[track_caller]
    pub fn operation<O: Operation>(&mut self) -> &mut Self {
        self.record.operation(
            OperationEntry::of::<O::Params>(O::NAME, O::CONTROL, run_operation::<O>)
                .schemas(O::schema, O::result_schema),
        );
        self
    }

    #[track_caller]
    pub fn observation<O: Operation>(&mut self) -> &mut Self
    where
        O::Params: Read,
    {
        self.record.operation(
            OperationEntry::of::<O::Params>(O::NAME, false, run_operation::<O>)
                .schemas(O::schema, O::result_schema),
        );
        self
    }

    #[track_caller]
    pub fn settled<O: Operation, M: 'static>(
        &mut self,
        settle: impl IntoSystem<(), (), M> + Send + Sync + 'static,
    ) -> &mut Self
    where
        O::Params: Read,
    {
        self.record.settle(Settle::of(
            O::NAME,
            Box::new(IntoSystem::into_system(settle)),
        ));
        self.record.operation(
            OperationEntry::of::<O::Params>(O::NAME, false, run_settled::<O>)
                .schemas(O::schema, O::result_schema),
        );
        self
    }

    #[track_caller]
    pub fn explain<E: Explain>(&mut self) -> &mut Self
    where
        E::Params: Read,
    {
        self.record
            .explain(ExplainEntry::of::<E::Params>(run_explain::<E>));
        self
    }

    pub fn on_remove<C: Component, M>(
        &mut self,
        system: impl IntoSystem<In<Entity>, (), M> + Send + Sync + 'static,
    ) -> &mut Self {
        self.record
            .on_remove(Removal::of::<C>(Box::new(IntoSystem::into_system(system))));
        self
    }

    pub fn named<S: Section, L: List>(
        &mut self,
        field: &'static str,
        kept: &'static [&'static str],
        name: fn(&L::Member) -> &'static str,
    ) -> &mut Self {
        self.record
            .named(NamedField::of(S::KEY, field, kept, move |world: &World| {
                world
                    .get_resource::<MembersOf<L>>()
                    .map_or_else(Vec::new, |members| members.all().iter().map(name).collect())
            }));
        self
    }

    pub fn named_by<S: Section, L: List>(
        &mut self,
        field: &'static str,
        kept: &'static [&'static str],
        name: fn(&L::Member) -> &'static str,
        order: fn(&L::Member) -> u32,
    ) -> &mut Self {
        self.record
            .named(NamedField::of(S::KEY, field, kept, move |world: &World| {
                world
                    .get_resource::<MembersOf<L>>()
                    .map_or_else(Vec::new, |members| {
                        let mut all: Vec<&L::Member> = members.all().iter().collect();
                        all.sort_by_key(|member| order(member));
                        all.into_iter().map(name).collect()
                    })
            }));
        self
    }
}

fn gate_of<M: 'static, F>(gate: F) -> GateFn
where
    F: SystemParamFunction<M, In = (), Out = bool>,
    F::Param: Read,
{
    Box::new(IntoSystem::<(), bool, (IsFunctionSystem, M)>::into_system(gate))
}

fn boxed_module(module: Module) -> Box<dyn Any + Send + Sync> {
    Box::new(module)
}

#[track_caller]
pub(crate) fn settings_entry<S: Section>() -> SettingsEntry {
    SettingsEntry::of::<S>(S::KEY, check_of::<S>, set_of::<S>, S::schema).scene_check(scene_of::<S>)
}

fn check_of<S: Section>(value: &Value, path: &str) -> Result<(), Error> {
    engine_registry::parse_section::<S>(value, path, S::check).map(drop)
}

fn set_of<S: Section>(world: &mut World, value: &Value, path: &str) -> Result<(), Error> {
    let section: S = engine_settings::parse_value_at(value.clone(), path)?;
    if let Some(registry) = world.get_resource::<Registry>() {
        registry.check_named(S::KEY, value, path)?;
    }
    section.check().map_err(|error| error.within(path))?;
    if let Some(scene) = world.get_resource::<SceneContext>() {
        scene
            .scope(|| section.check_scene(scene))
            .map_err(|error| error.within(path))?;
    }
    world.insert_resource(section);
    Ok(())
}

fn scene_of<S: Section>(value: &Value, path: &str, scene: &dyn Any) -> Result<(), Error> {
    let Some(scene) = scene.downcast_ref::<SceneContext>() else {
        return Ok(());
    };
    scene.scope(|| {
        let section: S = engine_settings::parse_value_at(value.clone(), path)?;
        section
            .check_scene(scene)
            .map_err(|error| error.within(path))
    })
}

fn pending_of<S: Section>(world: &World) -> u32 {
    world.get_resource::<S>().map_or(0, Section::pending)
}

pub(crate) fn run_operation<O: Operation>(
    world: &mut World,
    request: Value,
) -> Result<Value, Error> {
    let request: O::Request = engine_settings::parse_value(request)?;
    answer::<O>(world, request)
}

fn run_settled<O: Operation>(world: &mut World, request: Value) -> Result<Value, Error> {
    let request: O::Request = engine_settings::parse_value(request)?;
    settle_stats(world, O::NAME);
    answer::<O>(world, request)
}

pub fn settle_stats(world: &mut World, key: &str) {
    world.run_schedule(crate::schedules::TransformPropagation);
    engine_registry::run_settles(world, key);
}

fn run_explain<E: Explain>(world: &mut World, ask: &ExplainAsk<'_>) -> Option<ExplainRecord> {
    let mut state = SystemState::<E::Params>::new(world);
    let params = state.get_mut(world).ok()?;
    E::explain(params, ask)
}

fn answer<O: Operation>(world: &mut World, request: O::Request) -> Result<Value, Error> {
    let mut state = SystemState::<O::Params>::new(world);
    let params = state.get_mut(world).map_err(|refused| {
        Error::new(ErrorCode::MissingResource, format!("$.operations.{}", O::NAME))
            .got(refused.to_string())
    })?;
    let answer = O::run(params, request);
    state.apply(world);
    answer
}

pub trait Listing: Sized {
    #[must_use]
    fn with<B: Brick>(self) -> Self;
    #[must_use]
    fn without<B: Brick>(self) -> Self;
    #[must_use]
    fn places(self, places: &'static [PlaceDecl]) -> Self;
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Listed {
    source: &'static Source,
    jobs: &'static [&'static JobKind],
    smokes: &'static [Smoke],
    record: fn() -> BrickRecord,
}

impl Listed {
    fn of<B: Brick>() -> Self {
        Self {
            source: B::SOURCE,
            jobs: B::JOBS,
            smokes: B::SMOKES,
            record: record_of::<B>,
        }
    }

    pub fn source(&self) -> &'static Source {
        self.source
    }

    pub fn package(&self) -> &'static str {
        self.source.package()
    }

    pub fn jobs(&self) -> &'static [&'static JobKind] {
        self.jobs
    }

    pub fn smokes(&self) -> &'static [Smoke] {
        self.smokes
    }

    pub fn record(&self) -> BrickRecord {
        (self.record)()
    }
}

fn record_of<B: Brick>() -> BrickRecord {
    let source = B::SOURCE;
    let mut record = BrickRecord::new::<B>(source.package(), source.hash(), B::SMOKES);
    if let Some(first) = source.modules().first() {
        let path = first.path();
        record.shader_prefix(path.split("::").next().unwrap_or(path));
    }
    B::register(&mut Bricks::recording(&mut record));
    record
}

#[doc(hidden)]
#[derive(Clone, Default)]
pub struct BrickList {
    bricks: Vec<Listed>,
    left: Vec<Listed>,
    places: Vec<&'static [PlaceDecl]>,
    left_out: Vec<&'static str>,
    core: Vec<fn(&mut Registry)>,
}

impl BrickList {
    pub fn listed(&self) -> &[Listed] {
        &self.bricks
    }

    pub fn declared(&self) -> impl Iterator<Item = &'static PlaceDecl> + '_ {
        self.places.iter().flat_map(|places| places.iter())
    }

    pub fn place_sets(&self) -> &[&'static [PlaceDecl]] {
        &self.places
    }

    pub fn left_out(&self) -> &[&'static str] {
        &self.left_out
    }

    pub fn core_places(&self) -> &[fn(&mut Registry)] {
        &self.core
    }

    pub fn contains(&self, package: &str) -> bool {
        self.bricks
            .iter()
            .any(|listed| listed.package() == package)
    }

    pub fn jobs(&self) -> Vec<&'static JobKind> {
        self.bricks
            .iter()
            .flat_map(|listed| listed.jobs().iter().copied())
            .collect()
    }

    pub fn job_list(&self) -> Result<JobList, Error> {
        let mut records: Vec<BrickRecord> = self.bricks.iter().map(Listed::record).collect();
        records.sort_by(|a, b| a.package.cmp(b.package));
        let members = records
            .iter_mut()
            .flat_map(BrickRecord::take_members)
            .collect();
        let mut kinds = self.jobs();
        kinds.extend(
            self.left
                .iter()
                .flat_map(|left| left.jobs().iter().copied()),
        );
        crate::job_list_with(&kinds, members)
    }

    #[must_use]
    pub fn with_core(mut self, register: fn(&mut Registry)) -> Self {
        self.core.push(register);
        self
    }

    #[must_use]
    pub fn without_package(mut self, package: &str) -> Self {
        let (out, kept) = std::mem::take(&mut self.bricks)
            .into_iter()
            .partition::<Vec<_>, _>(|listed| listed.package() == package);
        self.bricks = kept;
        self.leave(out)
    }

    #[must_use]
    fn leave(mut self, out: Vec<Listed>) -> Self {
        for listed in out {
            let package = listed.package();
            if !self.left_out.contains(&package) {
                self.left_out.push(package);
            }
            if !self
                .left
                .iter()
                .any(|left| std::ptr::eq(left.source, listed.source))
            {
                self.left.push(listed);
            }
        }
        self
    }
}

impl Listing for BrickList {
    fn with<B: Brick>(mut self) -> Self {
        let package = B::SOURCE.package();
        self.left_out.retain(|&left| left != package);
        self.left
            .retain(|left| !std::ptr::eq(left.source, B::SOURCE));
        self.bricks.push(Listed::of::<B>());
        self
    }

    fn without<B: Brick>(mut self) -> Self {
        let (out, kept) = std::mem::take(&mut self.bricks)
            .into_iter()
            .partition::<Vec<_>, _>(|listed| std::ptr::eq(listed.source, B::SOURCE));
        self.bricks = kept;
        self.leave(out)
    }

    fn places(mut self, places: &'static [PlaceDecl]) -> Self {
        self.places.push(places);
        self
    }
}

mod sealed {
    pub trait Sealed {}
}

pub trait Read: ReadOnlySystemParam + sealed::Sealed {}

impl<T: Resource> sealed::Sealed for Res<'_, T> {}
impl<T: Resource> Read for Res<'_, T> {}
impl<T: Resource> sealed::Sealed for Option<Res<'_, T>> {}
impl<T: Resource> Read for Option<Res<'_, T>> {}
impl<D: ReadOnlyQueryData + 'static, F: QueryFilter + 'static> sealed::Sealed
    for Query<'_, '_, D, F>
{
}
impl<D: ReadOnlyQueryData + 'static, F: QueryFilter + 'static> Read for Query<'_, '_, D, F> {}
impl<L: List> sealed::Sealed for Joined<'_, L> {}
impl<L: List> Read for Joined<'_, L> {}
impl sealed::Sealed for SceneIds<'_, '_> {}
impl Read for SceneIds<'_, '_> {}
impl<S: Slot> sealed::Sealed for Filled<'_, S> {}
impl<S: Slot> Read for Filled<'_, S> {}

macro_rules! read_tuples {
    ($($param:ident),*) => {
        impl<$($param: Read),*> sealed::Sealed for ($($param,)*) {}
        impl<$($param: Read),*> Read for ($($param,)*) {}
    };
}

read_tuples!();
read_tuples!(P0);
read_tuples!(P0, P1);
read_tuples!(P0, P1, P2);
read_tuples!(P0, P1, P2, P3);
read_tuples!(P0, P1, P2, P3, P4);
read_tuples!(P0, P1, P2, P3, P4, P5);
read_tuples!(P0, P1, P2, P3, P4, P5, P6);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14);
read_tuples!(P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15);

#[derive(SystemParam)]
pub struct Joined<'w, L: List> {
    members: Option<Res<'w, engine_registry::MembersOf<L>>>,
}

impl<L: List> Joined<'_, L> {
    pub fn iter(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.members.iter().flat_map(|members| members.live())
    }

    pub fn owned(&self) -> impl Iterator<Item = (std::any::TypeId, u32, &L::Member)> + '_ {
        self.members.iter().flat_map(|members| members.owned())
    }

    pub fn all(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.members.iter().flat_map(|members| members.all().iter())
    }

    pub fn every(&self) -> impl Iterator<Item = (std::any::TypeId, u32, &L::Member)> + '_ {
        self.members.iter().flat_map(|members| members.every())
    }
}

#[derive(SystemParam)]
pub struct Filled<'w, S: Slot> {
    registry: Option<Res<'w, Registry>>,
    gates: Option<Res<'w, Gates>>,
    slot: PhantomData<fn() -> S>,
}

impl<S: Slot> Filled<'_, S> {
    pub fn on(&self) -> bool {
        self.registry
            .as_ref()
            .zip(self.gates.as_ref())
            .is_some_and(|(registry, gates)| registry.slot_on(gates, TypeId::of::<S>()))
    }

    pub fn listed(&self) -> bool {
        self.registry
            .as_ref()
            .is_some_and(|registry| registry.filler(TypeId::of::<S>()).is_some())
    }
}

#[derive(SystemParam)]
pub struct Spawner<'w, 's, B: Brick> {
    commands: Commands<'w, 's>,
    brick: PhantomData<fn() -> B>,
}

impl<B: Brick> Spawner<'_, '_, B> {
    pub fn spawn(&mut self, bundle: impl Bundle) -> Entity {
        let entity = self.commands.spawn_empty().id();
        self.commands
            .queue(move |world: &mut World| spawned::<B, _>(world, entity, bundle));
        entity
    }

    pub fn spawn_then<R: Resource<Mutability = Mutable>>(
        &mut self,
        bundle: impl Bundle,
        then: impl FnOnce(&mut R, Entity) + Send + 'static,
    ) {
        self.commands
            .queue(move |world: &mut World| spawned_then::<B, R, _>(world, bundle, then));
    }

    pub fn insert(&mut self, entity: Entity, bundle: impl Bundle) {
        self.commands
            .queue(move |world: &mut World| inserted::<B, _>(world, entity, bundle));
    }

    pub fn mark<C: Component, T: Component + Default>(&mut self, entity: Entity) {
        self.commands
            .queue(move |world: &mut World| marked::<B, C, T>(world, entity));
    }

    pub fn remove<T: Bundle>(&mut self, entity: Entity) {
        self.commands
            .queue(move |world: &mut World| removed::<B, T>(world, entity));
    }

    pub fn despawn(&mut self, entity: Entity) {
        self.commands
            .queue(move |world: &mut World| despawned::<B>(world, entity));
    }

    pub fn author(&mut self, entity: Entity, authored: Authored) {
        self.commands
            .queue(move |world: &mut World| authored_by::<B>(world, entity, authored));
    }

    pub fn insert_scene(&mut self, entity: Entity, key: &str, value: Value) {
        let key = key.to_owned();
        self.commands
            .queue(move |world: &mut World| scene_inserted::<B>(world, entity, &key, &value));
    }

    pub fn set_parent(&mut self, child: Entity, parent: Option<Entity>) {
        self.set_parents(vec![(child, parent)]);
    }

    pub fn set_parents(&mut self, changes: Vec<(Entity, Option<Entity>)>) {
        self.commands
            .queue(move |world: &mut World| parented::<B>(world, &changes));
    }

    pub fn set_parents_then<R: Resource<Mutability = Mutable>>(
        &mut self,
        changes: Vec<(Entity, Option<Entity>)>,
        then: impl FnOnce(&mut R, Result<(), Error>) + Send + 'static,
    ) {
        self.commands.queue(move |world: &mut World| {
            parented_then::<B, R>(world, &changes, then);
        });
    }

    pub fn set_parents_from<R: Resource<Mutability = Mutable>>(
        &mut self,
        changes: impl FnOnce(&R) -> Vec<(Entity, Option<Entity>)> + Send + 'static,
        then: impl FnOnce(&mut R, Result<(), Error>) + Send + 'static,
    ) {
        self.commands.queue(move |world: &mut World| {
            if !holds::<B, R>(world) {
                return;
            }
            let Some(held) = world.get_resource::<R>() else {
                return;
            };
            let changes = changes(held);
            parented_then::<B, R>(world, &changes, then);
        });
    }
}

fn spawned<B: Brick, T: Bundle>(world: &mut World, entity: Entity, bundle: T) {
    if carries_resource::<T>(world) {
        world.despawn(entity);
        foreign::<B>(world, entity);
        return;
    }
    if let Ok(mut spawned) = world.get_entity_mut(entity) {
        spawned.insert((bundle, SpawnedBy::<B>::new()));
    }
}

fn spawned_then<B: Brick, R: Resource<Mutability = Mutable>, T: Bundle>(
    world: &mut World,
    bundle: T,
    then: impl FnOnce(&mut R, Entity),
) {
    if carries_resource::<T>(world) {
        let package = B::SOURCE.package();
        let error = Error::new(
            ErrorCode::ForeignEntity,
            format!("$.bricks.{package}.spawner"),
        )
        .got(std::any::type_name::<T>());
        report(world, error);
        return;
    }
    if !holds::<B, R>(world) {
        return;
    }
    let entity = world.spawn((bundle, SpawnedBy::<B>::new())).id();
    if let Some(mut held) = world.get_resource_mut::<R>() {
        then(&mut held, entity);
    }
}

fn marked<B: Brick, C: Component, T: Component + Default>(world: &mut World, entity: Entity) {
    let component = world.register_component::<C>();
    let kinds: Vec<fn(&mut World) -> bevy_ecs::component::ComponentId> = world
        .get_resource::<MembersOf<crate::SceneComponents>>()
        .map(|members| {
            members
                .every()
                .filter(|&(brick, _, _)| brick == TypeId::of::<B>())
                .map(|(_, _, kind)| kind.registrar())
                .collect()
        })
        .unwrap_or_default();
    let registered = kinds.into_iter().any(|register| register(world) == component);
    if !registered || world.get::<C>(entity).is_none() {
        foreign::<B>(world, entity);
        return;
    }
    if let Ok(mut held) = world.get_entity_mut(entity) {
        held.insert(T::default());
    }
}

fn inserted<B: Brick, T: Bundle>(world: &mut World, entity: Entity, bundle: T) {
    if !owned::<B>(world, entity) || carries_resource::<T>(world) {
        foreign::<B>(world, entity);
        return;
    }
    if let Ok(mut owned) = world.get_entity_mut(entity) {
        owned.insert(bundle);
    }
}

fn removed<B: Brick, T: Bundle>(world: &mut World, entity: Entity) {
    if !owned::<B>(world, entity) || carries_resource::<T>(world) {
        foreign::<B>(world, entity);
        return;
    }
    if let Ok(mut owned) = world.get_entity_mut(entity) {
        owned.remove::<T>();
    }
}

fn despawned<B: Brick>(world: &mut World, entity: Entity) {
    if !owned::<B>(world, entity) {
        foreign::<B>(world, entity);
        return;
    }
    world.despawn(entity);
}

fn authored_by<B: Brick>(world: &mut World, entity: Entity, authored: Authored) {
    if !owned::<B>(world, entity) {
        foreign::<B>(world, entity);
        return;
    }
    let scene_id = if let Some(mut scene) = world.get_resource_mut::<OperationScene>() {
        scene.streamed = true;
        scene.id.clone()
    } else {
        world
            .get_resource::<SceneContext>()
            .map_or_else(String::new, |context| context.id.clone())
    };
    if let Ok(mut owned) = world.get_entity_mut(entity) {
        owned.insert(SceneEntity {
            scene_id,
            authored_id: authored.id,
            name: authored.name,
            tags: authored.tags,
            authored_parent: authored.parent,
            prefab_id: authored.prefab_id,
            prefab_instance: authored.prefab_instance,
            prefab_local_id: authored.prefab_local_id,
        });
    }
}

fn scene_inserted<B: Brick>(world: &mut World, entity: Entity, key: &str, value: &Value) {
    if !owned::<B>(world, entity) {
        foreign::<B>(world, entity);
        return;
    }
    let path = format!("$.bricks.{}.spawner.{key}", B::SOURCE.package());
    let read = match crate::scene_component(world, key) {
        Some(kind) => {
            let context = world.get_resource::<SceneContext>().cloned();
            let mut insert = || kind.read(world, entity, value);
            match context {
                Some(context) => context.scope(insert),
                None => insert(),
            }
            .map_err(|error| error.within(&path))
        }
        None => Err(Error::new(ErrorCode::SectionUnlisted, path).got(key)),
    };
    if let Err(error) = read {
        report(world, error);
    }
}

fn parented<B: Brick>(world: &mut World, changes: &[(Entity, Option<Entity>)]) {
    let stranger = changes
        .iter()
        .find(|&&(child, _)| !owned::<B>(world, child));
    if let Some(&(child, _)) = stranger {
        foreign::<B>(world, child);
        return;
    }
    if let Err(error) = crate::set_parents(world, changes) {
        let path = format!("$.bricks.{}.spawner.parent", B::SOURCE.package());
        report(world, error.within(&path));
    }
}

fn parented_then<B: Brick, R: Resource<Mutability = Mutable>>(
    world: &mut World,
    changes: &[(Entity, Option<Entity>)],
    then: impl FnOnce(&mut R, Result<(), Error>),
) {
    let stranger = changes
        .iter()
        .find(|&&(child, _)| !owned::<B>(world, child));
    if let Some(&(child, _)) = stranger {
        foreign::<B>(world, child);
        return;
    }
    if !holds::<B, R>(world) {
        return;
    }
    let parented = crate::set_parents(world, changes);
    if let Some(mut held) = world.get_resource_mut::<R>() {
        then(&mut held, parented);
    }
}

fn holds<B: Brick, R: Resource>(world: &mut World) -> bool {
    let package = B::SOURCE.package();
    let holds = world
        .get_resource::<Registry>()
        .and_then(|registry| registry.brick(package))
        .is_some_and(|record| record.owns(TypeId::of::<R>()));
    if !holds {
        let name = std::any::type_name::<R>();
        let path = format!("$.bricks.{package}.spawner.{name}");
        report(world, Error::new(ErrorCode::ForeignAccess, path).got(name));
    }
    holds
}

fn owned<B: Brick>(world: &World, entity: Entity) -> bool {
    world.get::<SpawnedBy<B>>(entity).is_some()
}

fn carries_resource<T: Bundle>(world: &mut World) -> bool {
    let marker = world.register_component::<IsResource>();
    world
        .register_bundle::<T>()
        .contributed_components()
        .contains(&marker)
}

fn foreign<B: Brick>(world: &mut World, entity: Entity) {
    let package = B::SOURCE.package();
    let error = Error::new(
        ErrorCode::ForeignEntity,
        format!("$.bricks.{package}.spawner"),
    )
    .got(entity.to_string());
    report(world, error);
}

fn report(world: &mut World, error: Error) {
    if let Some(mut registry) = world.get_resource_mut::<Registry>() {
        registry.report(error);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Schedule {
    FrameStart,
    FixedPreSimulation,
    FixedMove,
    FixedSimulation,
    FixedPostSimulation,
    Presentation,
    WorldSync,
    ViewSync,
    BetweenFrames,
    Waiting,
    Stream,
    Rendered,
}
