use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::{Component, ComponentId, Immutable, Mutable};
use bevy_ecs::entity::Entity;
use bevy_ecs::query::{QueryFilter, ReadOnlyQueryData};
use bevy_ecs::resource::{IsResource, Resource};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::{
    Commands, In, IntoSystem, IsFunctionSystem, Query, ReadOnlySystemParam, Res, ScheduleSystem,
    SystemParam, SystemParamFunction,
};
use bevy_ecs::world::World;
use engine_place::{Call, Event, EventDecl, Raises};
use engine_registry::{
    BrickRecord, BrickStates, EventEntry, GateFn, JobList, MembersOf, NamedField, OperationEntry,
    OperationScene, PlainSystems, Registry, RegistryBuilder, Removal, SceneEntity, SectionRule,
    SettingsEntry, SpawnedBy, StatsEntry,
};
use serde_json::Value;

use crate::moves::NodeKind;
use crate::{
    Board, Error, ErrorCode, GpuStruct, JobKind, KindDecl, List, Module, Mover, MoverId, Operation,
    Pace, PlaceDecl, RecordKey, SceneContext, SceneIds, Section, Slot, Smoke, Source, SubScene,
};

pub trait Brick: 'static {
    const SOURCE: &'static Source;
    const JOBS: &'static [&'static JobKind] = &[];
    const SMOKES: &'static [Smoke] = &[];
    const CORPORA: &'static [Corpus] = &[];
    fn register(bricks: &mut Bricks<'_>);
}

#[derive(Clone, Copy)]
pub struct Corpus {
    pub name: &'static str,
    pub replays: &'static [&'static JobKind],
    pub run: fn(&mut dyn PureSession) -> Result<(), String>,
}

pub trait PureSession {
    fn root(&self) -> &std::path::Path;
    fn list(&self) -> &JobList;
    fn record(&mut self, name: &str, value: Value);
    fn hash(&mut self, name: &str, bytes: &[u8]);
    fn digest(&self, bytes: &[u8]) -> String;
    fn check(&mut self, holds: bool, what: &dyn Fn() -> String);
    fn share(&mut self, bytes: &[u8]) -> RecordKey;
    fn run(&mut self, kind: &'static JobKind, input: &[u8]) -> Result<Vec<u8>, String>;
    fn sliced(
        &mut self,
        kind: &'static JobKind,
        input: &[u8],
        slice: u64,
    ) -> Result<Vec<u8>, String>;
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

    #[track_caller]
    pub fn fill<S: Slot>(&mut self, wgsl: Option<Module>) -> &mut Self {
        self.record.fill::<S>(wgsl.map(boxed_module));
        self
    }

    pub fn requires<S: Slot>(&mut self) -> &mut Self {
        self.record.require::<S>();
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
        systems: impl IntoScheduleConfigs<ScheduleSystem, M> + PlainSystems<M>,
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

    pub fn section_rule(&mut self, rule: SectionRule) -> &mut Self {
        self.record.section_rule(rule);
        self
    }

    pub fn errors(&mut self, codes: &'static [ErrorCode]) -> &mut Self {
        self.record.add_error_codes(codes);
        self
    }

    #[track_caller]
    pub fn stats<S: Section + Resource<Mutability = Mutable>>(&mut self) -> &mut Self {
        self.record.stats(StatsEntry::of::<S>(S::KEY, S::schema));
        self
    }

    pub fn piece<T: Resource<Mutability = Mutable> + Default + GpuStruct>(&mut self) -> &mut Self {
        const {
            assert!(
                std::mem::size_of::<T>().is_multiple_of(16),
                "a piece is not a multiple of 16 bytes"
            );
        }
        self.record.piece::<T>();
        self
    }

    pub fn draw_kind(&mut self, kind: KindDecl) -> &mut Self {
        self.record.render.push(Box::new(kind));
        self
    }

    #[track_caller]
    pub fn operation<O: Operation>(&mut self) -> &mut Self {
        self.record.operation(OperationEntry::of(Call::of::<O>()));
        self
    }

    #[track_caller]
    pub fn observation<O: Operation>(&mut self) -> &mut Self
    where
        O::Params: Read,
    {
        self.record
            .operation(OperationEntry::of(Call::of::<O>().observed()));
        self
    }

    #[track_caller]
    pub fn event<E: Event, R: Raises<E>>(&mut self) -> &mut Self {
        self.record.event(EventEntry::of(EventDecl::of::<E, R>()));
        self
    }

    pub fn moves<C: Component>(&mut self, pace: Pace) -> &mut Self {
        let by = if self.record.game() {
            MoverId::Game
        } else {
            MoverId::Brick(self.record.package)
        };
        self.record.entries.push(Box::new(NodeKind::Claimed {
            claim: crate::moves::claim::<C>,
            mover: Mover { by, pace },
        }));
        self
    }

    pub fn moving<C: Component>(&mut self) -> &mut Self {
        self.record.entries.push(Box::new(NodeKind::Moving {
            track: crate::moves::moving::<C>,
        }));
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
}

fn gate_of<M: 'static, F>(gate: F) -> GateFn
where
    F: SystemParamFunction<M, In = (), Out = bool>,
    F::Param: Read,
{
    Box::new(IntoSystem::<(), bool, (IsFunctionSystem, M)>::into_system(
        gate,
    ))
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

pub trait Listing: Sized {
    #[must_use]
    fn with<B: Brick>(self) -> Self;
    #[must_use]
    fn without<B: Brick>(self) -> Self;
    #[must_use]
    fn knows<B: Brick>(self) -> Self;
    #[must_use]
    fn places(self, places: &'static [PlaceDecl]) -> Self;
    #[must_use]
    fn settings<S: Section>(self, value: S) -> Self;
}

pub type ListDefault = Arc<dyn Fn(&mut World) + Send + Sync>;

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
    let corpora = B::CORPORA
        .iter()
        .map(|corpus| corpus as &'static (dyn Any + Send + Sync))
        .collect();
    let mut record = BrickRecord::new::<B>(source.package(), source.hash(), B::SMOKES, corpora);
    record.set_game(source.game());
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
    known: Vec<Listed>,
    places: Vec<&'static [PlaceDecl]>,
    core: Vec<fn(&mut RegistryBuilder)>,
    defaults: Vec<ListDefault>,
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

    pub fn left(&self) -> &[Listed] {
        &self.left
    }

    pub fn known(&self) -> &[Listed] {
        &self.known
    }

    pub fn core_places(&self) -> &[fn(&mut RegistryBuilder)] {
        &self.core
    }

    pub fn defaults(&self) -> &[ListDefault] {
        &self.defaults
    }

    pub fn contains(&self, package: &str) -> bool {
        self.bricks.iter().any(|listed| listed.package() == package)
    }

    pub fn jobs(&self) -> Vec<&'static JobKind> {
        self.bricks
            .iter()
            .flat_map(|listed| listed.jobs().iter().copied())
            .collect()
    }

    pub fn job_list(&self) -> Result<JobList, Error> {
        let mut records: Vec<BrickRecord> = self.bricks.iter().map(Listed::record).collect();
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
    pub fn with_core(mut self, register: fn(&mut RegistryBuilder)) -> Self {
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
        self.left
            .retain(|left| !std::ptr::eq(left.source, B::SOURCE));
        self.known
            .retain(|known| !std::ptr::eq(known.source, B::SOURCE));
        self.bricks.push(Listed::of::<B>());
        self
    }

    fn without<B: Brick>(mut self) -> Self {
        self.known
            .retain(|known| !std::ptr::eq(known.source, B::SOURCE));
        let (mut out, kept) = std::mem::take(&mut self.bricks)
            .into_iter()
            .partition::<Vec<_>, _>(|listed| std::ptr::eq(listed.source, B::SOURCE));
        self.bricks = kept;
        if out.is_empty() {
            out.push(Listed::of::<B>());
        }
        self.leave(out)
    }

    fn knows<B: Brick>(mut self) -> Self {
        let held = |listed: &Listed| std::ptr::eq(listed.source, B::SOURCE);
        if !self.bricks.iter().any(held)
            && !self.left.iter().any(held)
            && !self.known.iter().any(held)
        {
            self.known.push(Listed::of::<B>());
        }
        self
    }

    fn places(mut self, places: &'static [PlaceDecl]) -> Self {
        self.places.push(places);
        self
    }

    fn settings<S: Section>(mut self, value: S) -> Self {
        self.defaults.push(Arc::new(move |world: &mut World| {
            world.insert_resource(value.clone());
        }));
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
read_tuples!(
    P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14
);
read_tuples!(
    P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15
);

#[derive(SystemParam)]
pub struct Joined<'w, L: List> {
    members: Option<Res<'w, engine_registry::MembersOf<L>>>,
    states: Option<Res<'w, BrickStates>>,
}

impl<L: List> Joined<'_, L> {
    pub fn iter(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.members
            .iter()
            .zip(self.states.iter())
            .flat_map(|(members, states)| members.live(states))
    }

    pub fn owned(&self) -> impl Iterator<Item = (std::any::TypeId, u32, &L::Member)> + '_ {
        self.members
            .iter()
            .zip(self.states.iter())
            .flat_map(|(members, states)| members.owned(states))
    }

    pub fn all(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.members.iter().flat_map(|members| members.all().iter())
    }

    pub fn every(&self) -> impl Iterator<Item = (std::any::TypeId, u32, &L::Member)> + '_ {
        self.members.iter().flat_map(|members| members.every())
    }
}

pub fn members_in<L: List>(world: &World) -> &[L::Member] {
    match world.get_resource::<MembersOf<L>>() {
        Some(members) => members.all(),
        None => &[],
    }
}

#[derive(SystemParam)]
pub struct Filled<'w, S: Slot> {
    registry: Option<Res<'w, Registry>>,
    states: Option<Res<'w, BrickStates>>,
    slot: PhantomData<fn() -> S>,
}

impl<S: Slot> Filled<'_, S> {
    pub fn on(&self) -> bool {
        self.registry
            .as_ref()
            .zip(self.states.as_ref())
            .is_some_and(|(registry, states)| registry.slot_on(states, TypeId::of::<S>()))
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

    pub fn insert(&mut self, entity: Entity, bundle: impl Bundle) {
        self.commands
            .queue(move |world: &mut World| inserted::<B, _>(world, entity, bundle));
    }

    pub fn remove<T: Bundle>(&mut self, entity: Entity) {
        self.commands
            .queue(move |world: &mut World| removed::<B, T>(world, entity));
    }

    pub fn despawn(&mut self, entity: Entity) {
        self.commands
            .queue(move |world: &mut World| despawned::<B>(world, entity));
    }

    pub fn install(&mut self, scene: SubScene) -> Vec<Entity> {
        let entities: Vec<Entity> = scene
            .entities
            .iter()
            .map(|_| self.commands.spawn_empty().id())
            .collect();
        let reserved = entities.clone();
        self.commands
            .queue(move |world: &mut World| installed::<B>(world, &reserved, &scene));
        entities
    }

    pub fn set_parent(&mut self, child: Entity, parent: Option<Entity>) {
        self.set_parents(vec![(child, parent)]);
    }

    pub fn fault(&mut self, error: Error) {
        self.commands
            .queue(move |world: &mut World| engine_registry::fault_of::<B>(world, error));
    }

    pub fn set_parents(&mut self, changes: Vec<(Entity, Option<Entity>)>) {
        self.commands
            .queue(move |world: &mut World| parented::<B>(world, &changes));
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

fn scene_kinds<B: Brick>(world: &mut World) -> Vec<ComponentId> {
    let kinds: Vec<fn(&mut World) -> ComponentId> = world
        .get_resource::<MembersOf<crate::SceneComponents>>()
        .map(|members| {
            members
                .every()
                .filter(|&(brick, _, _)| brick == TypeId::of::<B>())
                .map(|(_, _, kind)| kind.registrar())
                .collect()
        })
        .unwrap_or_default();
    kinds.into_iter().map(|register| register(world)).collect()
}

fn private<B: Brick, T: Bundle>(world: &mut World) -> bool {
    let ids = world
        .register_bundle::<T>()
        .contributed_components()
        .to_vec();
    let Some(registry) = world.get_resource::<Registry>() else {
        return false;
    };
    let package = B::SOURCE.package();
    !ids.is_empty()
        && ids.iter().all(|&id| {
            world.components().get_info(id).is_some_and(|info| {
                info.type_id().is_some_and(|type_id| {
                    registry.private_type(package, type_id, &info.name().to_string())
                })
            })
        })
}

fn touches<B: Brick, T: Bundle>(world: &mut World, entity: Entity) -> bool {
    !carries_resource::<T>(world) && (owned::<B>(world, entity) || private::<B, T>(world))
}

fn inserted<B: Brick, T: Bundle>(world: &mut World, entity: Entity, bundle: T) {
    if !touches::<B, T>(world, entity) {
        foreign::<B>(world, entity);
        return;
    }
    if let Ok(mut held) = world.get_entity_mut(entity) {
        held.insert(bundle);
    }
}

fn removed<B: Brick, T: Bundle>(world: &mut World, entity: Entity) {
    if !touches::<B, T>(world, entity) {
        foreign::<B>(world, entity);
        return;
    }
    if let Ok(mut held) = world.get_entity_mut(entity) {
        held.remove::<T>();
    }
}

fn despawned<B: Brick>(world: &mut World, entity: Entity) {
    if !owned::<B>(world, entity) {
        foreign::<B>(world, entity);
        return;
    }
    world.despawn(entity);
}

fn installed<B: Brick>(world: &mut World, entities: &[Entity], scene: &SubScene) {
    let context = world.get_resource::<SceneContext>().cloned();
    let landed = Board::owned_by(B::SOURCE.package(), || match &context {
        Some(context) => context.scope(|| install::<B>(world, entities, scene)),
        None => install::<B>(world, entities, scene),
    });
    if let Err(error) = landed {
        for &entity in entities {
            let _ = world.try_despawn(entity);
        }
        let path = format!("$.bricks.{}.spawner.install", B::SOURCE.package());
        report(world, error.within(&path));
    }
}

fn install<B: Brick>(
    world: &mut World,
    entities: &[Entity],
    scene: &SubScene,
) -> Result<(), Error> {
    let scene_id = world
        .get_resource::<OperationScene>()
        .map(|scene| scene.id.clone())
        .or_else(|| {
            world
                .get_resource::<SceneContext>()
                .map(|context| context.id.clone())
        })
        .unwrap_or_default();
    let mut parents = Vec::new();
    for (index, (&entity, item)) in entities.iter().zip(&scene.entities).enumerate() {
        let at = format!("$.entities[{index}]");
        let authored = item.authored.clone();
        world
            .get_entity_mut(entity)
            .map_err(|_| Error::new(ErrorCode::UnknownEntity, &at).got(entity.to_string()))?
            .insert((
                item.transform,
                SpawnedBy::<B>::new(),
                SceneEntity {
                    scene_id: scene_id.clone(),
                    authored_id: authored.id,
                    name: authored.name,
                    tags: authored.tags,
                    authored_parent: authored.parent,
                    prefab_id: authored.prefab_id,
                    prefab_instance: authored.prefab_instance,
                    prefab_local_id: authored.prefab_local_id,
                },
            ));
        if let Some(parent) = item.parent {
            let parent = entities.get(parent).copied().ok_or_else(|| {
                Error::new(ErrorCode::UnknownParent, format!("{at}.parent")).got(parent)
            })?;
            parents.push((entity, Some(parent)));
        }
    }
    for (index, (&entity, item)) in entities.iter().zip(&scene.entities).enumerate() {
        for (key, value) in &item.components {
            let at = format!("$.entities[{index}].{key}");
            let kind = crate::scene_component(world, key)
                .ok_or_else(|| Error::new(ErrorCode::SectionUnlisted, &at).got(key))?;
            if let (crate::Placement::Roots(code), Some(_)) = (kind.placement(), item.parent) {
                return Err(Error::new(code, at).expected(crate::Expected::Absent));
            }
            kind.validate(value, &at)?;
            kind.read(world, entity, value)
                .map_err(|error| error.within(&at))?;
        }
    }
    crate::set_parents(world, &parents).map_err(|error| error.within("$.entities"))
}

fn parented<B: Brick>(world: &mut World, changes: &[(Entity, Option<Entity>)]) {
    let kinds = scene_kinds::<B>(world);
    let stranger = changes.iter().find(|&&(child, _)| {
        !owned::<B>(world, child)
            && !world
                .get_entity(child)
                .is_ok_and(|held| kinds.iter().any(|&kind| held.contains_id(kind)))
    });
    if let Some(&(child, _)) = stranger {
        foreign::<B>(world, child);
        return;
    }
    if let Err(error) = crate::set_parents(world, changes) {
        let path = format!("$.bricks.{}.spawner.parent", B::SOURCE.package());
        report(world, error.within(&path));
    }
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
}
