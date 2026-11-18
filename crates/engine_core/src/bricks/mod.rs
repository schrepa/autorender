use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::{Component, Immutable, Mutable};
use bevy_ecs::entity::Entity;
use bevy_ecs::query::{QueryFilter, ReadOnlyQueryData};
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::{Commands, Query, ReadOnlySystemParam, Res, ScheduleSystem, SystemParam};
use bevy_ecs::world::World;
use engine_profile::Carried;
use engine_registry::{
    BrickRecord, BrickStates, JobList, MembersOf, Registry, SettingsEntry, SpawnedBy,
    StatsEntry,
};
use serde_json::Value;

use crate::moves::NodeKind;
use crate::{
    Error, JobKind, KindDecl, List, Mover, MoverId, Pace, Parent, RecordKey, SceneContext, Section,
    Source, Switch,
};

pub trait Brick: 'static {
    const SOURCE: &'static Source;
    const JOBS: &'static [&'static JobKind] = &[];
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
    #[track_caller]
    pub fn switch<S: Switch + Resource<Mutability = Immutable>>(&mut self) -> &mut Self {
        self.record.settings(settings_entry::<S>());
        self.record.switch(S::KEY, switch_on::<S>);
        self
    }

    pub fn join<L: List>(&mut self, member: impl Into<L::Member>) -> &mut Self {
        self.record.join::<L>(member.into());
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
        self.record.stats(StatsEntry::of::<S>(S::KEY, S::schema));
        self
    }

    pub fn draw_kind(&mut self, kind: KindDecl) -> &mut Self {
        self.record.render.push(Arc::new(kind));
        self
    }

    #[cfg(any(feature = "dev", feature = "profiling"))]
    pub fn name(&mut self, entry: engine_registry::NameEntry) -> &mut Self {
        self.record.entries.push(Box::new(entry));
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
}

fn switch_on<S: Switch + Resource>(world: &World) -> bool {
    world.get_resource::<S>().is_some_and(Switch::on)
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
    fn settings<S: Section>(self, value: S) -> Self;
}

pub type ListDefault = Arc<dyn Fn(&mut World) + Send + Sync>;

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Listed {
    source: &'static Source,
    jobs: &'static [&'static JobKind],
    record: fn() -> BrickRecord,
}

impl Listed {
    fn of<B: Brick>() -> Self {
        Self {
            source: B::SOURCE,
            jobs: B::JOBS,
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
    let mut record = BrickRecord::new::<B>(source.package(), source.hash(), corpora);
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
    defaults: Vec<ListDefault>,
}

impl BrickList {
    pub fn listed(&self) -> &[Listed] {
        &self.bricks
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
        let kinds = self.jobs();
        crate::job_list_with(&kinds, members)
    }
}

impl Listing for BrickList {
    fn with<B: Brick>(mut self) -> Self {
        self.bricks.push(Listed::of::<B>());
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
pub struct Spawner<'w, 's, B: Brick> {
    commands: Commands<'w, 's>,
    brick: PhantomData<fn() -> B>,
}

impl<B: Brick> Spawner<'_, '_, B> {
    pub fn spawn(&mut self, bundle: impl Bundle) -> Entity {
        let entity = self.commands.spawn_empty().id();
        let cause = engine_profile::carry();
        self.commands
            .queue(move |world: &mut World| spawned::<B, _>(world, entity, bundle, cause));
        entity
    }

    pub fn insert(&mut self, entity: Entity, bundle: impl Bundle) {
        self.commands
            .queue(move |world: &mut World| inserted::<B, _>(world, entity, bundle));
    }

    pub fn remove<T: Bundle>(&mut self, entity: Entity) {
        self.commands.queue(move |world: &mut World| {
            if let Ok(mut held) = world.get_entity_mut(entity) {
                held.remove::<T>();
            }
        });
    }

    pub fn despawn(&mut self, entity: Entity) {
        self.commands.queue(move |world: &mut World| {
            world.despawn(entity);
        });
    }

    pub fn set_parent(&mut self, child: Entity, parent: Option<Entity>) {
        self.set_parents(vec![(child, parent)]);
    }

    pub fn set_parents(&mut self, changes: Vec<(Entity, Option<Entity>)>) {
        self.commands
            .queue(move |world: &mut World| parented::<B>(world, &changes));
    }
}

#[cfg(any(feature = "dev", feature = "profiling"))]
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Caused(pub u32);

#[cfg(any(feature = "dev", feature = "profiling"))]
fn stamp(cause: Carried) -> Caused {
    Caused(cause.cause().0)
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
fn stamp(_: Carried) {}

fn spawned<B: Brick, T: Bundle>(world: &mut World, entity: Entity, bundle: T, cause: Carried) {
    let Ok(mut spawned) = world.get_entity_mut(entity) else {
        return;
    };
    spawned.insert((bundle, SpawnedBy::<B>::new(), stamp(cause)));
    let admitted = engine_profile::caused(cause.cause(), || crate::admit(world, entity, None, "$"));
    if let Err(error) = admitted {
        let _ = world.try_despawn(entity);
        refused::<B>(world, "spawn", error);
    }
}

fn inserted<B: Brick, T: Bundle>(world: &mut World, entity: Entity, bundle: T) {
    let Ok(mut held) = world.get_entity_mut(entity) else {
        return;
    };
    held.insert(bundle);
    let parts = world
        .register_bundle::<T>()
        .contributed_components()
        .to_vec();
    if let Err(error) = crate::admit(world, entity, Some(&parts), "$") {
        if let Ok(mut held) = world.get_entity_mut(entity) {
            held.remove::<T>();
        }
        refused::<B>(world, "insert", error);
    }
}

fn parented<B: Brick>(world: &mut World, changes: &[(Entity, Option<Entity>)]) {
    let before: Vec<(Entity, Option<Entity>)> = changes
        .iter()
        .map(|&(child, _)| (child, world.get::<Parent>(child).map(Parent::parent)))
        .collect();
    if let Err(error) = crate::set_parents(world, changes) {
        refused::<B>(world, "parent", error);
        return;
    }
    let admitted = changes
        .iter()
        .try_for_each(|&(child, _)| crate::admit(world, child, None, "$"));
    if let Err(error) = admitted {
        let _ = crate::set_parents(world, &before);
        refused::<B>(world, "parent", error);
    }
}

fn refused<B: Brick>(world: &mut World, call: &str, error: Error) {
    let path = format!("$.bricks.{}.spawner.{call}", B::SOURCE.package());
    report(world, error.within(&path));
}

fn report(world: &mut World, error: Error) {
    if let Some(mut registry) = world.get_resource_mut::<Registry>() {
        registry.report(error);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Schedule {
    FrameStart,
    FixedSimulation,
    Presentation,
}
