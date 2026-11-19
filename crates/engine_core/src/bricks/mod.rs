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
use engine_registry::{BrickRecord, Registry, SettingsEntry, SpawnedBy, StatsEntry};
use serde_json::Value;

use crate::moves::NodeKind;
use crate::{
    Error, KindDecl, List, Mover, MoverId, Pace, Parent, SceneContext, Section, Source, Switch,
};

pub trait Brick: 'static {
    const SOURCE: &'static Source;
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
    SettingsEntry::of::<S>(S::KEY, check_of::<S>, set_of::<S>, scene_of::<S>, S::schema)
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
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Listed {
    record: fn() -> BrickRecord,
}

impl Listed {
    pub fn record(&self) -> BrickRecord {
        (self.record)()
    }
}

fn record_of<B: Brick>() -> BrickRecord {
    let source = B::SOURCE;
    let mut record = BrickRecord::new::<B>(source.package());
    record.set_game(source.game());
    if let Some(first) = source.modules().first() {
        let path = first.path();
        record.shader_prefix(path.split("::").next().unwrap_or(path));
    }
    B::register(&mut Bricks {
        record: &mut record,
    });
    record
}

#[doc(hidden)]
#[derive(Clone, Default)]
pub struct BrickList {
    bricks: Vec<Listed>,
}

impl BrickList {
    pub fn listed(&self) -> &[Listed] {
        &self.bricks
    }
}

impl Listing for BrickList {
    fn with<B: Brick>(mut self) -> Self {
        self.bricks.push(Listed {
            record: record_of::<B>,
        });
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

fn spawned<B: Brick, T: Bundle>(world: &mut World, entity: Entity, bundle: T) {
    let Ok(mut spawned) = world.get_entity_mut(entity) else {
        return;
    };
    spawned.insert((bundle, SpawnedBy::<B>::new()));
    if let Err(error) = crate::admit(world, entity, None, "$") {
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
