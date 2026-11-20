use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::bundle::Bundle;
use bevy_ecs::component::{Component, Immutable};
use bevy_ecs::entity::Entity;
use bevy_ecs::query::{QueryFilter, ReadOnlyQueryData};
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_ecs::system::{
    Commands, Local, Query, ReadOnlySystemParam, Res, ScheduleSystem, SystemParam,
};
use bevy_ecs::world::World;
use engine_registry::{BrickId, BrickRecord, BrickSystems, Registry, SettingsEntry, SpawnedBy};
use serde_json::Value;

use crate::moves::NodeKind;
use crate::{
    DevOnly, Error, ErrorCode, KindDecl, List, Mover, MoverId, Pace, SceneContext, Schedule,
    Section, Source, Switch,
};

pub trait Brick: 'static {
    const SOURCE: &'static Source;
    fn register(bricks: &mut Bricks<'_>);
}

pub struct Bricks<'r> {
    pub(crate) record: &'r mut BrickRecord,
    systems: &'r mut BrickSystems,
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
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        self.systems.add(schedule.label(), systems);
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
pub(crate) fn settings_entry<S: Section + Resource>() -> SettingsEntry {
    SettingsEntry::of::<S>(S::KEY, check_of::<S>, set_of::<S, S>, scene_of::<S>)
}

#[track_caller]
pub(crate) fn dev_settings_entry<S: Section>() -> SettingsEntry {
    SettingsEntry::of::<DevOnly<S>>(
        S::KEY,
        check_of::<S>,
        set_of::<S, DevOnly<S>>,
        scene_of::<S>,
    )
}

fn checked<S: Section>(value: &Value, path: &str) -> Result<S, Error> {
    let section: S = engine_settings::parse_value_at(value, path)?;
    section.check().map_err(|error| error.within(path))?;
    Ok(section)
}

fn check_of<S: Section>(value: &Value, path: &str) -> Result<(), Error> {
    checked::<S>(value, path).map(drop)
}

fn set_of<S: Section, R: Resource + From<S>>(
    world: &mut World,
    value: &Value,
    path: &str,
) -> Result<(), Error> {
    let section = checked::<S>(value, path)?;
    if let (Some(scene), Some(assets)) = (
        world.get_resource::<SceneContext>(),
        world.get_resource::<crate::Assets>(),
    ) {
        assets
            .scope(|| section.check_scene(scene))
            .map_err(|error| error.within(path))?;
    }
    world.insert_resource(R::from(section));
    Ok(())
}

fn scene_of<S: Section>(value: &Value, path: &str, scene: &dyn Any) -> Result<(), Error> {
    let Some(scene) = scene.downcast_ref::<SceneContext>() else {
        return Ok(());
    };
    let section: S = engine_settings::parse_value_at(value, path)?;
    section
        .check_scene(scene)
        .map_err(|error| error.within(path))
}

pub trait Listing: Sized {
    #[must_use]
    fn with<B: Brick>(self) -> Self;
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct Listed {
    record: fn() -> (BrickRecord, BrickSystems),
}

impl Listed {
    pub fn record(&self) -> (BrickRecord, BrickSystems) {
        (self.record)()
    }
}

fn record_of<B: Brick>() -> (BrickRecord, BrickSystems) {
    let source = B::SOURCE;
    let mut record = BrickRecord::new(source.package());
    record.set_game(source.game());
    if let Some(first) = source.modules().first() {
        let path = first.path();
        record.shader_prefix(path.split("::").next().unwrap_or(path));
    }
    record.render.push(Arc::new(source));
    let mut systems = BrickSystems::default();
    B::register(&mut Bricks {
        record: &mut record,
        systems: &mut systems,
    });
    (record, systems)
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
    registry: Res<'w, Registry>,
    by: Local<'s, Option<BrickId>>,
    brick: PhantomData<fn() -> B>,
}

impl<B: Brick> Spawner<'_, '_, B> {
    pub fn spawn(&mut self, bundle: impl Bundle) -> Entity {
        let entity = self.commands.spawn_empty().id();
        let package = B::SOURCE.package();
        if self.by.is_none() {
            *self.by = self.registry.id(package);
        }
        let by = *self.by;
        self.commands
            .queue(move |world: &mut World| spawned(world, entity, bundle, by, package));
        entity
    }

    pub fn insert(&mut self, entity: Entity, bundle: impl Bundle) {
        self.commands.queue(move |world: &mut World| {
            if let Ok(mut held) = world.get_entity_mut(entity) {
                held.insert(bundle);
            }
        });
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
        let package = B::SOURCE.package();
        self.commands
            .queue(move |world: &mut World| parented(world, &changes, package));
    }
}

fn spawned<T: Bundle>(
    world: &mut World,
    entity: Entity,
    bundle: T,
    by: Option<BrickId>,
    package: &'static str,
) {
    let Some(by) = by else {
        let _ = world.try_despawn(entity);
        let unlisted = Error::new(ErrorCode::UnknownReference, "$").got(package);
        return refused(world, package, "spawn", unlisted);
    };
    if let Ok(mut spawned) = world.get_entity_mut(entity) {
        spawned.insert((bundle, SpawnedBy(by)));
    }
}

fn parented(world: &mut World, changes: &[(Entity, Option<Entity>)], package: &'static str) {
    if let Err(error) = crate::set_parents(world, changes) {
        refused(world, package, "parent", error);
    }
}

fn refused(world: &mut World, package: &str, call: &str, error: Error) {
    let path = format!("$.bricks.{package}.spawner.{call}");
    report(world, error.within(&path));
}

fn report(world: &mut World, error: Error) {
    if let Some(mut registry) = world.get_resource_mut::<Registry>() {
        registry.report(error);
    }
}
