use std::any::TypeId;
use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::component::Component;
use bevy_ecs::prelude::{Entity, Resource, World};
use engine_settings::ErrorCode;
use serde_json::Value;

use super::{
    Brick, BrickSet, BrickStats, EnvironmentSection, Error, Operation, crate_of, parse_value,
    parse_value_at, short_type_name,
};
use crate::RenderSnapshot;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Owned {
    pub type_id: TypeId,
    pub name: &'static str,
}

impl Owned {
    fn of<T: ?Sized + 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
        }
    }
}

pub struct BrickRecord {
    pub name: &'static str,
    pub type_path: &'static str,
    pub crate_name: &'static str,
    pub slot: Option<&'static str>,
    pub lists: Vec<&'static str>,
    pub settings: Vec<SettingsEntry>,
    pub stats: Vec<StatsEntry>,
    pub operations: Vec<OperationEntry>,
    pub inspectors: Vec<InspectorEntry>,
    pub extractions: Vec<fn(&World, &mut RenderSnapshot)>,
    pub resources: Vec<Owned>,
    pub components: Vec<Owned>,
}

impl BrickRecord {
    pub(super) fn new<B: Brick>(name: &'static str) -> Self {
        Self {
            name,
            type_path: std::any::type_name::<B>(),
            crate_name: crate_of::<B>(),
            slot: None,
            lists: Vec::new(),
            settings: Vec::new(),
            stats: Vec::new(),
            operations: Vec::new(),
            inspectors: Vec::new(),
            extractions: Vec::new(),
            resources: Vec::new(),
            components: Vec::new(),
        }
    }

    pub fn system_set(&self) -> BrickSet {
        BrickSet(self.name)
    }

    pub fn owns(&self, type_id: TypeId) -> bool {
        self.resources
            .iter()
            .chain(&self.components)
            .any(|owned| owned.type_id == type_id)
    }

    pub(super) fn own_resource<R: Resource>(&mut self) {
        let owned = Owned::of::<R>();
        if !self.resources.contains(&owned) {
            self.resources.push(owned);
        }
    }

    pub(super) fn own_component<C: Component>(&mut self) {
        let owned = Owned::of::<C>();
        if !self.components.contains(&owned) {
            self.components.push(owned);
        }
    }
}

#[derive(Resource, Default)]
pub struct BrickRegistry {
    bricks: Vec<BrickRecord>,
}

impl BrickRegistry {
    pub fn has(&self, name: &str) -> bool {
        self.brick(name).is_some()
    }

    pub fn brick(&self, name: &str) -> Option<&BrickRecord> {
        self.bricks.iter().find(|brick| brick.name == name)
    }

    pub fn bricks(&self) -> &[BrickRecord] {
        &self.bricks
    }

    pub fn names(&self) -> impl ExactSizeIterator<Item = &'static str> + '_ {
        self.bricks.iter().map(|brick| brick.name)
    }

    pub fn owner_of(&self, type_id: TypeId) -> Option<&BrickRecord> {
        self.bricks.iter().find(|brick| brick.owns(type_id))
    }

    pub fn owner_named(&self, name: &str) -> Option<&BrickRecord> {
        self.bricks.iter().find(|brick| {
            brick
                .resources
                .iter()
                .chain(&brick.components)
                .any(|owned| {
                    owned.name == name
                        || owned
                            .name
                            .split('<')
                            .next()
                            .and_then(|path| path.rsplit("::").next())
                            == Some(name)
                })
        })
    }

    pub fn sections(&self) -> impl Iterator<Item = &SettingsEntry> + '_ {
        self.bricks.iter().flat_map(|brick| &brick.settings)
    }

    pub fn section(&self, key: &str) -> Option<&SettingsEntry> {
        self.sections().find(|section| section.key == key)
    }

    pub fn operations(&self) -> impl Iterator<Item = &OperationEntry> + '_ {
        self.bricks.iter().flat_map(|brick| &brick.operations)
    }

    pub fn operation(&self, name: &str) -> Option<&OperationEntry> {
        self.operations().find(|operation| operation.name == name)
    }

    pub fn inspectors(&self) -> impl Iterator<Item = &InspectorEntry> + '_ {
        self.bricks.iter().flat_map(|brick| &brick.inspectors)
    }

    pub fn inspector(&self, name: &str) -> Option<&InspectorEntry> {
        self.inspectors().find(|inspector| inspector.name == name)
    }

    pub fn stats(&self) -> impl Iterator<Item = &StatsEntry> + '_ {
        self.bricks.iter().flat_map(|brick| &brick.stats)
    }

    pub fn extractions(&self) -> impl Iterator<Item = fn(&World, &mut RenderSnapshot)> + '_ {
        self.bricks
            .iter()
            .flat_map(|brick| brick.extractions.iter().copied())
    }

    pub fn frame_stats(&self, world: &World) -> BTreeMap<&'static str, Value> {
        self.stats()
            .map(|stats| (stats.key, stats.read(world)))
            .collect()
    }

    pub(super) fn admits(&self, name: &'static str) -> Result<(), Error> {
        if self.has(name) {
            return Err(Error::new(
                ErrorCode::DuplicateBrick,
                format!("$.bricks.{name}"),
                format!("the {name} brick is added twice: add each brick once"),
            ));
        }
        Ok(())
    }

    pub(super) fn add(&mut self, record: BrickRecord) -> Result<(), Error> {
        self.admits(record.name)?;
        let name = record.name;
        let taken = |kind: &str, key: &str, owner: &str| {
            Error::new(
                ErrorCode::DuplicateBrick,
                format!("$.bricks.{name}.{kind}.{key}"),
                format!("the {owner} brick already registers {kind} {key}: each has one brick"),
            )
        };
        for section in &record.settings {
            if let Some(known) = self.section(section.key) {
                return Err(taken("settings", section.key, known.brick));
            }
        }
        for stats in &record.stats {
            if let Some(known) = self.stats().find(|known| known.key == stats.key) {
                return Err(taken("stats", stats.key, known.brick));
            }
        }
        for operation in &record.operations {
            if let Some(known) = self.operation(operation.name) {
                return Err(taken("operations", operation.name, known.brick));
            }
        }
        for inspector in &record.inspectors {
            if let Some(known) = self.inspector(inspector.name) {
                return Err(taken("inspectors", inspector.name, known.brick));
            }
        }
        let mut record = record;
        record
            .resources
            .retain(|owned| self.owner_of(owned.type_id).is_none());
        record
            .components
            .retain(|owned| self.owner_of(owned.type_id).is_none());
        self.bricks.push(record);
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct SettingsEntry {
    pub key: &'static str,
    pub name: &'static str,
    pub opt_in: bool,
    pub brick: &'static str,
    pub type_id: TypeId,
    check: fn(Value, &str) -> Result<(), Error>,
    insert: fn(&mut World, Value, &str) -> Result<(), Error>,
    remove: fn(&mut World),
    reset: fn(&mut World),
    current: fn(&World) -> Value,
    sanitized: fn(&World) -> Value,
    default: fn() -> Value,
    present: fn(&World) -> bool,
    #[cfg(feature = "schema")]
    schema: fn() -> schemars::Schema,
}

impl SettingsEntry {
    pub(super) fn of<T: EnvironmentSection>(brick: &'static str) -> Self {
        Self {
            key: T::KEY,
            name: T::NAME,
            opt_in: T::OPT_IN,
            brick,
            type_id: TypeId::of::<T>(),
            check: |value, path| parse_section::<T>(value, path).map(drop),
            insert: |world, value, path| {
                let section = parse_section::<T>(value, path)?;
                world.insert_resource(section);
                Ok(())
            },
            remove: |world| {
                world.remove_resource::<T>();
            },
            reset: |world| world.insert_resource(T::default()),
            current: |world| {
                to_json(
                    world
                        .get_resource::<T>()
                        .copied()
                        .unwrap_or_else(T::section_off),
                )
            },
            sanitized: |world| {
                to_json(
                    world
                        .get_resource::<T>()
                        .copied()
                        .unwrap_or_else(T::section_off)
                        .section_sanitized(),
                )
            },
            default: || to_json(T::default()),
            present: |world| world.contains_resource::<T>(),
            #[cfg(feature = "schema")]
            schema: T::schema,
        }
    }

    pub fn check(&self, value: Value, path: &str) -> Result<(), Error> {
        (self.check)(value, path)
    }

    pub fn insert(&self, world: &mut World, value: Value, path: &str) -> Result<(), Error> {
        (self.insert)(world, value, path)
    }

    pub fn remove(&self, world: &mut World) {
        (self.remove)(world);
    }

    pub fn reset(&self, world: &mut World) {
        (self.reset)(world);
    }

    pub fn current(&self, world: &World) -> Value {
        (self.current)(world)
    }

    pub fn sanitized(&self, world: &World) -> Value {
        (self.sanitized)(world)
    }

    pub fn default_value(&self) -> Value {
        (self.default)()
    }

    pub fn present(&self, world: &World) -> bool {
        (self.present)(world)
    }

    #[cfg(feature = "schema")]
    pub fn schema(&self) -> schemars::Schema {
        (self.schema)()
    }
}

fn parse_section<T: EnvironmentSection>(value: Value, path: &str) -> Result<T, Error> {
    let section: T = parse_value_at(value, path)?;
    section.section_check().map_err(|invalid| {
        Error::new(
            ErrorCode::InvalidEnvironment,
            format!("{path}.{}", invalid.field),
            invalid.rule,
        )
    })?;
    Ok(section)
}

fn to_json(value: impl serde::Serialize) -> Value {
    serde_json::to_value(value).expect("registered values serialize")
}

#[derive(Clone, Copy)]
pub struct StatsEntry {
    pub key: &'static str,
    pub brick: &'static str,
    pub type_id: TypeId,
    read: fn(&World) -> Value,
    #[cfg(feature = "schema")]
    schema: fn() -> schemars::Schema,
}

impl StatsEntry {
    pub(super) fn of<T: BrickStats>(brick: &'static str) -> Self {
        Self {
            key: T::KEY,
            brick,
            type_id: TypeId::of::<T>(),
            read: |world| match world.get_resource::<T>() {
                Some(stats) => to_json(stats),
                None => to_json(T::default()),
            },
            #[cfg(feature = "schema")]
            schema: T::schema,
        }
    }

    pub fn read(&self, world: &World) -> Value {
        (self.read)(world)
    }

    #[cfg(feature = "schema")]
    pub fn schema(&self) -> schemars::Schema {
        (self.schema)()
    }
}

#[derive(Clone, Copy)]
pub struct OperationEntry {
    pub name: &'static str,
    pub control: bool,
    pub shows: &'static str,
    pub brick: &'static str,
    run: fn(&mut World, Value) -> Result<Value, Error>,
    #[cfg(feature = "schema")]
    schema: fn() -> schemars::Schema,
    #[cfg(feature = "schema")]
    result_schema: fn() -> Option<schemars::Schema>,
}

impl OperationEntry {
    pub(super) fn of<O: Operation>(brick: &'static str) -> Self {
        Self {
            name: O::NAME,
            control: O::CONTROL,
            shows: O::SHOWS,
            brick,
            run: |world, request| {
                let request: O::Request = parse_value(request)?;
                O::run(world, request)
            },
            #[cfg(feature = "schema")]
            schema: O::schema,
            #[cfg(feature = "schema")]
            result_schema: O::result_schema,
        }
    }

    pub fn run(&self, world: &mut World, request: Value) -> Result<Value, Error> {
        (self.run)(world, request)
    }

    #[cfg(feature = "schema")]
    pub fn schema(&self) -> schemars::Schema {
        (self.schema)()
    }

    #[cfg(feature = "schema")]
    pub fn result_schema(&self) -> Option<schemars::Schema> {
        (self.result_schema)()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InspectedKind {
    Component,
    Resource,
}

type ReadInspected = Arc<dyn Fn(&World, Option<Entity>) -> Option<Value> + Send + Sync>;

#[derive(Clone)]
pub struct InspectorEntry {
    pub name: &'static str,
    pub kind: InspectedKind,
    pub brick: &'static str,
    pub type_id: TypeId,
    read: ReadInspected,
    has: fn(&World, Option<Entity>) -> bool,
    #[cfg(feature = "schema")]
    schema: Option<fn() -> schemars::Schema>,
}

impl InspectorEntry {
    pub(super) fn resource<R: Resource, V: serde::Serialize + 'static>(
        brick: &'static str,
        view: fn(&World, &R) -> V,
    ) -> Self {
        Self {
            name: short_type_name::<R>(),
            kind: InspectedKind::Resource,
            brick,
            type_id: TypeId::of::<R>(),
            read: Arc::new(move |world: &World, _: Option<Entity>| {
                world.get_resource::<R>().map(|value| to_json(view(world, value)))
            }),
            has: |world, _| world.contains_resource::<R>(),
            #[cfg(feature = "schema")]
            schema: None,
        }
    }

    pub(super) fn component<C: Component, V: serde::Serialize + 'static>(
        brick: &'static str,
        view: fn(&World, Entity, &C) -> V,
    ) -> Self {
        Self {
            name: short_type_name::<C>(),
            kind: InspectedKind::Component,
            brick,
            type_id: TypeId::of::<C>(),
            read: Arc::new(move |world: &World, entity: Option<Entity>| {
                let entity = entity?;
                world
                    .get::<C>(entity)
                    .map(|value| to_json(view(world, entity, value)))
            }),
            has: |world, entity| entity.is_some_and(|entity| world.get::<C>(entity).is_some()),
            #[cfg(feature = "schema")]
            schema: None,
        }
    }

    #[cfg(feature = "schema")]
    #[must_use]
    pub(super) fn with_schema(mut self, schema: fn() -> schemars::Schema) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn read(&self, world: &World, entity: Option<Entity>) -> Option<Value> {
        (self.read)(world, entity)
    }

    pub fn has(&self, world: &World, entity: Option<Entity>) -> bool {
        (self.has)(world, entity)
    }

    #[cfg(feature = "schema")]
    pub fn schema(&self) -> Option<schemars::Schema> {
        self.schema.map(|schema| schema())
    }
}

#[cfg(feature = "schema")]
pub(super) fn schema_of<V: schemars::JsonSchema>() -> schemars::Schema {
    schemars::schema_for!(V)
}
