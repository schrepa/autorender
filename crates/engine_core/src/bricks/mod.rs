mod error;
mod extracted;
mod registry;
mod scene;

pub use error::{Error, parse_json, parse_value, parse_value_at};
pub use extracted::Extracted;
pub use registry::{
    BrickRecord, BrickRegistry, InspectedKind, InspectorEntry, OperationEntry, Owned,
    SettingsEntry, StatsEntry,
};
pub use scene::{AuthoredId, OperationScene, authored_entity, authored_id};

use std::sync::{Mutex, OnceLock, PoisonError};

use bevy_app::App;
use bevy_ecs::component::Component;
use bevy_ecs::message::{Message, Messages};
use bevy_ecs::prelude::{Entity, Resource, World};
use bevy_ecs::schedule::{InternedSystemSet, IntoScheduleConfigs, ScheduleLabel, SystemSet};
use bevy_ecs::system::ScheduleSystem;
use engine_settings::{ErrorCode, InvalidSetting};
use serde_json::Value;

use crate::RenderSnapshot;

pub trait Brick: Send + Sync + 'static {
    fn register(bricks: &mut Bricks<'_>);
}

pub trait AddBrick {
    fn add_brick<B: Brick>(&mut self) -> &mut Self;

    fn add_bricks(&mut self, bricks: BrickList) -> &mut Self;
}

#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BrickSet(pub &'static str);

pub trait EnvironmentSection:
    Resource
    + Copy
    + Default
    + PartialEq
    + serde::Serialize
    + serde::de::DeserializeOwned
    + std::fmt::Debug
{
    const KEY: &'static str;
    const NAME: &'static str;
    const OPT_IN: bool;
    fn section_off() -> Self;
    fn section_check(&self) -> Result<(), InvalidSetting>;
    #[must_use]
    fn section_sanitized(self) -> Self;
    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema;
}

pub trait BrickStats: Resource + Clone + Default + serde::Serialize {
    const KEY: &'static str;
    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema;
}

pub trait Operation: Send + Sync + 'static {
    const NAME: &'static str;
    const CONTROL: bool;
    const SHOWS: &'static str = "";
    type Request: serde::de::DeserializeOwned;
    fn run(world: &mut World, request: Self::Request) -> Result<Value, Error>;
    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema;
    #[cfg(feature = "schema")]
    fn result_schema() -> Option<schemars::Schema> {
        None
    }
}

pub trait Slot: 'static {
    const NAME: &'static str;
}

pub trait List: 'static {
    const NAME: &'static str;
}

pub struct Bricks<'a> {
    app: &'a mut App,
    record: BrickRecord,
}

impl Bricks<'_> {
    pub fn name(&self) -> &'static str {
        self.record.name
    }

    pub fn systems<M>(
        &mut self,
        schedule: impl ScheduleLabel,
        systems: impl IntoScheduleConfigs<ScheduleSystem, M>,
    ) -> &mut Self {
        self.app
            .add_systems(schedule, systems.in_set(BrickSet(self.record.name)));
        self
    }

    pub fn configure_sets<M>(
        &mut self,
        schedule: impl ScheduleLabel,
        sets: impl IntoScheduleConfigs<InternedSystemSet, M>,
    ) -> &mut Self {
        self.app.configure_sets(schedule, sets);
        self
    }

    pub fn resource<R: Resource + bevy_ecs::world::FromWorld>(&mut self) -> &mut Self {
        self.app.init_resource::<R>();
        self.record.own_resource::<R>();
        self
    }

    pub fn insert_resource<R: Resource>(&mut self, value: R) -> &mut Self {
        self.app.insert_resource(value);
        self.record.own_resource::<R>();
        self
    }

    pub fn component<C: Component>(&mut self) -> &mut Self {
        self.app.world_mut().register_component::<C>();
        self.record.own_component::<C>();
        self
    }

    pub fn message<M: Message>(&mut self) -> &mut Self {
        self.app.add_message::<M>();
        self.record.own_resource::<Messages<M>>();
        self
    }

    pub fn settings<T: EnvironmentSection>(&mut self) -> &mut Self {
        if !T::OPT_IN {
            self.app.init_resource::<T>();
        }
        self.record.own_resource::<T>();
        self.record
            .settings
            .push(SettingsEntry::of::<T>(self.record.name));
        self
    }

    pub fn stats<T: BrickStats>(&mut self) -> &mut Self {
        self.app.init_resource::<T>();
        self.record.own_resource::<T>();
        self.record
            .stats
            .push(StatsEntry::of::<T>(self.record.name));
        self
    }

    pub fn operation<O: Operation>(&mut self) -> &mut Self {
        self.record
            .operations
            .push(OperationEntry::of::<O>(self.record.name));
        self
    }

    #[cfg(not(feature = "schema"))]
    pub fn inspector<R: Resource, V: serde::Serialize + 'static>(
        &mut self,
        view: fn(&World, &R) -> V,
    ) -> &mut Self {
        let entry = InspectorEntry::resource(self.record.name, view);
        self.record.inspectors.push(entry);
        self
    }

    #[cfg(feature = "schema")]
    pub fn inspector<R: Resource, V: serde::Serialize + schemars::JsonSchema + 'static>(
        &mut self,
        view: fn(&World, &R) -> V,
    ) -> &mut Self {
        let entry = InspectorEntry::resource(self.record.name, view)
            .with_schema(registry::schema_of::<V>);
        self.record.inspectors.push(entry);
        self
    }

    #[cfg(not(feature = "schema"))]
    pub fn component_inspector<C: Component, V: serde::Serialize + 'static>(
        &mut self,
        view: fn(&World, Entity, &C) -> V,
    ) -> &mut Self {
        let entry = InspectorEntry::component(self.record.name, view);
        self.record.inspectors.push(entry);
        self
    }

    #[cfg(feature = "schema")]
    pub fn component_inspector<
        C: Component,
        V: serde::Serialize + schemars::JsonSchema + 'static,
    >(
        &mut self,
        view: fn(&World, Entity, &C) -> V,
    ) -> &mut Self {
        let entry = InspectorEntry::component(self.record.name, view)
            .with_schema(registry::schema_of::<V>);
        self.record.inspectors.push(entry);
        self
    }

    pub fn extract(&mut self, extract: fn(&World, &mut RenderSnapshot)) -> &mut Self {
        self.record.extractions.push(extract);
        self
    }

    pub fn slot<S: Slot>(&mut self) -> &mut Self {
        self.record.slot = Some(S::NAME);
        self
    }

    pub fn list<L: List>(&mut self) -> &mut Self {
        if !self.record.lists.contains(&L::NAME) {
            self.record.lists.push(L::NAME);
        }
        self
    }

    pub fn world_mut(&mut self) -> &mut World {
        self.app.world_mut()
    }
}

impl AddBrick for App {
    fn add_brick<B: Brick>(&mut self) -> &mut Self {
        let name = brick_name::<B>();
        if !self.world().contains_resource::<BrickRegistry>() {
            self.init_resource::<BrickRegistry>();
        }
        if let Err(error) = self.world().resource::<BrickRegistry>().admits(name) {
            panic!("{error}");
        }
        let mut bricks = Bricks {
            app: &mut *self,
            record: BrickRecord::new::<B>(name),
        };
        B::register(&mut bricks);
        let record = bricks.record;
        if let Err(error) = self
            .world_mut()
            .resource_mut::<BrickRegistry>()
            .add(record)
        {
            panic!("{error}");
        }
        self
    }

    fn add_bricks(&mut self, bricks: BrickList) -> &mut Self {
        for (_, add) in bricks.entries {
            add(self);
        }
        self
    }
}

#[derive(Clone, Default)]
pub struct BrickList {
    entries: Vec<(&'static str, fn(&mut App))>,
}

impl BrickList {
    #[must_use]
    pub fn with<B: Brick>(mut self) -> Self {
        self.entries.push((brick_name::<B>(), add_one::<B>));
        self
    }

    #[must_use]
    pub fn without<'n>(mut self, names: impl IntoIterator<Item = &'n str>) -> Self {
        let names: Vec<&str> = names.into_iter().collect();
        self.entries.retain(|(name, _)| !names.contains(name));
        self
    }

    pub fn names(&self) -> impl ExactSizeIterator<Item = &'static str> + '_ {
        self.entries.iter().map(|&(name, _)| name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|&(listed, _)| listed == name)
    }

    pub fn check_names<'n>(
        &self,
        names: impl IntoIterator<Item = &'n str>,
        path: &str,
    ) -> Result<Vec<&'static str>, Error> {
        names
            .into_iter()
            .filter(|name| !name.is_empty())
            .map(|name| {
                self.names()
                    .find(|&listed| listed == name)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::InvalidArguments,
                            path,
                            format!(
                                "{name}, allowed: {}",
                                self.names().collect::<Vec<_>>().join(", ")
                            ),
                        )
                    })
            })
            .collect()
    }
}

fn add_one<B: Brick>(app: &mut App) {
    app.add_brick::<B>();
}

pub fn brick_name<B: Brick>() -> &'static str {
    let short = short_type_name::<B>();
    let short = short
        .strip_suffix("Brick")
        .filter(|stem| !stem.is_empty())
        .unwrap_or(short);
    intern(&snake_case(short))
}

pub(crate) fn short_type_name<T: ?Sized>() -> &'static str {
    let full = std::any::type_name::<T>();
    let path = full.split('<').next().unwrap_or(full);
    path.rsplit("::").next().unwrap_or(path)
}

pub(crate) fn crate_of<T: ?Sized>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.split("::").next().unwrap_or(full)
}

fn snake_case(name: &str) -> String {
    let mut snake = String::with_capacity(name.len() + 4);
    for (index, character) in name.char_indices() {
        if character.is_ascii_uppercase() {
            if index > 0 {
                snake.push('_');
            }
            snake.push(character.to_ascii_lowercase());
        } else {
            snake.push(character);
        }
    }
    snake
}

fn intern(name: &str) -> &'static str {
    static NAMES: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let mut names = NAMES
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if let Some(&known) = names.iter().find(|&&known| known == name) {
        return known;
    }
    let named: &'static str = Box::leak(name.to_owned().into_boxed_str());
    names.push(named);
    named
}

pub fn missing_brick(name: &str, path: impl Into<String>) -> Error {
    Error::new(
        ErrorCode::MissingPlugin,
        path,
        format!("the {name} brick is not added: add it, or the standard bricks, to the engine"),
    )
}
