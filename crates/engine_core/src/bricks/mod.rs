mod attribution;
mod explain;
mod extracted;
mod registry;
mod scene;

pub(crate) use attribution::under;
pub use attribution::{
    BrickCost, BrickCosts, BrickRow, BrickTable, CORE_PASSES, CORE_ROW, GPU_IDLE_ROW,
    UNATTRIBUTED_ROW, WARMUP_FRAMES,
};
pub use engine_settings::{
    Allowed, Detail, Error, Expected, parse_json, parse_value, parse_value_at,
};
pub use explain::{
    DrawState, ExplainEntry, ExplainRecord, ExplainRequest, ExplainedEntity, Explainer, source_line,
};
pub use extracted::Extracted;
pub use registry::{
    BrickRecord, BrickRegistry, InspectedKind, InspectorEntry, OperationEntry, Owned,
    SettingsEntry, StatsEntry,
};
pub use scene::{AuthoredId, OperationScene, authored_entity, authored_id};

use std::panic::Location;
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
use crate::smoke::{SmokeEntry, SmokeResult, SmokeSession};

pub trait Brick: Send + Sync + 'static {
    const SLOT: Option<&'static str> = None;

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
        let schedule = schedule.intern();
        #[cfg(any(feature = "dev", feature = "profiling"))]
        let before = attribution::system_keys(self.app, schedule);
        self.app
            .add_systems(schedule, systems.in_set(BrickSet(self.record.name)));
        #[cfg(any(feature = "dev", feature = "profiling"))]
        attribution::measure_systems(self.app, schedule, &before, self.record.name);
        self
    }

    pub fn passes(&mut self, names: &[&'static str]) -> &mut Self {
        for &name in names {
            if !self.record.passes.contains(&name) {
                self.record.passes.push(name);
            }
        }
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

    #[track_caller]
    pub fn settings<T: EnvironmentSection>(&mut self) -> &mut Self {
        let at = Location::caller();
        if !T::OPT_IN {
            self.app.init_resource::<T>();
        }
        self.record.own_resource::<T>();
        self.record
            .settings
            .push(SettingsEntry::of::<T>(self.record.name, at));
        self
    }

    #[track_caller]
    pub fn stats<T: BrickStats>(&mut self) -> &mut Self {
        let at = Location::caller();
        self.app.init_resource::<T>();
        self.record.own_resource::<T>();
        self.record
            .stats
            .push(StatsEntry::of::<T>(self.record.name, at));
        self
    }

    #[track_caller]
    pub fn operation<O: Operation>(&mut self) -> &mut Self {
        let at = Location::caller();
        self.record
            .operations
            .push(OperationEntry::of::<O>(self.record.name, at));
        self
    }

    #[track_caller]
    pub fn explain(&mut self, explain: Explainer) -> &mut Self {
        let at = Location::caller();
        self.record
            .explainers
            .push(ExplainEntry::of(self.record.name, explain, at));
        self
    }

    #[cfg(not(feature = "schema"))]
    #[track_caller]
    pub fn inspector<R: Resource, V: serde::Serialize + 'static>(
        &mut self,
        view: fn(&World, &R) -> V,
    ) -> &mut Self {
        let entry = InspectorEntry::resource(self.record.name, view, Location::caller());
        self.record.inspectors.push(entry);
        self
    }

    #[cfg(feature = "schema")]
    #[track_caller]
    pub fn inspector<R: Resource, V: serde::Serialize + schemars::JsonSchema + 'static>(
        &mut self,
        view: fn(&World, &R) -> V,
    ) -> &mut Self {
        let entry = InspectorEntry::resource(self.record.name, view, Location::caller())
            .with_schema(registry::schema_of::<V>);
        self.record.inspectors.push(entry);
        self
    }

    #[cfg(not(feature = "schema"))]
    #[track_caller]
    pub fn component_inspector<C: Component, V: serde::Serialize + 'static>(
        &mut self,
        view: fn(&World, Entity, &C) -> V,
    ) -> &mut Self {
        let entry = InspectorEntry::component(self.record.name, view, Location::caller());
        self.record.inspectors.push(entry);
        self
    }

    #[cfg(feature = "schema")]
    #[track_caller]
    pub fn component_inspector<
        C: Component,
        V: serde::Serialize + schemars::JsonSchema + 'static,
    >(
        &mut self,
        view: fn(&World, Entity, &C) -> V,
    ) -> &mut Self {
        let entry = InspectorEntry::component(self.record.name, view, Location::caller())
            .with_schema(registry::schema_of::<V>);
        self.record.inspectors.push(entry);
        self
    }

    pub fn extract(&mut self, extract: fn(&World, &mut RenderSnapshot)) -> &mut Self {
        self.record.extractions.push(extract);
        self
    }

    pub fn records(&mut self, kinds: &[&'static str]) -> &mut Self {
        for &kind in kinds {
            if !self.record.records.contains(&kind) {
                self.record.records.push(kind);
            }
        }
        self
    }

    pub fn list<L: List>(&mut self) -> &mut Self {
        if !self.record.lists.contains(&L::NAME) {
            self.record.lists.push(L::NAME);
        }
        self
    }

    pub fn smoke(
        &mut self,
        name: &'static str,
        gpu: bool,
        run: fn(&mut dyn SmokeSession) -> SmokeResult,
    ) -> &mut Self {
        self.record.smokes.push(SmokeEntry {
            name,
            brick: self.record.name,
            gpu,
            run,
        });
        self
    }

    pub fn world_mut(&mut self) -> &mut World {
        self.app.world_mut()
    }

    pub fn extension<T: std::any::Any + Send + Sync + Default>(&mut self) -> Option<&mut T> {
        self.record.extension_mut::<T>()
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
        if let Err(error) = self.world_mut().resource_mut::<BrickRegistry>().add(record) {
            panic!("{error}");
        }
        self
    }

    fn add_bricks(&mut self, bricks: BrickList) -> &mut Self {
        if let Err(error) = bricks.check() {
            panic!("{error}");
        }
        if !self.world().contains_resource::<BrickRegistry>() {
            self.init_resource::<BrickRegistry>();
        }
        self.world_mut()
            .resource_mut::<BrickRegistry>()
            .know_fillers(bricks.fillers.iter().copied());
        for listed in bricks.entries {
            (listed.add)(self);
        }
        self.world_mut()
            .resource_mut::<BrickRegistry>()
            .leave_out(bricks.left_out);
        self
    }
}

type AddFn = fn(&mut App);

#[derive(Clone, Copy)]
struct Listed {
    name: &'static str,
    slot: Option<&'static str>,
    add: AddFn,
}

impl Listed {
    fn of<B: Brick>() -> Self {
        Self {
            name: brick_name::<B>(),
            slot: B::SLOT,
            add: add_one::<B>,
        }
    }
}

#[derive(Clone, Default)]
pub struct BrickList {
    entries: Vec<Listed>,
    left_out: Vec<&'static str>,
    fillers: Vec<(&'static str, &'static str)>,
    errors: Vec<Error>,
}

impl BrickList {
    #[must_use]
    pub fn with<B: Brick>(mut self) -> Self {
        let listed = Listed::of::<B>();
        self.know(listed);
        self.entries.push(listed);
        self
    }

    #[must_use]
    pub fn with_before<B: Brick>(mut self, before: &str) -> Self {
        let listed = Listed::of::<B>();
        self.know(listed);
        match self.entries.iter().position(|known| known.name == before) {
            Some(index) => self.entries.insert(index, listed),
            None => {
                let error = Error::new(
                    ErrorCode::InvalidArguments,
                    format!("$.bricks.{}.before", listed.name),
                )
                .got(before)
                .variants(self.names());
                self.errors.push(error);
            }
        }
        self
    }

    #[must_use]
    pub fn fill<B: Brick>(mut self) -> Self {
        let listed = Listed::of::<B>();
        let Some(slot) = listed.slot else {
            let error = Error::new(
                ErrorCode::InvalidArguments,
                format!("$.bricks.{}.slot", listed.name),
            )
            .expected(Expected::Present);
            self.errors.push(error);
            return self;
        };
        self.know(listed);
        match self
            .entries
            .iter()
            .position(|known| known.slot == Some(slot))
        {
            Some(index) => {
                let replaced = std::mem::replace(&mut self.entries[index], listed);
                if replaced.name != listed.name && !self.left_out.contains(&replaced.name) {
                    self.left_out.push(replaced.name);
                }
            }
            None => self.entries.push(listed),
        }
        self.left_out.retain(|&name| name != listed.name);
        self
    }

    #[must_use]
    pub fn without<'n>(mut self, names: impl IntoIterator<Item = &'n str>) -> Self {
        let names: Vec<&str> = names.into_iter().collect();
        let left_out = &mut self.left_out;
        self.entries.retain(|listed| {
            let kept = !names.contains(&listed.name);
            if !kept && !left_out.contains(&listed.name) {
                left_out.push(listed.name);
            }
            kept
        });
        self
    }

    fn know(&mut self, listed: Listed) {
        if let Some(slot) = listed.slot
            && !self.fillers.contains(&(slot, listed.name))
        {
            self.fillers.push((slot, listed.name));
        }
    }

    pub fn names(&self) -> impl ExactSizeIterator<Item = &'static str> + '_ {
        self.entries.iter().map(|listed| listed.name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|listed| listed.name == name)
    }

    pub fn filler(&self, slot: &str) -> Option<&'static str> {
        self.entries
            .iter()
            .find(|listed| listed.slot == Some(slot))
            .map(|listed| listed.name)
    }

    pub fn fillers<'a>(&'a self, slot: &'a str) -> impl Iterator<Item = &'static str> + 'a {
        self.fillers
            .iter()
            .filter(move |&&(known, _)| known == slot)
            .map(|&(_, name)| name)
    }

    pub fn check(&self) -> Result<(), Error> {
        if let Some(error) = self.errors.first() {
            return Err(error.clone());
        }
        for (index, listed) in self.entries.iter().enumerate() {
            let Some(slot) = listed.slot else {
                continue;
            };
            if let Some(first) = self.entries[..index]
                .iter()
                .find(|earlier| earlier.slot == Some(slot))
            {
                return Err(
                    Error::new(ErrorCode::SlotFilledTwice, format!("$.slots.{slot}"))
                        .expected(Expected::Unique)
                        .got(listed.name)
                        .nearest(first.name)
                        .variants(self.fillers(slot)),
                );
            }
        }
        Ok(())
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
                self.names().find(|&listed| listed == name).ok_or_else(|| {
                    Error::new(ErrorCode::InvalidArguments, path)
                        .got(name)
                        .variants(self.names())
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
    Error::new(ErrorCode::MissingPlugin, path)
        .expected(Expected::Present)
        .got(name)
}
