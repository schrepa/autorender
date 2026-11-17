use std::any::{Any, TypeId};
use std::panic::Location;

use bevy_ecs::component::{ComponentId, Mutable};
use bevy_ecs::query::FilteredAccessSet;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_place::{Call, Callable, Error, ErrorCode, Lands, List, PlaceDecl};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::members::{add_member, drop_members, each_member, init_members};
use crate::registry::MemberEach;
use crate::state::{GateId, Scope};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Kept {
    Dropped(fn(&mut World)),
    Survives(fn(&mut World)),
}

#[derive(Clone, Copy, Debug)]
pub struct Owned {
    pub type_id: TypeId,
    pub name: &'static str,
    pub(crate) init: fn(&mut World),
    pub(crate) kept: Kept,
}

impl Owned {
    pub fn of<T: Resource + Default>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
            init: init_of::<T>,
            kept: Kept::Dropped(remove_of::<T>),
        }
    }

    pub fn init(&self, world: &mut World) {
        (self.init)(world);
    }

    pub fn survives(&self) -> bool {
        matches!(self.kept, Kept::Survives(_))
    }
}

pub struct Filled {
    pub place: PlaceDecl,
    pub slot: TypeId,
    pub scope: Scope,
    pub(crate) reset: fn(&mut World),
    pub(crate) id: fn(&mut World) -> ComponentId,
}

impl Filled {
    pub fn reset(&self, world: &mut World) {
        (self.reset)(world);
    }

    pub fn component_id(&self, world: &mut World) -> ComponentId {
        (self.id)(world)
    }
}

type MemberAdd = fn(&mut World, Box<dyn Any + Send + Sync>, GateId, (&'static str, TypeId));

pub struct JoinEntry {
    pub place: PlaceDecl,
    pub list: TypeId,
    pub scope: Scope,
    pub(crate) member: Option<Box<dyn Any + Send + Sync>>,
    pub(crate) add: MemberAdd,
    pub(crate) init: fn(&mut World),
    pub(crate) each: MemberEach,
    pub(crate) left: fn(&mut World, GateId),
}

impl JoinEntry {
    pub(crate) fn of<L: List>(scope: Scope, member: L::Member) -> Self {
        Self {
            place: PlaceDecl::list::<L>(),
            list: TypeId::of::<L>(),
            scope,
            member: Some(Box::new(member)),
            add: add_member::<L>,
            init: init_members::<L>,
            each: each_member::<L>,
            left: drop_members::<L>,
        }
    }

    pub fn member<M: 'static>(&self) -> Option<&M> {
        self.member.as_deref()?.downcast_ref::<M>()
    }

    pub fn take_member(&mut self) -> Option<Box<dyn Any + Send + Sync>> {
        self.member.take()
    }
}

#[derive(Clone, Copy)]
pub struct SettingsEntry {
    pub key: &'static str,
    pub brick: &'static str,
    pub type_id: TypeId,
    pub type_name: &'static str,
    pub at: &'static Location<'static>,
    init: fn(&mut World),
    parse: fn(&Value, &str) -> Result<(), Error>,
    check: fn(&Value, &str) -> Result<(), Error>,
    set: fn(&mut World, &Value, &str) -> Result<(), Error>,
    reset: fn(&mut World),
    touch: fn(&mut World),
    current: fn(&World) -> Result<Value, Error>,
    default: fn() -> Value,
    present: fn(&World) -> bool,
    schema: fn() -> schemars::Schema,
    scene: fn(&Value, &str, &dyn Any) -> Result<(), Error>,
}

impl SettingsEntry {
    #[track_caller]
    pub fn of<T>(
        key: &'static str,
        check: fn(&Value, &str) -> Result<(), Error>,
        set: fn(&mut World, &Value, &str) -> Result<(), Error>,
        schema: fn() -> schemars::Schema,
    ) -> Self
    where
        T: Resource + Clone + Default + Serialize + DeserializeOwned,
    {
        Self {
            key,
            brick: "",
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            at: Location::caller(),
            init: init_of::<T>,
            parse: parse_of::<T>,
            check,
            set,
            reset: reset_of::<T>,
            touch: touch_of::<T>,
            current: written_of::<T>,
            default: default_of::<T>,
            present: present_of::<T>,
            schema,
            scene: no_scene_check,
        }
    }

    #[must_use]
    pub fn scene_check(mut self, scene: fn(&Value, &str, &dyn Any) -> Result<(), Error>) -> Self {
        self.scene = scene;
        self
    }

    pub fn owned(&self) -> Owned {
        Owned {
            type_id: self.type_id,
            name: self.type_name,
            init: self.init,
            kept: Kept::Survives(self.touch),
        }
    }

    pub fn parse(&self, value: &Value, path: &str) -> Result<(), Error> {
        (self.parse)(value, path)
    }

    pub fn check(&self, value: &Value, path: &str) -> Result<(), Error> {
        (self.check)(value, path)
    }

    pub fn check_scene(&self, value: &Value, path: &str, scene: &dyn Any) -> Result<(), Error> {
        (self.scene)(value, path, scene)
    }

    pub fn set(&self, world: &mut World, value: &Value) -> Result<(), Error> {
        (self.set)(world, value, &format!("$.{}", self.key))
    }

    pub fn set_at(&self, world: &mut World, value: &Value, path: &str) -> Result<(), Error> {
        (self.set)(world, value, path)
    }

    pub fn reset(&self, world: &mut World) {
        (self.reset)(world);
    }

    pub fn current(&self, world: &World) -> Result<Value, Error> {
        (self.current)(world)
    }

    pub fn default_value(&self) -> Value {
        (self.default)()
    }

    pub fn present(&self, world: &World) -> bool {
        (self.present)(world)
    }

    pub fn schema(&self) -> schemars::Schema {
        (self.schema)()
    }
}

#[derive(Clone, Copy)]
pub struct StatsEntry {
    pub key: &'static str,
    pub brick: &'static str,
    pub type_id: TypeId,
    pub type_name: &'static str,
    pub at: &'static Location<'static>,
    init: fn(&mut World),
    remove: fn(&mut World),
    read: fn(&World) -> Value,
    schema: fn() -> schemars::Schema,
}

impl StatsEntry {
    #[track_caller]
    pub fn of<T>(key: &'static str, schema: fn() -> schemars::Schema) -> Self
    where
        T: Resource<Mutability = Mutable> + Default + Serialize,
    {
        Self {
            key,
            brick: "",
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            at: Location::caller(),
            init: init_of::<T>,
            remove: remove_of::<T>,
            read: current_of::<T>,
            schema,
        }
    }

    pub fn owned(&self) -> Owned {
        Owned {
            type_id: self.type_id,
            name: self.type_name,
            init: self.init,
            kept: Kept::Dropped(self.remove),
        }
    }

    pub fn read(&self, world: &World) -> Value {
        (self.read)(world)
    }

    pub fn schema(&self) -> schemars::Schema {
        (self.schema)()
    }
}

#[derive(Clone, Copy)]
pub struct OperationEntry {
    pub name: &'static str,
    pub brick: &'static str,
    pub control: bool,
    pub lands: Lands,
    pub at: &'static Location<'static>,
    call: Call,
}

impl OperationEntry {
    #[track_caller]
    pub fn of(call: Call) -> Self {
        Self {
            name: call.name,
            brick: "",
            control: call.control,
            lands: call.lands,
            at: Location::caller(),
            call,
        }
    }

    pub fn declares<O: Callable>(&self) -> bool {
        self.call.declares::<O>()
    }

    pub fn run(&self, world: &mut World, request: Value) -> Result<Value, Error> {
        self.call.run(world, request)
    }

    pub fn access(&self, world: &mut World) -> FilteredAccessSet {
        self.call.access(world)
    }

    pub fn request_schema(&self, world: &World) -> Value {
        self.call.request_schema(world)
    }

    pub fn reply_schema(&self, world: &World) -> Value {
        self.call.reply_schema(world)
    }
}

fn no_scene_check(_: &Value, _: &str, _: &dyn Any) -> Result<(), Error> {
    Ok(())
}

pub fn parse_section<T: DeserializeOwned>(
    value: &Value,
    path: &str,
    check: fn(&T) -> Result<(), Error>,
) -> Result<T, Error> {
    let section: T = engine_settings::parse_value_at(value.clone(), path)?;
    check(&section).map_err(|error| error.within(path))?;
    Ok(section)
}

pub fn set_section<T: Resource + DeserializeOwned>(
    world: &mut World,
    value: &Value,
    path: &str,
    check: fn(&T) -> Result<(), Error>,
) -> Result<(), Error> {
    let section = parse_section::<T>(value, path, check)?;
    world.insert_resource(section);
    Ok(())
}

pub(crate) fn init_of<T: Resource + Default>(world: &mut World) {
    if !world.contains_resource::<T>() {
        world.insert_resource(T::default());
    }
}

fn reset_of<T: Resource + Default>(world: &mut World) {
    world.insert_resource(T::default());
}

fn remove_of<T: Resource>(world: &mut World) {
    world.remove_resource::<T>();
}

fn touch_of<T: Resource>(world: &mut World) {
    let _ = world.modify_resource::<T, ()>(|_| {});
}

fn current_of<T: Resource + Default + Serialize>(world: &World) -> Value {
    match world.get_resource::<T>() {
        Some(value) => to_json(value),
        None => to_json(&T::default()),
    }
}

fn written_of<T: Resource + Default + Serialize>(world: &World) -> Result<Value, Error> {
    let written = match world.get_resource::<T>() {
        Some(value) => serde_json::to_value(value),
        None => serde_json::to_value(T::default()),
    };
    written.map_err(|error| {
        Error::new(ErrorCode::InvalidValue, "$").got(format!(
            "{} cannot be written here: {error}",
            std::any::type_name::<T>()
        ))
    })
}

fn parse_of<T: DeserializeOwned>(value: &Value, path: &str) -> Result<(), Error> {
    engine_settings::parse_value_at::<T>(value.clone(), path).map(drop)
}

fn default_of<T: Default + Serialize>() -> Value {
    to_json(&T::default())
}

fn present_of<T: Resource>(world: &World) -> bool {
    world.contains_resource::<T>()
}

fn to_json(value: &impl Serialize) -> Value {
    serde_json::to_value(value).expect("registered_values_serialize")
}
