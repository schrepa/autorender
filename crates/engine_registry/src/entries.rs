use std::any::{Any, TypeId};
use std::panic::Location;

use bevy_ecs::component::{ComponentId, Mutable};
use bevy_ecs::query::FilteredAccessSet;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{IntoSystem, StaticSystemParam, System, SystemParam};
use bevy_ecs::world::World;
use engine_place::{Error, GpuStruct, List, PlaceDecl, Slot};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::gates::{GateId, Scope};
use crate::members::{add_member, init_members, refresh_members};

#[derive(Clone, Copy, Debug)]
pub struct Owned {
    pub type_id: TypeId,
    pub name: &'static str,
    pub(crate) init: fn(&mut World),
}

impl Owned {
    pub fn of<T: ?Sized + 'static>(init: fn(&mut World)) -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
            init,
        }
    }

    pub fn init(&self, world: &mut World) {
        (self.init)(world);
    }
}

pub struct Filled {
    pub place: PlaceDecl,
    pub slot: TypeId,
    pub scope: Scope,
    pub wgsl: Option<Box<dyn Any + Send + Sync>>,
    pub(crate) reset: fn(&mut World),
    pub(crate) id: fn(&mut World) -> ComponentId,
}

impl Filled {
    pub(crate) fn of<S: Slot>(scope: Scope, wgsl: Option<Box<dyn Any + Send + Sync>>) -> Self {
        Self {
            place: PlaceDecl::slot::<S>(),
            slot: TypeId::of::<S>(),
            scope,
            wgsl,
            reset: reset_of::<S>,
            id: resource_id_of::<S>,
        }
    }

    pub fn reset(&self, world: &mut World) {
        (self.reset)(world);
    }

    pub fn component_id(&self, world: &mut World) -> ComponentId {
        (self.id)(world)
    }
}

pub struct JoinEntry {
    pub place: PlaceDecl,
    pub list: TypeId,
    pub scope: Scope,
    pub wgsl: Option<Box<dyn Any + Send + Sync>>,
    pub(crate) member: Option<Box<dyn Any + Send + Sync>>,
    pub(crate) add: fn(&mut World, Box<dyn Any + Send + Sync>, GateId, (&'static str, TypeId)),
    pub(crate) init: fn(&mut World),
    pub(crate) refresh: fn(&mut World),
}

impl JoinEntry {
    pub(crate) fn of<L: List>(
        scope: Scope,
        member: L::Member,
        wgsl: Option<Box<dyn Any + Send + Sync>>,
    ) -> Self {
        Self {
            place: PlaceDecl::list::<L>(),
            list: TypeId::of::<L>(),
            scope,
            wgsl,
            member: Some(Box::new(member)),
            add: add_member::<L>,
            init: init_members::<L>,
            refresh: refresh_members::<L>,
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
    check: fn(&Value, &str) -> Result<(), Error>,
    set: fn(&mut World, &Value, &str) -> Result<(), Error>,
    reset: fn(&mut World),
    current: fn(&World) -> Value,
    default: fn() -> Value,
    present: fn(&World) -> bool,
    schema: fn() -> schemars::Schema,
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
        T: Resource + Clone + Default + Serialize,
    {
        Self {
            key,
            brick: "",
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
            at: Location::caller(),
            init: init_of::<T>,
            check,
            set,
            reset: reset_of::<T>,
            current: current_of::<T>,
            default: default_of::<T>,
            present: present_of::<T>,
            schema,
        }
    }

    pub fn owned(&self) -> Owned {
        Owned {
            type_id: self.type_id,
            name: self.type_name,
            init: self.init,
        }
    }

    pub fn check(&self, value: &Value, path: &str) -> Result<(), Error> {
        (self.check)(value, path)
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

    pub fn current(&self, world: &World) -> Value {
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
    read: fn(&World) -> Value,
    pending: fn(&World) -> u32,
    schema: fn() -> schemars::Schema,
}

impl StatsEntry {
    #[track_caller]
    pub fn of<T>(
        key: &'static str,
        pending: fn(&World) -> u32,
        schema: fn() -> schemars::Schema,
    ) -> Self
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
            read: current_of::<T>,
            pending,
            schema,
        }
    }

    pub fn owned(&self) -> Owned {
        Owned {
            type_id: self.type_id,
            name: self.type_name,
            init: self.init,
        }
    }

    pub fn read(&self, world: &World) -> Value {
        (self.read)(world)
    }

    pub fn pending(&self, world: &World) -> u32 {
        (self.pending)(world)
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
    pub at: &'static Location<'static>,
    run: fn(&mut World, Value) -> Result<Value, Error>,
    access: fn(&mut World) -> FilteredAccessSet,
}

impl OperationEntry {
    #[track_caller]
    pub fn of<P: SystemParam + 'static>(
        name: &'static str,
        control: bool,
        run: fn(&mut World, Value) -> Result<Value, Error>,
    ) -> Self {
        Self {
            name,
            brick: "",
            control,
            at: Location::caller(),
            run,
            access: access_of::<P>,
        }
    }

    pub fn run(&self, world: &mut World, request: Value) -> Result<Value, Error> {
        (self.run)(world, request)
    }

    pub fn access(&self, world: &mut World) -> FilteredAccessSet {
        (self.access)(world)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PieceEntry {
    pub type_id: TypeId,
    pub name: &'static str,
    pub wgsl_name: &'static str,
    pub wgsl: &'static str,
    pub size: usize,
    bytes: fn(&World) -> Option<&[u8]>,
    id: fn(&World) -> Option<ComponentId>,
}

impl PieceEntry {
    pub(crate) fn of<T: Resource<Mutability = Mutable> + Default + GpuStruct>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
            wgsl_name: T::WGSL_NAME,
            wgsl: T::WGSL,
            size: size_of::<T>(),
            bytes: bytes_of::<T>,
            id: World::component_id::<T>,
        }
    }

    pub fn bytes<'w>(&self, world: &'w World) -> Option<&'w [u8]> {
        (self.bytes)(world)
    }

    pub fn component_id(&self, world: &World) -> Option<ComponentId> {
        (self.id)(world)
    }
}

pub fn parse_section<T: DeserializeOwned>(
    value: &Value,
    path: &str,
    check: fn(&T) -> Result<(), Error>,
) -> Result<T, Error> {
    let section: T = engine_settings::parse_value_at(value.clone(), path)?;
    check(&section)?;
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

pub fn access_of<P: SystemParam + 'static>(world: &mut World) -> FilteredAccessSet {
    let mut probe = IntoSystem::into_system(probe::<P>);
    probe.initialize(world)
}

fn probe<P: SystemParam + 'static>(_: StaticSystemParam<P>) {}

pub(crate) fn init_of<T: Resource + Default>(world: &mut World) {
    if !world.contains_resource::<T>() {
        world.insert_resource(T::default());
    }
}

fn reset_of<T: Resource + Default>(world: &mut World) {
    world.insert_resource(T::default());
}

fn resource_id_of<T: Resource>(world: &mut World) -> ComponentId {
    world.register_component::<T>()
}

fn current_of<T: Resource + Default + Serialize>(world: &World) -> Value {
    match world.get_resource::<T>() {
        Some(value) => to_json(value),
        None => to_json(&T::default()),
    }
}

fn default_of<T: Default + Serialize>() -> Value {
    to_json(&T::default())
}

fn present_of<T: Resource>(world: &World) -> bool {
    world.contains_resource::<T>()
}

fn bytes_of<T: Resource + GpuStruct>(world: &World) -> Option<&[u8]> {
    world.get_resource::<T>().map(bytemuck::bytes_of)
}

fn to_json(value: &impl Serialize) -> Value {
    serde_json::to_value(value).expect("registered_values_serialize")
}
