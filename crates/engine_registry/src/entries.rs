use std::any::{Any, TypeId};
use std::panic::Location;

use bevy_ecs::component::Mutable;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_place::{Error, ErrorCode, List};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::BrickRecord;
use crate::members::{drop_members, fill_members};
use crate::registry::BrickId;

#[derive(Clone, Copy, Debug)]
pub struct Owned {
    pub type_id: TypeId,
    pub name: &'static str,
    pub(crate) init: fn(&mut World),
    pub(crate) survives: bool,
}

impl Owned {
    pub fn of<T: Resource + Default>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
            init: init_of::<T>,
            survives: false,
        }
    }

    pub fn init(&self, world: &mut World) {
        (self.init)(world);
    }
}

pub(crate) struct JoinEntry {
    pub(crate) list: TypeId,
    pub(crate) member: Option<Box<dyn Any + Send + Sync>>,
    pub(crate) fill: fn(&mut World, &mut [BrickRecord]),
    pub(crate) left: fn(&mut World, BrickId),
}

impl JoinEntry {
    pub(crate) fn of<L: List>(member: L::Member) -> Self {
        Self {
            list: TypeId::of::<L>(),
            member: Some(Box::new(member)),
            fill: fill_members::<L>,
            left: drop_members::<L>,
        }
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
    current: fn(&World) -> Result<Value, Error>,
    default: fn() -> Value,
    #[cfg(feature = "dev")]
    schema: fn() -> schemars::Schema,
    scene: fn(&Value, &str, &dyn Any) -> Result<(), Error>,
}

impl SettingsEntry {
    #[track_caller]
    pub fn of<T>(
        key: &'static str,
        check: fn(&Value, &str) -> Result<(), Error>,
        set: fn(&mut World, &Value, &str) -> Result<(), Error>,
        scene: fn(&Value, &str, &dyn Any) -> Result<(), Error>,
    ) -> Self
    where
        T: Resource + Clone + Default + Serialize + DeserializeOwned + schemars::JsonSchema,
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
            current: written_of::<T>,
            default: default_of::<T>,
            #[cfg(feature = "dev")]
            schema: schema_of::<T>,
            scene,
        }
    }

    pub fn owned(&self) -> Owned {
        Owned {
            type_id: self.type_id,
            name: self.type_name,
            init: self.init,
            survives: true,
        }
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

    #[cfg(feature = "dev")]
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
}

impl StatsEntry {
    #[track_caller]
    pub fn of<T>(key: &'static str) -> Self
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
        }
    }

    pub fn owned(&self) -> Owned {
        Owned {
            type_id: self.type_id,
            name: self.type_name,
            init: self.init,
            survives: false,
        }
    }

    pub fn read(&self, world: &World) -> Value {
        (self.read)(world)
    }
}

pub(crate) fn init_of<T: Resource + Default>(world: &mut World) {
    if !world.contains_resource::<T>() {
        world.insert_resource(T::default());
    }
}

fn reset_of<T: Resource + Default>(world: &mut World) {
    world.insert_resource(T::default());
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

fn default_of<T: Default + Serialize>() -> Value {
    to_json(&T::default())
}

#[cfg(feature = "dev")]
fn schema_of<T: schemars::JsonSchema>() -> schemars::Schema {
    schemars::schema_for!(T)
}

fn to_json(value: &impl Serialize) -> Value {
    serde_json::to_value(value).expect("registered_values_serialize")
}

#[cfg(feature = "dev")]
pub mod names {
    use std::any::Any;
    use std::borrow::Cow;

    use bevy_ecs::world::World;
    use engine_place::Error;
    use serde::Serialize;
    use serde_json::Value;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Kind {
        Setting,
        State,
        Count,
        Bytes,
        Time,
        Reason,
        Link,
        Call,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Lands {
        ThisFrame,
        NextFrame,
    }

    pub struct Reach<'a> {
        pub world: &'a World,
        pub parts: &'a [&'a dyn Any],
    }

    impl Reach<'_> {
        pub fn part<T: Any>(&self) -> Option<&T> {
            self.parts.iter().find_map(|part| part.downcast_ref::<T>())
        }
    }

    pub struct ReachMut<'a> {
        pub world: &'a mut World,
        pub parts: &'a mut [&'a mut dyn Any],
    }

    impl ReachMut<'_> {
        pub fn part<T: Any>(&mut self) -> Option<&mut T> {
            self.parts
                .iter_mut()
                .find_map(|part| part.downcast_mut::<T>())
        }

        pub fn read<R>(&self, run: impl FnOnce(&Reach<'_>) -> R) -> R {
            let parts: Vec<&dyn Any> = self.parts.iter().map(|part| &**part).collect();
            run(&Reach {
                world: self.world,
                parts: &parts,
            })
        }
    }

    pub enum Answer {
        Value(Value),
        Loop(Box<dyn Any + Send>),
    }

    type ReadFn = Box<dyn Fn(&Reach<'_>, &Value) -> Result<Value, Error> + Send + Sync>;
    type SetFn = Box<dyn Fn(&mut ReachMut<'_>, &Value) -> Result<Value, Error> + Send + Sync>;
    type CallFn = Box<dyn Fn(&mut ReachMut<'_>, &Value) -> Result<Answer, Error> + Send + Sync>;

    pub struct NameEntry {
        pub name: Cow<'static, str>,
        pub kind: Kind,
        pub unit: Option<&'static str>,
        pub owner: &'static str,
        pub args: Value,
        pub example: Value,
        pub lands: Option<Lands>,
        pub missing: Option<String>,
        pub changes_frame: Option<&'static str>,
        pub pages: bool,
        read: Option<ReadFn>,
        set: Option<SetFn>,
        call: Option<CallFn>,
    }

    impl NameEntry {
        fn named(name: Cow<'static, str>, kind: Kind) -> Self {
            Self {
                name,
                kind,
                unit: None,
                owner: "",
                args: Value::Object(serde_json::Map::new()),
                example: Value::Object(serde_json::Map::new()),
                lands: None,
                missing: None,
                changes_frame: None,
                pages: false,
                read: None,
                set: None,
                call: None,
            }
        }

        pub fn read(
            name: impl Into<Cow<'static, str>>,
            kind: Kind,
            read: impl Fn(&Reach<'_>, &Value) -> Result<Value, Error> + Send + Sync + 'static,
        ) -> Self {
            Self {
                read: Some(Box::new(read)),
                ..Self::named(name.into(), kind)
            }
        }

        pub fn call(
            name: impl Into<Cow<'static, str>>,
            call: impl Fn(&mut ReachMut<'_>, &Value) -> Result<Answer, Error> + Send + Sync + 'static,
        ) -> Self {
            Self {
                call: Some(Box::new(call)),
                ..Self::named(name.into(), Kind::Call)
            }
        }

        #[must_use]
        pub fn unit(mut self, unit: &'static str) -> Self {
            self.unit = Some(unit);
            self
        }

        #[must_use]
        pub fn args(mut self, shape: Value, example: Value) -> Self {
            self.args = shape;
            self.example = example;
            self
        }

        #[must_use]
        pub fn set(
            mut self,
            lands: Lands,
            set: impl Fn(&mut ReachMut<'_>, &Value) -> Result<Value, Error> + Send + Sync + 'static,
        ) -> Self {
            self.lands = Some(lands);
            self.set = Some(Box::new(set));
            self
        }

        #[must_use]
        pub fn unavailable(mut self, reason: impl Into<String>) -> Self {
            self.missing = Some(reason.into());
            self
        }

        #[must_use]
        pub fn changes_frame(mut self, text: &'static str) -> Self {
            self.changes_frame = Some(text);
            self
        }

        #[must_use]
        pub fn pages(mut self) -> Self {
            self.pages = true;
            self
        }

        #[must_use]
        pub fn owned_by(mut self, owner: &'static str) -> Self {
            self.owner = owner;
            self
        }

        pub fn readable(&self) -> bool {
            self.read.is_some()
        }

        pub fn settable(&self) -> bool {
            self.set.is_some()
        }

        pub fn callable(&self) -> bool {
            self.call.is_some()
        }

        pub fn read_value(&self, reach: &Reach<'_>, args: &Value) -> Option<Result<Value, Error>> {
            if let Some(reason) = &self.missing {
                return self.read.as_ref().map(|_| Ok(unavailable(reason)));
            }
            self.read.as_ref().map(|read| read(reach, args))
        }

        pub fn set_value(
            &self,
            reach: &mut ReachMut<'_>,
            args: &Value,
        ) -> Option<Result<Value, Error>> {
            self.set.as_ref().map(|set| set(reach, args))
        }

        pub fn call_with(
            &self,
            reach: &mut ReachMut<'_>,
            args: &Value,
        ) -> Option<Result<Answer, Error>> {
            self.call.as_ref().map(|call| call(reach, args))
        }
    }

    pub fn unavailable(reason: impl std::fmt::Display) -> Value {
        serde_json::json!({ "unavailable": reason.to_string() })
    }
}
