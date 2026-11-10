use std::any::TypeId;

use bevy_ecs::query::FilteredAccessSet;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{
    IntoSystem, StaticSystemParam, System, SystemParam, SystemParamItem, SystemState,
};
use bevy_ecs::world::World;
use engine_settings::{Error, ErrorCode, Expected};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::feeds::{Feed, StoreId};

pub const RESERVED_REPLY_KEYS: [&str; 4] = ["observation", "paused", "source", "rendering"];

pub trait Operation: Send + Sync + 'static {
    const NAME: &'static str;
    const CONTROL: bool = true;
    type Params: SystemParam + 'static;
    type Request: DeserializeOwned + schemars::JsonSchema + 'static;
    type Reply: Serialize + schemars::JsonSchema + 'static;

    fn run(
        params: SystemParamItem<'_, '_, Self::Params>,
        request: Self::Request,
    ) -> Result<Self::Reply, Error>;
}

pub type NamedRun = fn(&mut World, Value) -> Result<Value, Error>;

#[derive(Clone, Copy)]
pub struct Call {
    pub name: &'static str,
    pub control: bool,
    operation: fn() -> TypeId,
    run: NamedRun,
    access: fn(&mut World) -> FilteredAccessSet,
    request: fn() -> Value,
    reply: fn() -> Value,
}

impl Call {
    pub const fn of<O: Operation>() -> Self {
        Self {
            name: O::NAME,
            control: O::CONTROL,
            operation: TypeId::of::<O>,
            run: run_named::<O>,
            access: access_of::<O::Params>,
            request: schema_of::<O::Request>,
            reply: schema_of::<O::Reply>,
        }
    }

    pub const fn exclusive<Request, Reply>(
        name: &'static str,
        control: bool,
        run: NamedRun,
    ) -> Self
    where
        Request: schemars::JsonSchema + 'static,
        Reply: schemars::JsonSchema + 'static,
    {
        Self {
            name,
            control,
            operation: TypeId::of::<Request>,
            run,
            access: access_of::<()>,
            request: schema_of::<Request>,
            reply: schema_of::<Reply>,
        }
    }

    #[must_use]
    pub const fn observed(self) -> Self {
        Self {
            control: false,
            ..self
        }
    }

    pub fn declares<O: Operation>(&self) -> bool {
        (self.operation)() == TypeId::of::<O>()
    }

    pub fn run(&self, world: &mut World, request: Value) -> Result<Value, Error> {
        (self.run)(world, request)
    }

    pub fn access(&self, world: &mut World) -> FilteredAccessSet {
        (self.access)(world)
    }

    pub fn request_schema(&self) -> Value {
        (self.request)()
    }

    pub fn reply_schema(&self) -> Value {
        (self.reply)()
    }
}

pub fn schema_of<T: schemars::JsonSchema>() -> Value {
    schemars::schema_for!(T).to_value()
}

pub fn open_schema(schema: &Value) -> bool {
    match schema {
        Value::Object(fields) => fields
            .keys()
            .all(|key| ["$schema", "title", "description", "default"].contains(&key.as_str())),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_) => {
            true
        }
    }
}

pub fn access_of<P: SystemParam + 'static>(world: &mut World) -> FilteredAccessSet {
    let mut probe = IntoSystem::into_system(probe::<P>);
    probe.initialize(world)
}

fn probe<P: SystemParam + 'static>(_: StaticSystemParam<P>) {}

pub fn answer<O: Operation>(world: &mut World, request: O::Request) -> Result<O::Reply, Error> {
    let mut state = SystemState::<O::Params>::new(world);
    let params = state.get_mut(world).map_err(|refused| {
        Error::new(
            ErrorCode::MissingResource,
            format!("$.operations.{}", O::NAME),
        )
        .got(refused.to_string())
    })?;
    let reply = O::run(params, request);
    state.apply(world);
    reply
}

fn run_named<O: Operation>(world: &mut World, request: Value) -> Result<Value, Error> {
    let request: O::Request = engine_settings::parse_value(request)?;
    let reply = answer::<O>(world, request)?;
    let path = format!("$.operations.{}", O::NAME);
    let reply = serde_json::to_value(reply).map_err(|error| {
        Error::new(ErrorCode::InvalidValue, path.clone()).got(error.to_string())
    })?;
    match reply.as_object() {
        Some(fields) => {
            if let Some(key) = RESERVED_REPLY_KEYS
                .iter()
                .find(|key| fields.contains_key(**key))
            {
                return Err(Error::new(ErrorCode::InvalidValue, format!("{path}.{key}")).got(*key));
            }
        }
        None if reply.is_null() => {}
        None => {
            return Err(Error::new(ErrorCode::InvalidValue, path)
                .expected(Expected::Object)
                .got(reply));
        }
    }
    Ok(reply)
}

pub trait Event: Send + Sync + 'static {
    const NAME: &'static str;
    type Row: Serialize + schemars::JsonSchema + 'static;

    fn row(&self) -> Self::Row;
}

pub trait Raises<E: Event>: Resource {
    fn raised(&self) -> &Feed<E>;
}

#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct EventPage {
    #[schemars(with = "u64")]
    pub store: StoreId,
    pub first: u64,
    pub next: u64,
    pub missed: u64,
    pub rows: Vec<Value>,
}

#[derive(Clone, Copy)]
pub struct EventDecl {
    pub name: &'static str,
    read: fn(&World, u64) -> Option<EventPage>,
    row: fn() -> Value,
}

impl EventDecl {
    pub const fn of<E: Event, R: Raises<E>>() -> Self {
        Self {
            name: E::NAME,
            read: read_raised::<E, R>,
            row: schema_of::<E::Row>,
        }
    }

    pub const fn custom<Row: schemars::JsonSchema + 'static>(
        name: &'static str,
        read: fn(&World, u64) -> Option<EventPage>,
    ) -> Self {
        Self {
            name,
            read,
            row: schema_of::<Row>,
        }
    }

    pub fn read(&self, world: &World, after: u64) -> Option<EventPage> {
        (self.read)(world, after)
    }

    pub fn row_schema(&self) -> Value {
        (self.row)()
    }
}

pub fn page_of<T, Row: Serialize>(
    feed: &Feed<T>,
    after: u64,
    row: impl Fn(&T) -> Row,
) -> EventPage {
    let (first, listed) = feed.after(after);
    EventPage {
        store: feed.store(),
        first,
        next: feed.next(),
        missed: first.saturating_sub(after),
        rows: listed
            .iter()
            .map(|entry| serde_json::to_value(row(entry)).unwrap_or(Value::Null))
            .collect(),
    }
}

fn read_raised<E: Event, R: Raises<E>>(world: &World, after: u64) -> Option<EventPage> {
    let raised = world.get_resource::<R>()?.raised();
    Some(page_of(raised, after, E::row))
}
