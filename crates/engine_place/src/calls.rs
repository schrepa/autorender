use std::any::TypeId;

use bevy_ecs::query::FilteredAccessSet;
use bevy_ecs::system::{
    IntoSystem, StaticSystemParam, System, SystemParam, SystemParamItem, SystemState,
};
use bevy_ecs::world::World;
use engine_codec::Allowed;
use engine_settings::{Error, ErrorCode, Expected};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

pub const RESERVED_REPLY_KEYS: [&str; 4] = ["observation", "paused", "source", "rendering"];

pub const SCENE_SCHEMA: &str = "scene.schema.json";

pub const AUTHORED_COMPONENTS: &str = "/$defs/SceneObject/properties";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Lands {
    Now,
    NextStep,
    NextFrame,
}

impl Lands {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Now => "now",
            Self::NextStep => "next_step",
            Self::NextFrame => "next_frame",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Refusal {
    pub reason: String,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub got: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nearest: Option<String>,
}

fn text_of(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) | Value::Object(_) => {
            value.to_string()
        }
    }
}

impl Refusal {
    pub fn of(error: &Error) -> Self {
        let detail = &error.detail;
        let allowed = match &detail.allowed {
            None => Vec::new(),
            Some(Allowed::Range { min, max }) => vec![min.to_string(), max.to_string()],
            Some(Allowed::AtMost { max }) => vec![max.to_string()],
            Some(Allowed::AtLeast { min }) => vec![min.to_string()],
            Some(Allowed::Variants { values }) => values.clone(),
        };
        Self {
            reason: error.code.as_str().to_owned(),
            path: error.path.clone(),
            expected: detail
                .expected
                .and_then(|expected| serde_json::to_value(expected).ok())
                .map(|expected| text_of(&expected)),
            allowed,
            got: detail.got.as_ref().map(text_of),
            nearest: detail.nearest.as_ref().map(text_of),
        }
    }
}

impl From<&Error> for Refusal {
    fn from(error: &Error) -> Self {
        Self::of(error)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Each<T> {
    Done(T),
    Refused(Refusal),
}

impl<T> Each<T> {
    pub fn of(index: usize, list: &str, result: Result<T, Error>) -> Self {
        match result {
            Ok(done) => Self::Done(done),
            Err(error) => Self::Refused(Refusal::of(&error.within(&format!("$.{list}[{index}]")))),
        }
    }

    pub fn done(&self) -> bool {
        matches!(self, Self::Done(_))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct Rows<T> {
    pub done: usize,
    pub refused: usize,
    pub rows: Vec<Each<T>>,
}

impl<T> Rows<T> {
    pub fn each<R>(
        list: &str,
        requests: impl IntoIterator<Item = R>,
        mut run: impl FnMut(R) -> Result<T, Error>,
    ) -> Self {
        Self::of(
            requests
                .into_iter()
                .enumerate()
                .map(|(index, request)| Each::of(index, list, run(request)))
                .collect(),
        )
    }

    pub fn of(rows: Vec<Each<T>>) -> Self {
        let done = rows.iter().filter(|row| row.done()).count();
        Self {
            done,
            refused: rows.len() - done,
            rows,
        }
    }
}

pub trait Operation: Send + Sync + 'static {
    const NAME: &'static str;
    const CONTROL: bool = true;
    const LANDS: Lands = Lands::Now;
    type Params: SystemParam + 'static;
    type Request: DeserializeOwned + schemars::JsonSchema + 'static;
    type Reply: Serialize + schemars::JsonSchema + 'static;

    fn run(
        params: SystemParamItem<'_, '_, Self::Params>,
        request: Self::Request,
    ) -> Result<Self::Reply, Error>;

    fn request_schema(_: &World) -> Value {
        schema_of::<Self::Request>()
    }
}

pub trait Callable: Send + Sync + 'static {
    const NAME: &'static str;
    const CONTROL: bool = true;
    const LANDS: Lands = Lands::Now;
    type Request: DeserializeOwned + schemars::JsonSchema + 'static;
    type Reply: Serialize + schemars::JsonSchema + 'static;

    fn answer(world: &mut World, request: Self::Request) -> Result<Self::Reply, Error>;

    fn access(world: &mut World) -> FilteredAccessSet {
        access_of::<()>(world)
    }

    fn request_schema(_: &World) -> Value {
        schema_of::<Self::Request>()
    }
}

impl<O: Operation> Callable for O {
    const NAME: &'static str = <O as Operation>::NAME;
    const CONTROL: bool = <O as Operation>::CONTROL;
    const LANDS: Lands = <O as Operation>::LANDS;
    type Request = <O as Operation>::Request;
    type Reply = <O as Operation>::Reply;

    fn answer(world: &mut World, request: Self::Request) -> Result<Self::Reply, Error> {
        answer::<O>(world, request)
    }

    fn access(world: &mut World) -> FilteredAccessSet {
        access_of::<O::Params>(world)
    }

    fn request_schema(world: &World) -> Value {
        <O as Operation>::request_schema(world)
    }
}

pub type NamedRun = fn(&mut World, Value) -> Result<Value, Error>;

pub type SchemaIn = fn(&World) -> Value;

#[derive(Clone, Copy)]
pub struct Call {
    pub name: &'static str,
    pub control: bool,
    pub lands: Lands,
    operation: fn() -> TypeId,
    run: NamedRun,
    access: fn(&mut World) -> FilteredAccessSet,
    request: SchemaIn,
    reply: fn() -> Value,
}

impl Call {
    pub const fn of<O: Callable>() -> Self {
        Self {
            name: O::NAME,
            control: O::CONTROL,
            lands: O::LANDS,
            operation: TypeId::of::<O>,
            run: run_named::<O>,
            access: O::access,
            request: O::request_schema,
            reply: schema_of::<O::Reply>,
        }
    }

    #[must_use]
    pub const fn observed(self) -> Self {
        Self {
            control: false,
            ..self
        }
    }

    pub fn declares<O: Callable>(&self) -> bool {
        (self.operation)() == TypeId::of::<O>()
    }

    pub fn run(&self, world: &mut World, request: Value) -> Result<Value, Error> {
        (self.run)(world, request)
    }

    pub fn access(&self, world: &mut World) -> FilteredAccessSet {
        (self.access)(world)
    }

    pub fn request_schema(&self, world: &World) -> Value {
        (self.request)(world)
    }

    pub fn reply_schema(&self, _: &World) -> Value {
        (self.reply)()
    }
}

pub fn schema_of<T: schemars::JsonSchema>() -> Value {
    schemars::schema_for!(T).to_value()
}

pub fn authored_component(key: &str) -> Value {
    serde_json::json!({ "$ref": format!("{SCENE_SCHEMA}#{AUTHORED_COMPONENTS}/{key}") })
}

pub fn scene_references(schema: &Value) -> Vec<String> {
    let mut found = Vec::new();
    references_at(schema, &mut found);
    found
}

fn references_at(schema: &Value, found: &mut Vec<String>) {
    match schema {
        Value::Object(fields) => {
            for (key, value) in fields {
                let pointer = value
                    .as_str()
                    .filter(|_| key == "$ref")
                    .and_then(|target| target.strip_prefix(SCENE_SCHEMA))
                    .and_then(|rest| rest.strip_prefix('#'));
                match pointer {
                    Some(pointer) => found.push(pointer.to_owned()),
                    None => references_at(value, found),
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                references_at(value, found);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
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

pub fn open_parts(schema: &Value) -> Vec<String> {
    let mut found = Vec::new();
    open_at(schema, "$", &mut found);
    found
}

fn open_at(schema: &Value, at: &str, found: &mut Vec<String>) {
    let fields = match schema {
        Value::Bool(true) => {
            found.push(at.to_owned());
            return;
        }
        Value::Object(fields) => fields,
        Value::Null
        | Value::Bool(false)
        | Value::Number(_)
        | Value::String(_)
        | Value::Array(_) => {
            return;
        }
    };
    if open_schema(schema) {
        found.push(at.to_owned());
        return;
    }
    for (key, value) in fields {
        match key.as_str() {
            "properties" | "$defs" | "patternProperties" => {
                for (name, inner) in value.as_object().into_iter().flatten() {
                    open_at(inner, &format!("{at}.{name}"), found);
                }
            }
            "items" | "additionalProperties" | "not" => {
                open_at(value, &format!("{at}.{key}"), found);
            }
            "anyOf" | "oneOf" | "allOf" | "prefixItems" => {
                for (index, inner) in value.as_array().into_iter().flatten().enumerate() {
                    open_at(inner, &format!("{at}.{key}[{index}]"), found);
                }
            }
            _ => {}
        }
    }
}

pub fn access_of<P: SystemParam + 'static>(world: &mut World) -> FilteredAccessSet {
    let mut reader = IntoSystem::into_system(reads::<P>);
    reader.initialize(world)
}

fn reads<P: SystemParam + 'static>(_: StaticSystemParam<P>) {}

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

fn run_named<O: Callable>(world: &mut World, request: Value) -> Result<Value, Error> {
    let request: O::Request = engine_settings::parse_value(request)?;
    let reply = O::answer(world, request)?;
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
