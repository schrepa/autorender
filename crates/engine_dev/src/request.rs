use std::any::Any;
use std::sync::Arc;

use bevy_ecs::world::World;
use engine_core::{FixedTime, FrameTime, Readiness};
use engine_registry::{Answer, NameEntry, ReachMut, Registry};
use engine_settings::{Error, ErrorCode, Expected};
use serde::Serialize;
use serde_json::{Map, Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verb {
    Describe,
    Read,
    Set,
    Call,
}

impl Verb {
    pub const ALL: [Self; 4] = [Self::Describe, Self::Read, Self::Set, Self::Call];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Describe => "describe",
            Self::Read => "read",
            Self::Set => "set",
            Self::Call => "call",
        }
    }

    pub fn names() -> impl Iterator<Item = &'static str> {
        Self::ALL.into_iter().map(Self::name)
    }

    fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|verb| verb.name() == name)
    }
}

pub struct Request {
    pub id: Option<Value>,
    pub verb: Verb,
    pub name: String,
    pub args: Map<String, Value>,
}

pub struct Refused {
    pub id: Option<Value>,
    pub error: Error,
}

const FIELDS: [&str; 4] = ["verb", "name", "args", "id"];

impl Request {
    pub fn parse(line: &str) -> Result<Self, Refused> {
        let refused = |id: &Option<Value>, error| Refused {
            id: id.clone(),
            error,
        };
        let value: Value = engine_settings::parse_json(line).map_err(|error| refused(&None, error))?;
        let Value::Object(mut fields) = value else {
            return Err(refused(
                &None,
                Error::new(ErrorCode::InvalidType, "$").expected(Expected::Object),
            ));
        };
        let id = fields.remove("id");
        let verb = match fields.remove("verb") {
            Some(Value::String(name)) => Verb::named(&name).ok_or_else(|| {
                Error::new(ErrorCode::UnknownVariant, "$.verb")
                    .got(&name)
                    .variants(Verb::names())
            }),
            None => Err(Error::new(ErrorCode::MissingField, "$.verb")
                .expected(Expected::Present)
                .variants(Verb::names())),
            Some(other) => Err(Error::new(ErrorCode::InvalidType, "$.verb")
                .expected(Expected::String)
                .got(other)),
        }
        .map_err(|error| refused(&id, error))?;
        let name = match fields.remove("name") {
            Some(Value::String(name)) => name,
            None if verb == Verb::Describe => String::new(),
            None => {
                return Err(refused(
                    &id,
                    Error::new(ErrorCode::MissingField, "$.name").expected(Expected::Present),
                ));
            }
            Some(other) => {
                return Err(refused(
                    &id,
                    Error::new(ErrorCode::InvalidType, "$.name")
                        .expected(Expected::String)
                        .got(other),
                ));
            }
        };
        let args = match fields.remove("args") {
            None => Map::new(),
            Some(Value::Object(args)) => args,
            Some(other) => {
                return Err(refused(
                    &id,
                    Error::new(ErrorCode::InvalidType, "$.args")
                        .expected(Expected::Object)
                        .got(other),
                ));
            }
        };
        if let Some(extra) = fields.keys().next() {
            return Err(refused(
                &id,
                Error::new(ErrorCode::UnknownField, format!("$.{extra}"))
                    .expected(Expected::Absent)
                    .variants(FIELDS),
            ));
        }
        Ok(Self {
            id,
            verb,
            name,
            args,
        })
    }
}

pub enum Applied {
    Value(Value),
    Loop(Box<dyn Any + Send>),
}

pub fn apply(reach: &mut ReachMut<'_>, request: &Request) -> Result<Applied, Error> {
    if request.verb == Verb::Describe {
        return reach
            .read(|reach| crate::table::describe(reach, &request.name))
            .map(Applied::Value);
    }
    let entry = entry(reach.world, &request.name)?;
    let mut args = request.args.clone();
    let page = match request.verb {
        Verb::Read => {
            let mut page = Page::take(&mut args)?;
            if entry.pages {
                for (key, count) in [("offset", page.offset.take()), ("limit", page.limit.take())] {
                    if let Some(count) = count {
                        args.insert(key.to_owned(), json!(count));
                    }
                }
            }
            Some(page)
        }
        Verb::Describe | Verb::Set | Verb::Call => None,
    };
    let args = Value::Object(args);
    let unsupported = || {
        Error::new(ErrorCode::UnsupportedCommand, "$.verb")
            .got(request.verb.name())
            .variants(crate::table::verbs(&entry))
    };
    match request.verb {
        Verb::Describe | Verb::Read => {
            let value = reach
                .read(|reach| entry.read_value(reach, &args))
                .ok_or_else(unsupported)??;
            page.unwrap_or_default().cut(value).map(Applied::Value)
        }
        Verb::Set => {
            let mut done = entry.set_value(reach, &args).ok_or_else(unsupported)??;
            if entry.changes_frame.is_some() && engine_profile::recording() {
                let after = done.get("after").map_or_else(String::new, Value::to_string);
                engine_profile::event(
                    "changes_frame",
                    &entry.name,
                    &[("after", engine_profile::Fact::Text(&after))],
                );
            }
            if let Some(fields) = done.as_object_mut() {
                fields.insert("lands".to_owned(), json!(entry.lands));
            }
            Ok(Applied::Value(done))
        }
        Verb::Call => match entry.call_with(reach, &args).ok_or_else(unsupported)?? {
            Answer::Value(value) => Ok(Applied::Value(value)),
            Answer::Loop(plan) => Ok(Applied::Loop(plan)),
        },
    }
}

fn entry(world: &World, name: &str) -> Result<Arc<NameEntry>, Error> {
    let registry = world.get_resource::<Registry>().ok_or_else(|| {
        Error::new(ErrorCode::MissingResource, "$.name")
            .expected(Expected::Present)
            .got("Registry")
    })?;
    if let Some(entry) = registry.entry(name) {
        return Ok(Arc::clone(entry));
    }
    let names: Vec<&str> = registry.table().map(|entry| &*entry.name).collect();
    Err(crate::table::unknown(name, "$.name", &names))
}

#[derive(Default)]
struct Page {
    pointer: Option<String>,
    offset: Option<usize>,
    limit: Option<usize>,
}

impl Page {
    fn take(args: &mut Map<String, Value>) -> Result<Self, Error> {
        let count = |args: &mut Map<String, Value>, key: &str| -> Result<Option<usize>, Error> {
            match args.remove(key) {
                None => Ok(None),
                Some(value) => value
                    .as_u64()
                    .and_then(|count| usize::try_from(count).ok())
                    .map(Some)
                    .ok_or_else(|| {
                        Error::new(ErrorCode::InvalidPage, format!("$.args.{key}"))
                            .expected(Expected::Integer)
                            .got(value)
                    }),
            }
        };
        let pointer = match args.remove("pointer") {
            None => None,
            Some(Value::String(pointer)) => Some(pointer),
            Some(other) => {
                return Err(Error::new(ErrorCode::InvalidType, "$.args.pointer")
                    .expected(Expected::String)
                    .got(other));
            }
        };
        Ok(Self {
            pointer,
            offset: count(args, "offset")?,
            limit: count(args, "limit")?,
        })
    }

    fn cut(self, value: Value) -> Result<Value, Error> {
        let value = match &self.pointer {
            None => value,
            Some(pointer) => pointed(value, pointer)?,
        };
        if self.offset.is_none() && self.limit.is_none() {
            return Ok(value);
        }
        let offset = self.offset.unwrap_or(0);
        let limit = self.limit.unwrap_or(usize::MAX);
        let (total, page) = match value {
            Value::Array(items) => (
                items.len(),
                Value::Array(items.into_iter().skip(offset).take(limit).collect()),
            ),
            Value::Object(fields) => (
                fields.len(),
                Value::Object(fields.into_iter().skip(offset).take(limit).collect()),
            ),
            other @ (Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)) => {
                return Err(Error::new(ErrorCode::InvalidPage, "$.args.offset")
                    .expected(Expected::Array)
                    .got(other));
            }
        };
        Ok(json!({ "total": total, "offset": offset, "page": page }))
    }
}

fn pointed(value: Value, pointer: &str) -> Result<Value, Error> {
    if let Some(found) = value.pointer(pointer) {
        return Ok(found.clone());
    }
    let (parent, last) = pointer.rsplit_once('/').unwrap_or(("", pointer));
    let known: Vec<String> = match value.pointer(parent) {
        Some(Value::Object(fields)) => fields.keys().cloned().collect(),
        Some(Value::Array(items)) => (0..items.len()).map(|index| index.to_string()).collect(),
        Some(Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)) | None => {
            Vec::new()
        }
    };
    let known: Vec<&str> = known.iter().map(String::as_str).collect();
    Err(crate::table::unknown(last, "$.args.pointer", &known))
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct Reply {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    pub ok: bool,
    pub frame: u64,
    pub tick: u64,
    pub ready: Readiness,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Error>,
}

impl Reply {
    pub fn schema() -> Value {
        serde_json::to_value(schemars::schema_for!(Reply)).expect("schema_serializes")
    }
}

pub fn reply(id: Option<&Value>, world: &World, outcome: Result<Value, Error>) -> String {
    let (value, error) = match outcome {
        Ok(value) => (Some(value), None),
        Err(error) => (None, Some(error)),
    };
    let reply = Reply {
        id: id.cloned(),
        ok: error.is_none(),
        frame: world.resource::<FrameTime>().frame,
        tick: world.resource::<FixedTime>().tick,
        ready: world.resource::<Readiness>().clone(),
        value,
        error,
    };
    serde_json::to_string(&reply).expect("reply_serializes")
}
