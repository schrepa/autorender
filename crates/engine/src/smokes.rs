mod harness;

use engine_core::{BrickList, Engine, ErrorCode};
pub use engine_core::{Smoke, SmokeFailure, SmokeSession};
use engine_registry::Registry;
use engine_scene::{SceneFiles, SceneKinds};
use serde_json::{Map, Value, json};

use crate::{Error, Result};

pub use harness::{GpuSession, SmokeHost, run};

pub type Build<'b, G> = &'b dyn Fn(&mut Engine) -> Result<G>;

pub const ALL: &str = "all";

pub const LIST: &str = "list";

#[derive(Clone, Copy, Debug)]
pub struct SmokeEntry {
    pub name: &'static str,
    pub owner: &'static str,
    pub run: fn(&mut dyn SmokeSession) -> std::result::Result<(), SmokeFailure>,
    pub without: &'static [&'static str],
}

impl SmokeEntry {
    pub fn of(owner: &'static str, smoke: &Smoke) -> Self {
        Self {
            name: smoke.name,
            owner,
            run: smoke.run,
            without: smoke.without,
        }
    }

    pub fn list(&self, bricks: &BrickList) -> BrickList {
        self.without.iter().fold(bricks.clone(), |list, package| {
            list.without_package(package)
        })
    }
}

pub fn registered(bricks: &BrickList) -> Result<Vec<SmokeEntry>> {
    let engine = crate::game::listed(bricks.clone())?;
    Ok(engine
        .app
        .world()
        .get_resource::<Registry>()
        .map(|registry| {
            registry
                .smokes()
                .map(|(owner, smoke)| SmokeEntry::of(owner, smoke))
                .collect()
        })
        .unwrap_or_default())
}

pub fn select<'a>(entries: &'a [SmokeEntry], names: &str) -> Result<Vec<&'a SmokeEntry>> {
    if names == ALL {
        return Ok(entries.iter().collect());
    }
    names
        .split(',')
        .filter(|name| !name.is_empty())
        .map(|name| {
            entries
                .iter()
                .find(|entry| entry.name == name)
                .ok_or_else(|| {
                    Error::new(ErrorCode::UnknownVariant, "$.smoke")
                        .got(name)
                        .variants(entries.iter().map(|entry| entry.name))
                })
        })
        .collect()
}

pub fn listed(entry: &SmokeEntry) -> Value {
    json!({ "smoke": entry.name, "brick": entry.owner })
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SmokeRecord {
    pub smoke: &'static str,
    pub brick: &'static str,
    pub ok: bool,
    pub ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<SmokeFailure>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub faults: Vec<SmokeFailure>,
}

impl SmokeRecord {
    pub fn of(
        entry: &SmokeEntry,
        ms: f64,
        outcome: std::result::Result<(), SmokeFailure>,
        faults: Vec<SmokeFailure>,
    ) -> Self {
        let mut faults: Vec<SmokeFailure> = faults
            .into_iter()
            .map(|fault| SmokeFailure {
                smoke: entry.name,
                ..fault
            })
            .collect();
        let failure = match outcome {
            Err(failure) => Some(SmokeFailure {
                smoke: entry.name,
                ..failure
            }),
            Ok(()) if faults.is_empty() => None,
            Ok(()) => Some(faults.remove(0)),
        };
        Self {
            smoke: entry.name,
            brick: entry.owner,
            ok: failure.is_none(),
            ms,
            failure,
            faults,
        }
    }

    pub fn failed(entry: &SmokeEntry, ms: f64, failure: SmokeFailure) -> Self {
        Self::of(entry, ms, Err(failure), Vec::new())
    }

    pub fn line(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

pub fn fault(check: &'static str, got: &impl serde::Serialize) -> SmokeFailure {
    SmokeFailure::new(check, got, &Value::Null, "")
}

pub fn request_line(verb: &str, body: Value) -> String {
    let mut fields = match body {
        Value::Object(fields) => fields,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_) => {
            Map::new()
        }
    };
    fields.insert("verb".to_owned(), Value::from(verb));
    fields.insert(
        "version".to_owned(),
        Value::from(engine_dev::PROTOCOL_VERSION),
    );
    Value::Object(fields).to_string()
}

pub fn routed(world: &bevy_ecs::world::World, verb: &str, body: Value) -> String {
    if engine_dev::Verb::named(verb).is_some() {
        return request_line(verb, body);
    }
    let control = world
        .get_resource::<Registry>()
        .and_then(|registry| registry.operation(verb))
        .is_some_and(|operation| operation.control);
    let asked = if control { "control" } else { "query" };
    request_line(asked, json!({ "name": verb, "args": body }))
}

pub fn answered(reply: Value) -> std::result::Result<Value, Value> {
    if reply["ok"] == true {
        Ok(reply)
    } else {
        Err(reply)
    }
}

pub fn step_line(ticks: u32) -> String {
    request_line(
        "control",
        json!({ "name": "step", "args": { "ticks": ticks } }),
    )
}

pub fn scene_files(document: &str, kinds: &SceneKinds) -> Result<SceneFiles> {
    crate::host::SceneLoad::new(document, kinds)?.finish()
}
