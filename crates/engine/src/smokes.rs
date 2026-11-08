mod harness;

use engine_core::{BrickList, Engine, ErrorCode, MAX_DEV_TICKS};
pub use engine_core::{Smoke, SmokeFailure, SmokeSession};
use engine_registry::Registry;
use engine_scene::{SceneFiles, SceneInstance, SceneKinds};
use serde_json::{Map, Value, json};

use crate::{Error, Game, Inspectors, Result};

pub use harness::{GpuSession, SmokeHost, run};

pub type Build<'b, G> = &'b dyn Fn(Option<&str>, &mut Engine) -> Result<G>;

pub const ALL: &str = "all";

pub const LIST: &str = "list";

#[derive(Clone, Copy, Debug)]
pub struct SmokeEntry {
    pub name: &'static str,
    pub owner: &'static str,
    pub run: fn(&mut dyn SmokeSession) -> std::result::Result<(), SmokeFailure>,
}

impl SmokeEntry {
    pub fn of(owner: &'static str, smoke: &Smoke) -> Self {
        Self {
            name: smoke.name,
            owner,
            run: smoke.run,
        }
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

pub type SceneBuild<G> = fn(&str, &SceneFiles, &mut Engine) -> Result<G>;

pub enum Played<G> {
    Plain(SceneBuild<G>),
    Scene(G),
}

pub fn played<G>(
    source: Option<&str>,
    engine: &mut Engine,
    scene: SceneBuild<G>,
) -> Result<Played<G>> {
    let Some(document) = source else {
        return Ok(Played::Plain(scene));
    };
    Ok(Played::Scene(scene(
        document,
        &SceneFiles::default(),
        engine,
    )?))
}

impl<G: Game> Game for Played<G> {
    fn before_frame(&mut self, engine: &mut Engine) {
        if let Self::Scene(game) = self {
            game.before_frame(engine);
        }
    }

    fn reset_camera(&mut self, engine: &mut Engine) {
        if let Self::Scene(game) = self {
            game.reset_camera(engine);
        }
    }

    fn scene(&self) -> Option<&SceneInstance> {
        match self {
            Self::Plain(_) => None,
            Self::Scene(game) => game.scene(),
        }
    }

    fn content(&self, engine: &Engine) -> Option<Vec<u8>> {
        match self {
            Self::Plain(_) => None,
            Self::Scene(game) => game.content(engine),
        }
    }

    fn reload_scene(
        &mut self,
        engine: &mut Engine,
        source: &str,
        files: &SceneFiles,
    ) -> Result<()> {
        match self {
            Self::Plain(scene) => {
                let built = scene(source, files, engine)?;
                *self = Self::Scene(built);
                Ok(())
            }
            Self::Scene(game) => game.reload_scene(engine, source, files),
        }
    }

    fn inspectors(inspectors: &mut Inspectors) {
        G::inspectors(inspectors);
    }
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

pub fn step_lines(ticks: u32) -> impl Iterator<Item = String> {
    (0..ticks.div_ceil(MAX_DEV_TICKS)).map(move |chunk| {
        let count = (ticks - chunk * MAX_DEV_TICKS).min(MAX_DEV_TICKS);
        request_line(
            "control",
            json!({ "name": "step", "args": { "ticks": count } }),
        )
    })
}

pub fn scene_files(document: &str, kinds: &SceneKinds) -> Result<SceneFiles> {
    crate::host::SceneLoad::new(document, kinds)?.finish()
}
