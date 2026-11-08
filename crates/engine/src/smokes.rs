mod harness;
mod host;
mod proofs;

use std::any::{Any, TypeId};
use std::collections::BTreeMap;

use engine_core::{
    BrickList, Engine, ErrorCode, JobExchange, JobMessage, MAX_DEV_TICKS, MISSING_RECORD,
};
pub use engine_core::{Runs, Smoke, SmokeFailure, SmokeSession};
use engine_registry::Registry;
use engine_scene::{SceneFiles, SceneInstance, SceneKinds};
use serde_json::{Map, Value, json};

use crate::{Error, Game, Inspectors, Result};

pub use harness::{GpuSession, SmokeHost, run};

pub type Build<'b, G> = &'b dyn Fn(Option<&str>, &mut Engine) -> Result<G>;

pub const ALL: &str = "all";

pub const LIST: &str = "list";

pub const FRAME_MS: f64 = 1000.0 / 60.0;

const HOST: &str = "engine";

#[derive(Clone, Copy, Debug)]
pub struct Restart {
    pub first: fn(&mut dyn SmokeSession) -> std::result::Result<Value, SmokeFailure>,
    pub then: fn(&mut dyn SmokeSession, Value) -> std::result::Result<(), SmokeFailure>,
}

#[derive(Clone, Copy, Debug)]
pub struct SmokeEntry {
    pub name: &'static str,
    pub owner: &'static str,
    pub runs: Runs,
    pub run: fn(&mut dyn SmokeSession) -> std::result::Result<(), SmokeFailure>,
    pub list: Option<fn(BrickList) -> BrickList>,
    pub restart: Option<Restart>,
}

impl SmokeEntry {
    pub fn of(owner: &'static str, smoke: &Smoke) -> Self {
        Self {
            name: smoke.name,
            owner,
            runs: smoke.runs,
            run: smoke.run,
            list: None,
            restart: None,
        }
    }

    pub fn gpu(&self) -> bool {
        self.runs == Runs::Gpu
    }

    pub fn bricks(&self, bricks: &BrickList) -> BrickList {
        match self.list {
            Some(list) => list(bricks.clone()),
            None => bricks.clone(),
        }
    }

    pub fn run_in(&self, session: &mut dyn SmokeSession) -> std::result::Result<(), SmokeFailure> {
        (self.run)(session)
    }
}

pub fn idle(milliseconds: f64) {
    let until = crate::host::now_ms() + milliseconds;
    while crate::host::now_ms() < until {
        std::hint::spin_loop();
    }
}

pub fn joined(world: &bevy_ecs::world::World, list: TypeId, each: &mut dyn FnMut(&dyn Any)) {
    if let Some(registry) = world.get_resource::<Registry>() {
        registry.each_member(world, list, each);
    }
}

pub fn members(world: &bevy_ecs::world::World, place: &str) -> Vec<&'static str> {
    world
        .get_resource::<Registry>()
        .map(|registry| registry.place_members(place))
        .unwrap_or_default()
}

pub fn slot(world: &bevy_ecs::world::World, slot: TypeId, each: &mut dyn FnMut(&dyn Any)) {
    if let Some(registry) = world.get_resource::<Registry>() {
        registry.each_slot(world, slot, each);
    }
}

pub fn without(bricks: &BrickList, packages: &[&'static str]) -> BrickList {
    packages.iter().fold(bricks.clone(), |listed, package| {
        listed.without_package(package)
    })
}

pub fn registered(bricks: &BrickList) -> Result<Vec<SmokeEntry>> {
    let mut smokes: Vec<SmokeEntry> = host::SMOKES
        .iter()
        .map(|smoke| SmokeEntry::of(HOST, smoke))
        .collect();
    let engine = engine_of(bricks.clone())?;
    let Some(registry) = engine.app.world().get_resource::<Registry>() else {
        return Ok(smokes);
    };
    smokes.extend(
        registry
            .smokes()
            .map(|(owner, smoke)| SmokeEntry::of(owner, smoke)),
    );
    if proofs::room_listed(registry) {
        smokes.push(SmokeEntry {
            list: Some(proofs::room),
            ..SmokeEntry::of(HOST, &proofs::LIT_ROOM)
        });
    }
    if proofs::outdoor_listed(registry) {
        smokes.push(SmokeEntry {
            list: Some(proofs::outdoor),
            ..SmokeEntry::of(HOST, &proofs::OUTDOOR_WITHOUT_TREES)
        });
    }
    Ok(smokes)
}

pub fn engine_of(bricks: BrickList) -> Result<Engine> {
    crate::game::listed(bricks)
}

pub fn built<G>(bricks: BrickList, build: Build<'_, G>) -> Result<(Engine, G)> {
    let mut engine = engine_of(bricks)?;
    let game = build(None, &mut engine)?;
    Ok((engine, game))
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
    json!({ "smoke": entry.name, "brick": entry.owner, "gpu": entry.gpu() })
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
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub skipped: bool,
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
            skipped: false,
        }
    }

    pub fn failed(entry: &SmokeEntry, ms: f64, failure: SmokeFailure) -> Self {
        Self::of(entry, ms, Err(failure), Vec::new())
    }

    pub fn skipped(entry: &SmokeEntry) -> Self {
        Self {
            smoke: entry.name,
            brick: entry.owner,
            ok: true,
            ms: 0.0,
            failure: None,
            faults: Vec::new(),
            skipped: true,
        }
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

fn json(value: &impl serde::Serialize) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

pub fn scene_files(
    document: &str,
    files: &BTreeMap<String, Vec<u8>>,
    kinds: &SceneKinds,
) -> Result<SceneFiles> {
    let mut load = crate::host::SceneLoad::new(document, kinds)?;
    for read in load.take_reads() {
        let Some(bytes) = files.get(&read.uri) else {
            continue;
        };
        if let Some(image) = load.read(&read.uri, bytes.clone())? {
            load.decode(&read.uri, &image)?;
        }
    }
    load.finish()
}

pub struct Headless<G: Game> {
    pub engine: Engine,
    pub game: G,
    tools: engine_dev::DevTools,
    source: Option<String>,
    files: BTreeMap<String, Vec<u8>>,
    paused: bool,
}

impl<G: Game> Headless<G> {
    pub fn new(engine: Engine, game: G) -> Self {
        let mut tools = engine_dev::DevTools::default();
        G::inspectors(&mut crate::game::Inspectors::new(&mut tools));
        Self {
            engine,
            game,
            tools,
            source: None,
            files: BTreeMap::new(),
            paused: true,
        }
    }

    pub fn file(&mut self, name: &str, bytes: &[u8]) {
        self.files.insert(name.to_owned(), bytes.to_vec());
    }

    pub fn source(&mut self, document: &str) -> Result<()> {
        self.source = Some(document.to_owned());
        let kinds = SceneKinds::of(self.engine.app.world());
        let files = scene_files(document, &self.files, &kinds)?;
        self.game.reload_scene(&mut self.engine, document, &files)?;
        self.engine.reset_clock();
        Ok(())
    }

    fn reload(
        &mut self,
        expected_revision: &str,
        revision: &str,
    ) -> std::result::Result<Value, Value> {
        let Some(source) = self.source.clone() else {
            return Err(json(
                &Error::new(ErrorCode::UnsupportedCommand, "$.name").got("reload"),
            ));
        };
        let kinds = SceneKinds::of(self.engine.app.world());
        let files = scene_files(&source, &self.files, &kinds).map_err(|error| json(&error))?;
        crate::game_host::reload_checked(
            &mut self.game,
            &mut self.engine,
            &source,
            (expected_revision, revision),
            &files,
        )
        .map_err(|error| json(&error))?;
        self.engine.reset_clock();
        Ok(json!({"reloaded": true, "saved_by_reload": false}))
    }

    pub fn request(&mut self, line: &str) -> Value {
        use engine_dev::Verb;
        let reply = match engine_dev::Envelope::parse(line) {
            Err(error) => self.refused(json(&error)),
            Ok(envelope) => match envelope.verb {
                Verb::Describe | Verb::Inspect | Verb::Query => self.execute(line),
                Verb::Control => self.control(&envelope),
                Verb::Capture | Verb::Profile | Verb::Explain | Verb::Watch | Verb::Check => {
                    let error = Error::new(ErrorCode::UnsupportedCommand, "$.verb")
                        .got(envelope.verb.name());
                    self.refused(json(&error))
                }
            },
        };
        self.run_jobs();
        reply
    }

    fn control(&mut self, envelope: &engine_dev::Envelope) -> Value {
        use engine_dev::results::{RenderResult, StepResult, result};
        use engine_dev::{BrowserControl, Control};
        let control = match self
            .tools
            .parse_control(self.engine.app.world(), envelope, self.paused)
        {
            Ok(control) => control,
            Err(error) => return self.refused(json(&error)),
        };
        let unsupported = || {
            let name = envelope.name().unwrap_or_default();
            json(&Error::new(ErrorCode::UnsupportedCommand, "$.name").got(name))
        };
        let done = match control {
            Control::Operation(call) => call
                .apply(&mut self.engine, self.game.scene())
                .map_err(|error| json(&error)),
            Control::Host(BrowserControl::Pause {}) => {
                self.paused = true;
                Ok(self.status()["result"].clone())
            }
            Control::Host(BrowserControl::Resume {}) => {
                self.paused = false;
                Ok(self.status()["result"].clone())
            }
            Control::Host(BrowserControl::Reload {
                expected_revision,
                revision,
            }) => self.reload(&expected_revision, &revision),
            Control::Host(BrowserControl::Jobs { workers: 0 }) => {
                Ok(self.status()["result"].clone())
            }
            Control::Host(BrowserControl::Step { ticks }) => match self.engine.step_fixed(ticks) {
                Ok(_) => Ok(result::step(StepResult {
                    ticks,
                    temporary: true,
                })),
                Err(_) => Err(json(
                    &Error::new(ErrorCode::InvalidStep, "$.args.ticks")
                        .got(ticks)
                        .range(1.0, f64::from(MAX_DEV_TICKS)),
                )),
            },
            Control::Host(BrowserControl::Render {}) => {
                Ok(result::render(RenderResult { rendered: false }))
            }
            Control::Host(
                BrowserControl::Present { .. }
                | BrowserControl::Environment(_)
                | BrowserControl::Camera { .. }
                | BrowserControl::Input { .. }
                | BrowserControl::DebugPanic {}
                | BrowserControl::DebugFault { .. }
                | BrowserControl::Benchmark { .. }
                | BrowserControl::Frames { .. }
                | BrowserControl::Probe { .. }
                | BrowserControl::Jobs { .. }
                | BrowserControl::DebugDeviceLoss { .. }
                | BrowserControl::Game { .. }
                | BrowserControl::DebugMissingAsset { .. }
                | BrowserControl::Parent { .. },
            ) => Err(unsupported()),
        };
        match done {
            Ok(result) => self.answered(result),
            Err(error) => self.refused(error),
        }
    }

    fn execute(&mut self, line: &str) -> Value {
        let context = engine_dev::Context {
            scene: self.game.scene(),
            ..engine_dev::Context::default()
        };
        serde_json::from_str(&self.tools.execute(&mut self.engine, context, line))
            .unwrap_or(Value::Null)
    }

    fn status(&mut self) -> Value {
        let context = engine_dev::Context {
            scene: self.game.scene(),
            ..engine_dev::Context::default()
        };
        self.tools
            .call(&mut self.engine, context, engine_dev::ToolCall::status())
    }

    fn answered(&mut self, result: Value) -> Value {
        let mut envelope = self.status();
        envelope["ok"] = Value::Bool(true);
        envelope["result"] = result;
        envelope
    }

    fn refused(&mut self, error: Value) -> Value {
        let mut envelope = self.status();
        envelope["ok"] = Value::Bool(false);
        if let Some(fields) = envelope.as_object_mut() {
            fields.remove("result");
        }
        envelope["error"] = error;
        envelope
    }

    fn run_jobs(&mut self) {
        let world = self.engine.app.world_mut();
        let Some(mut jobs) = world.get_resource_mut::<JobExchange>() else {
            return;
        };
        let mut landed = Vec::new();
        for message in jobs.take() {
            match message {
                JobMessage::Run(request) => {
                    let output = jobs.run_here(&request.kind, &request.input);
                    landed.push((request.id, output.map(std::sync::Arc::from)));
                }
                JobMessage::Fetch(fetch) => {
                    landed.push((fetch.id, Err(MISSING_RECORD.to_owned())));
                }
                JobMessage::Cancel(_) | JobMessage::Share(_) | JobMessage::Keep(..) => {}
            }
        }
        for (id, output) in landed {
            self.engine.land_job(id, output);
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub mod page {
    use std::collections::BTreeMap;

    use engine_core::BrickList;
    use engine_dev::{Effect, HostContext, Outcome, Step};
    use engine_scene::SceneKinds;
    use serde_json::Value;
    use wasm_bindgen::JsValue;
    use web_sys::HtmlCanvasElement;

    use super::{
        Build, GpuSession, SmokeEntry, SmokeHost, SmokeRecord, built, json, scene_files,
    };
    use crate::{Game, GameHost, RenderTarget, Result};

    struct Canvas {
        canvas: HtmlCanvasElement,
        size: (u32, u32),
    }

    struct Page<G: Game> {
        host: Box<GameHost<G>>,
        size: (u32, u32),
        paused: bool,
        source: Option<String>,
        files: BTreeMap<String, Vec<u8>>,
    }

    pub async fn run<G: Game>(
        canvas: HtmlCanvasElement,
        names: &str,
        limit_ms: f64,
        entries: &[SmokeEntry],
        bricks: &BrickList,
        build: Build<'_, G>,
        report: &JsValue,
    ) -> std::result::Result<bool, JsValue> {
        let report = js_sys::Function::from(report.clone());
        let size = (canvas.width().max(1), canvas.height().max(1));
        let mut refused = None;
        let passed = super::run(
            &mut Canvas { canvas, size },
            names,
            limit_ms,
            entries,
            bricks,
            build,
            &mut |record: &SmokeRecord| {
                let line = JsValue::from_str(&record.line());
                if refused.is_none()
                    && let Err(error) = report.call1(&JsValue::NULL, &line)
                {
                    refused = Some(error);
                }
            },
        )
        .await
        .map_err(engine_settings::js_error)?;
        match refused {
            Some(error) => Err(error),
            None => Ok(passed),
        }
    }

    fn prepared<G: Game>(host: &mut GameHost<G>) {
        host.set_job_workers(0);
        host.set_job_cache(false, false);
        host.set_preparation(false);
        let _ = host.set_setting(crate::DYNAMIC_RESOLUTION, Value::Bool(false));
    }

    impl<G: Game> SmokeHost<G> for Canvas {
        type Gpu = Page<G>;

        fn open(
            &mut self,
            bricks: BrickList,
            build: Build<'_, G>,
        ) -> impl Future<Output = Result<Page<G>>> {
            let target = RenderTarget::Canvas(self.canvas.clone());
            let size = self.size;
            async move {
                let mut host =
                    GameHost::launch(target, move |_| bricks, |engine| build(None, engine)).await?;
                prepared(&mut host);
                Ok(Page {
                    host: Box::new(host),
                    size,
                    paused: true,
                    source: None,
                    files: BTreeMap::new(),
                })
            }
        }

        fn rebuild(
            &mut self,
            mut gpu: Page<G>,
            bricks: BrickList,
            build: Build<'_, G>,
        ) -> Result<Page<G>> {
            let (engine, game) = built(bricks, build)?;
            gpu.host.replace_engine(engine);
            *gpu.host.game_mut() = game;
            prepared(gpu.host.as_mut());
            Ok(gpu)
        }

        fn artifact(&mut self, _path: &str) -> Option<Vec<u8>> {
            None
        }
    }

    impl<G: Game> Page<G> {
        fn context(&self) -> HostContext {
            HostContext {
                width: self.size.0,
                height: self.size.1,
                paused: self.paused,
                device_pixel_ratio: 1.0,
                user_agent: String::new(),
                source: None,
                timing: String::new(),
            }
        }

        fn carry(&mut self, effect: Effect, context: &HostContext) -> Step {
            let failed = |error: crate::Error| Outcome {
                failed: Some(json(&error).to_string()),
                ..Outcome::default()
            };
            if let (Effect::Source, Some(document)) = (&effect, self.source.clone()) {
                let kinds = SceneKinds::of(self.host.engine().app.world());
                return match scene_files(&document, &self.files, &kinds) {
                    Ok(files) => self.host.dev_reloaded(&document, Some(&files), context),
                    Err(error) => self.host.dev_resume(failed(error), context),
                };
            }
            let outcome = match effect {
                Effect::Pause { paused } => {
                    self.paused = paused;
                    Outcome::default()
                }
                Effect::Wait { .. } => Outcome {
                    now: Some(crate::host::now_ms()),
                    ..Outcome::default()
                },
                Effect::Frame { timestamp } => match self.frame(timestamp) {
                    Ok(()) => Outcome::default(),
                    Err(error) => failed(error),
                },
                Effect::Show | Effect::Stream { .. } => Outcome::default(),
                Effect::Game { .. } | Effect::Source => Outcome {
                    missing: Some(Vec::new()),
                    ..Outcome::default()
                },
                Effect::Save { name, .. } => Outcome {
                    failed: Some(name),
                    ..Outcome::default()
                },
            };
            let context = HostContext {
                paused: self.paused,
                ..context.clone()
            };
            self.host.dev_resume(outcome, &context)
        }
    }

    impl<G: Game> GpuSession for Page<G> {
        fn world(&self) -> &bevy_ecs::world::World {
            self.host.engine().app.world()
        }

        fn send(&mut self, line: &str) -> Value {
            let context = self.context();
            let mut step = self.host.dev_request(line, &context);
            let reply = loop {
                match step {
                    Step::Reply(reply) => break reply,
                    Step::Effect(effect) => step = self.carry(effect, &context),
                }
            };
            let _ = self.host.take_jobs();
            reply
        }

        fn frame(&mut self, timestamp: f64) -> Result<()> {
            let drawn = self.host.frame(timestamp, self.size.0, self.size.1);
            let _ = self.host.take_jobs();
            drawn
        }

        fn play(&mut self, document: &str) -> Result<()> {
            self.source = Some(document.to_owned());
            let kinds = SceneKinds::of(self.host.engine().app.world());
            let files = scene_files(document, &self.files, &kinds)?;
            self.host.reload_scene(document, &files)
        }

        fn file(&mut self, name: &str, bytes: &[u8]) {
            self.files.insert(name.to_owned(), bytes.to_vec());
        }
    }

}
