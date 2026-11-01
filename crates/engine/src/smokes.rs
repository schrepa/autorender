mod host;

use std::collections::BTreeMap;

use engine_core::{
    BrickList, Engine, ErrorCode, JobExchange, MAX_DEV_TICKS, MISSING_RECORD, smoke_eq,
};
pub use engine_core::{Runs, Smoke, SmokeFailure, SmokeSession};
use engine_registry::Registry;
use engine_scene::{SceneFiles, SceneInstance};
use serde_json::{Map, Value, json};

use crate::{Error, Game, Inspectors, Result};

pub type Build<'b, G> = &'b dyn Fn(Option<&str>, &mut Engine) -> Result<G>;

pub const ALL: &str = "all";

pub const LIST: &str = "list";

pub const FRAME_MS: f64 = 1000.0 / 60.0;

const HOST: &str = "engine";

#[derive(Clone, Copy, Debug)]
pub struct SmokeEntry {
    pub name: &'static str,
    pub owner: &'static str,
    pub runs: Runs,
    pub run: fn(&mut dyn SmokeSession) -> std::result::Result<(), SmokeFailure>,
    pub without: Option<&'static str>,
    pub list: Option<fn(BrickList) -> BrickList>,
}

impl SmokeEntry {
    pub fn of(owner: &'static str, smoke: &Smoke) -> Self {
        Self {
            name: smoke.name,
            owner,
            runs: smoke.runs,
            run: smoke.run,
            without: None,
            list: None,
        }
    }

    fn left_out(package: &'static str) -> Self {
        Self {
            name: Box::leak(format!("left_out.{package}").into_boxed_str()),
            owner: package,
            runs: Runs::Gpu,
            run: |_| Ok(()),
            without: Some(package),
            list: None,
        }
    }

    pub fn gpu(&self) -> bool {
        self.runs == Runs::Gpu
    }

    pub fn bricks(&self, bricks: &BrickList) -> BrickList {
        let listed = match self.list {
            Some(list) => list(bricks.clone()),
            None => bricks.clone(),
        };
        match self.without {
            Some(package) => listed.without_package(package),
            None => listed,
        }
    }

    pub fn run_in(&self, session: &mut dyn SmokeSession) -> std::result::Result<(), SmokeFailure> {
        match self.without {
            Some(package) => left_out(session, package),
            None => (self.run)(session),
        }
    }
}

fn left_out(
    session: &mut dyn SmokeSession,
    package: &'static str,
) -> std::result::Result<(), SmokeFailure> {
    let paused = session.request("control", json!({ "name": "pause" }));
    smoke_eq!(paused.is_ok(), true, "pauses");
    let status = session
        .request("query", json!({ "name": "status" }))
        .map(|reply| reply["result"]["features"].clone())
        .unwrap_or_default();
    let features = status.as_array().cloned().unwrap_or_default();
    smoke_eq!(features.is_empty(), false, "status_names_bricks_added");
    smoke_eq!(
        json!({ "without": package, "listed": features.contains(&json!(package)) }),
        json!({ "without": package, "listed": false }),
        "status_leaves_brick_out"
    );
    session.step(1);
    let rendered = session
        .request("control", json!({ "name": "render" }))
        .map_err(|reply| reply["error"].clone());
    smoke_eq!(
        json!({ "without": package, "error": rendered.err() }),
        json!({ "without": package, "error": null }),
        "renders_without_brick"
    );
    smoke_eq!(session.frame(), true, "frames_without_brick");
    Ok(())
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
    smokes.extend(registry.names().map(SmokeEntry::left_out));
    Ok(smokes)
}

pub fn engine_of(bricks: BrickList) -> Result<Engine> {
    crate::game::listed(bricks)
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
    const ACTIONS: &'static [(&'static str, u32)] = G::ACTIONS;

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
        let failure = outcome
            .err()
            .or_else(|| faults.into_iter().next())
            .map(|failure| SmokeFailure {
                smoke: entry.name,
                ..failure
            });
        Self {
            smoke: entry.name,
            brick: entry.owner,
            ok: failure.is_none(),
            ms,
            failure,
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

pub fn scene_files(document: &str, files: &BTreeMap<String, Vec<u8>>) -> Result<SceneFiles> {
    let mut load = crate::host::SceneLoad::new(document)?;
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
        }
    }

    pub fn file(&mut self, name: &str, bytes: &[u8]) {
        self.files.insert(name.to_owned(), bytes.to_vec());
    }

    pub fn source(&mut self, document: &str) -> Result<()> {
        self.source = Some(document.to_owned());
        let files = scene_files(document, &self.files)?;
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
        let files = scene_files(&source, &self.files).map_err(|error| json(&error))?;
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
        self.engine.complete_pending_tick();
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
        use engine_dev::{BrowserControl, Control, HostControl};
        let control = match self
            .tools
            .parse_control(self.engine.app.world(), envelope, true)
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
            Control::Host(HostControl::Loop(BrowserControl::Reload {
                expected_revision,
                revision,
            })) => self.reload(&expected_revision, &revision),
            Control::Host(HostControl::Loop(BrowserControl::Jobs { workers: 0 })) => {
                Ok(self.status()["result"].clone())
            }
            Control::Host(HostControl::Step { ticks }) => match self.engine.step_fixed(ticks) {
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
            Control::Host(HostControl::Render) => {
                Ok(result::render(RenderResult { rendered: false }))
            }
            Control::Host(
                HostControl::Present { .. }
                | HostControl::Culling { .. }
                | HostControl::Environment(_)
                | HostControl::Camera { .. }
                | HostControl::Input { .. }
                | HostControl::DebugPanic
                | HostControl::DebugFault { .. }
                | HostControl::Loop(_),
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
        self.execute(&engine_dev::named(
            engine_dev::Verb::Query,
            "status",
            json!({}),
        ))
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
        let (mut requests, mut fetches) = (Vec::new(), Vec::new());
        let (mut shares, mut keeps, mut cancels) = (Vec::new(), Vec::new(), Vec::new());
        jobs.take_shares(&mut shares);
        jobs.take_keeps(&mut keeps);
        jobs.take_cancels(&mut cancels);
        jobs.take_requests(&mut requests);
        jobs.take_fetches(&mut fetches);
        for request in requests {
            let output = jobs.run_here(&request.kind, &request.input);
            jobs.complete(request.id, output);
        }
        for fetch in fetches {
            jobs.complete(fetch.id, Err(MISSING_RECORD.to_owned()));
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub mod page {
    use std::collections::BTreeMap;

    use engine_core::{BrickList, SmokeFailure, SmokeSession};
    use engine_dev::{Effect, HostContext, Outcome, Step};
    use serde_json::Value;
    use wasm_bindgen::JsValue;
    use web_sys::HtmlCanvasElement;

    use super::{
        Build, FRAME_MS, Headless, SmokeEntry, SmokeRecord, answered, engine_of, fault, json,
        routed, scene_files, select,
    };
    use crate::{Game, GameHost, RenderTarget, Result};

    enum Backend<G: Game> {
        Gpu(Box<GameHost<G>>),
        Headless(Box<Headless<G>>),
    }

    struct Page<G: Game> {
        backend: Backend<G>,
        source: Option<String>,
        files: BTreeMap<String, Vec<u8>>,
        size: (u32, u32),
        paused: bool,
        clock: f64,
        faults: Vec<SmokeFailure>,
    }

    pub async fn run<G: Game>(
        canvas: HtmlCanvasElement,
        names: &str,
        entries: &[SmokeEntry],
        bricks: &BrickList,
        build: Build<'_, G>,
        report: &JsValue,
    ) -> std::result::Result<bool, JsValue> {
        let report = js_sys::Function::from(report.clone());
        let size = (canvas.width().max(1), canvas.height().max(1));
        let mut gpu = None;
        let mut passed = true;
        for entry in select(entries, names).map_err(engine_settings::js_error)? {
            let record = record(entry, &canvas, size, bricks, build, &mut gpu).await;
            passed &= record.ok;
            report.call1(&JsValue::NULL, &JsValue::from_str(&record.line()))?;
        }
        Ok(passed)
    }

    async fn record<G: Game>(
        entry: &SmokeEntry,
        canvas: &HtmlCanvasElement,
        size: (u32, u32),
        bricks: &BrickList,
        build: Build<'_, G>,
        gpu: &mut Option<bool>,
    ) -> SmokeRecord {
        if entry.gpu() && *gpu == Some(false) {
            return SmokeRecord::skipped(entry);
        }
        let started = crate::host::now_ms();
        let backend = match open(canvas, entry.bricks(bricks), build, gpu).await {
            Ok(backend) => backend,
            Err(error) => {
                let ms = crate::host::now_ms() - started;
                return SmokeRecord::failed(entry, ms, fault("session", &error));
            }
        };
        if entry.gpu() && matches!(backend, Backend::Headless(_)) {
            return SmokeRecord::skipped(entry);
        }
        let mut page = Page {
            backend,
            source: None,
            files: BTreeMap::new(),
            size,
            paused: true,
            clock: 0.0,
            faults: Vec::new(),
        };
        let outcome = entry.run_in(&mut page);
        let ms = crate::host::now_ms() - started;
        SmokeRecord::of(entry, ms, outcome, page.faults)
    }

    async fn open<G: Game>(
        canvas: &HtmlCanvasElement,
        bricks: BrickList,
        build: Build<'_, G>,
        gpu: &mut Option<bool>,
    ) -> Result<Backend<G>> {
        if *gpu != Some(false) {
            let target = RenderTarget::Canvas(canvas.clone());
            let listed = bricks.clone();
            match GameHost::launch(target, move |_| listed, |engine| build(None, engine)).await {
                Ok(mut host) => {
                    prepared(&mut host);
                    *gpu = Some(true);
                    return Ok(Backend::Gpu(Box::new(host)));
                }
                Err(error) if *gpu == Some(true) => return Err(error),
                Err(_) => {}
            }
        }
        let mut engine = engine_of(bricks)?;
        let game = build(None, &mut engine)?;
        *gpu = Some(false);
        Ok(Backend::Headless(Box::new(Headless::new(engine, game))))
    }

    fn prepared<G: Game>(host: &mut GameHost<G>) {
        host.set_job_workers(0);
        host.set_job_cache(false, false);
        let _ = host.set_setting(crate::DYNAMIC_RESOLUTION, Value::Bool(false));
    }

    fn carry<G: Game>(
        host: &mut GameHost<G>,
        paused: &mut bool,
        (width, height): (u32, u32),
        effect: Effect,
    ) -> Outcome {
        match effect {
            Effect::Pause { paused: wanted } => {
                *paused = wanted;
                Outcome::default()
            }
            Effect::Wait { .. } => Outcome {
                now: Some(crate::host::now_ms()),
                ..Outcome::default()
            },
            Effect::Frame { timestamp } => {
                let drawn = host.frame(timestamp, width, height);
                let _ = host.take_jobs();
                match drawn {
                    Ok(()) => Outcome::default(),
                    Err(error) => Outcome {
                        failed: Some(json(&error).to_string()),
                        ..Outcome::default()
                    },
                }
            }
            Effect::Show | Effect::Stream { .. } => Outcome::default(),
            Effect::Game { .. } | Effect::Source => Outcome {
                missing: Some(Vec::new()),
                ..Outcome::default()
            },
            Effect::Save { name, .. } => Outcome {
                failed: Some(name),
                ..Outcome::default()
            },
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

        fn send(&mut self, line: &str) -> Value {
            let context = self.context();
            match &mut self.backend {
                Backend::Headless(headless) => headless.request(line),
                Backend::Gpu(host) => {
                    let mut step = host.dev_request(line, &context);
                    let reply = loop {
                        match step {
                            Step::Reply(reply) => break reply,
                            Step::Effect(effect) => {
                                step = match (effect, self.source.as_deref()) {
                                    (Effect::Source, Some(document)) => {
                                        match scene_files(document, &self.files) {
                                            Ok(files) => {
                                                host.dev_reloaded(document, Some(&files), &context)
                                            }
                                            Err(error) => {
                                                let outcome = Outcome {
                                                    failed: Some(json(&error).to_string()),
                                                    ..Outcome::default()
                                                };
                                                host.dev_resume(outcome, &context)
                                            }
                                        }
                                    }
                                    (effect, _) => {
                                        let outcome = carry(
                                            host.as_mut(),
                                            &mut self.paused,
                                            self.size,
                                            effect,
                                        );
                                        let context = HostContext {
                                            paused: self.paused,
                                            ..context.clone()
                                        };
                                        host.dev_resume(outcome, &context)
                                    }
                                };
                            }
                        }
                    };
                    let _ = host.take_jobs();
                    reply
                }
            }
        }
    }

    impl<G: Game> SmokeSession for Page<G> {
        fn request(&mut self, verb: &str, body: Value) -> std::result::Result<Value, Value> {
            let line = match &self.backend {
                Backend::Headless(headless) => routed(headless.engine.app.world(), verb, body),
                Backend::Gpu(host) => routed(host.engine().app.world(), verb, body),
            };
            answered(self.send(&line))
        }

        fn step(&mut self, ticks: u32) {
            for line in super::step_lines(ticks) {
                let reply = self.send(&line);
                if reply["ok"] != true {
                    self.faults.push(fault("step", &reply));
                    return;
                }
            }
        }

        fn frame(&mut self) -> bool {
            let Backend::Gpu(host) = &mut self.backend else {
                return false;
            };
            self.clock += FRAME_MS;
            if let Err(error) = host.frame(self.clock, self.size.0, self.size.1) {
                self.faults.push(fault("frame", &error));
            }
            let _ = host.take_jobs();
            true
        }

        fn source(&mut self, document: &str) {
            self.source = Some(document.to_owned());
            let loaded = match &mut self.backend {
                Backend::Headless(headless) => headless.source(document),
                Backend::Gpu(host) => scene_files(document, &self.files)
                    .and_then(|files| host.reload_scene(document, &files)),
            };
            if let Err(error) = loaded {
                self.faults.push(fault("source", &error));
            }
        }

        fn file(&mut self, name: &str, bytes: &[u8]) {
            self.files.insert(name.to_owned(), bytes.to_vec());
            if let Backend::Headless(headless) = &mut self.backend {
                headless.file(name, bytes);
            }
        }
    }
}
