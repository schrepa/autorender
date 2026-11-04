use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use engine::bricks::BrickList;
use engine::smokes::{
    Build, FRAME_MS, Headless, LIST, SmokeEntry, SmokeFailure, SmokeRecord, SmokeSession, answered,
    engine_of, fault, idle, joined, listed, members, routed, select, slot, step_lines, without,
};
use engine::{Engine, Game};
use serde_json::{Value, json};

use crate::dev::DevSession;
use crate::{HostOptions, Offscreen};

const SIZE: (u32, u32) = (960, 540);

pub type Attach<'a, G> = &'a dyn Fn(DevSession<G>) -> DevSession<G>;

pub fn run<G: Game>(
    options: &HostOptions,
    names: &str,
    entries: &[SmokeEntry],
    bricks: &BrickList,
    build: Build<'_, G>,
    attach: Attach<'_, G>,
) -> ExitCode {
    if names == LIST {
        for entry in entries {
            println!("{}", listed(entry));
        }
        return ExitCode::SUCCESS;
    }
    let selected = match select(entries, names) {
        Ok(selected) => selected,
        Err(error) => {
            eprintln!("{}", serde_json::to_value(&error).unwrap_or(Value::Null));
            return ExitCode::from(2);
        }
    };
    let mut runner = Runner {
        options,
        bricks,
        build,
        attach,
        gpu: None,
    };
    let mut failed = false;
    for entry in selected {
        let record = runner.record(entry);
        failed |= !record.ok;
        println!("{}", record.line());
    }
    if failed {
        ExitCode::from(3)
    } else {
        ExitCode::SUCCESS
    }
}

pub fn of_game<G: Game, E: Into<engine::Error>>(
    options: &HostOptions,
    names: &str,
    bricks: &dyn Fn(BrickList) -> BrickList,
    build: &dyn Fn(&mut Engine) -> Result<G, E>,
) -> ExitCode {
    let list = bricks(BrickList::default());
    let entries = match engine::smokes::registered(&list) {
        Ok(entries) => entries,
        Err(error) => return crate::exit(Err(error)),
    };
    let build = |_: Option<&str>, engine: &mut Engine| build(engine).map_err(Into::into);
    run::<G>(options, names, &entries, &list, &build, &|session| session)
}

struct Runner<'a, G: Game> {
    options: &'a HostOptions,
    bricks: &'a BrickList,
    build: Build<'a, G>,
    attach: Attach<'a, G>,
    gpu: Option<bool>,
}

enum Backend<G: Game> {
    Gpu(Box<DevSession<G>>),
    Headless(Box<Headless<G>>),
    Closed(Value),
}

impl<'a, G: Game> Runner<'a, G> {
    fn record(&mut self, entry: &SmokeEntry) -> SmokeRecord {
        if entry.gpu() && self.gpu == Some(false) {
            return SmokeRecord::skipped(entry);
        }
        let started = Instant::now();
        let elapsed = || started.elapsed().as_secs_f64() * 1000.0;
        let mut faults = Vec::new();
        let first = match entry.restart {
            Some(restart) => {
                let bricks = self.bricks.clone();
                let mut session = match self.session(entry, bricks) {
                    Ok(Some(session)) => session,
                    Ok(None) => return SmokeRecord::skipped(entry),
                    Err(error) => {
                        return SmokeRecord::failed(entry, elapsed(), fault("session", &error));
                    }
                };
                let first = (restart.first)(&mut session);
                faults = session.faults;
                match first {
                    Ok(value) => Some((restart.then, value)),
                    Err(failure) => return SmokeRecord::of(entry, elapsed(), Err(failure), faults),
                }
            }
            None => None,
        };
        let bricks = entry.bricks(self.bricks);
        let mut session = match self.session(entry, bricks) {
            Ok(Some(session)) => session,
            Ok(None) => return SmokeRecord::skipped(entry),
            Err(error) => return SmokeRecord::failed(entry, elapsed(), fault("session", &error)),
        };
        let outcome = match first {
            Some((then, value)) => then(&mut session, value),
            None => entry.run_in(&mut session),
        };
        faults.extend(session.faults);
        SmokeRecord::of(entry, elapsed(), outcome, faults)
    }

    fn session(
        &mut self,
        entry: &SmokeEntry,
        bricks: BrickList,
    ) -> engine::Result<Option<Native<'_, 'a, G>>> {
        let backend = self.open(bricks.clone())?;
        if entry.gpu() && matches!(backend, Backend::Headless(_)) {
            return Ok(None);
        }
        Ok(Some(Native {
            runner: self,
            backend,
            bricks,
            source: None,
            files: BTreeMap::new(),
            clock: 0.0,
            loaded: false,
            faults: Vec::new(),
        }))
    }

    fn open(&mut self, bricks: BrickList) -> engine::Result<Backend<G>> {
        let build = self.build;
        if self.gpu != Some(false) {
            let size = self.options.size.unwrap_or(SIZE);
            let listed = bricks.clone();
            match Offscreen::launch(
                size,
                self.options.gpu,
                &self.options.jobs,
                move |_| listed,
                |engine| build(None, engine),
            ) {
                Ok(mut game) => {
                    self.options.settings.apply(game.host_mut())?;
                    self.gpu = Some(true);
                    let session = DevSession::new(game, self.artifacts());
                    return Ok(Backend::Gpu(Box::new((self.attach)(session))));
                }
                Err(error) if self.gpu == Some(true) => return Err(error),
                Err(error) => {
                    eprintln!("{}", json!({"event": "gpu_unavailable", "error": error}));
                }
            }
        }
        let mut engine = engine_of(bricks)?;
        let game = build(None, &mut engine)?;
        self.gpu = Some(false);
        Ok(Backend::Headless(Box::new(Headless::new(engine, game))))
    }

    fn artifacts(&self) -> PathBuf {
        self.options
            .dev_artifacts
            .clone()
            .unwrap_or_else(|| std::env::temp_dir().join("engine-smoke-captures"))
    }
}

struct Native<'r, 'a, G: Game> {
    runner: &'r mut Runner<'a, G>,
    backend: Backend<G>,
    bricks: BrickList,
    source: Option<String>,
    files: BTreeMap<String, Vec<u8>>,
    clock: f64,
    loaded: bool,
    faults: Vec<SmokeFailure>,
}

impl<G: Game> Native<'_, '_, G> {
    fn engine(&self) -> Option<&Engine> {
        match &self.backend {
            Backend::Headless(headless) => Some(&headless.engine),
            Backend::Gpu(session) => Some(session.game.host().engine()),
            Backend::Closed(_) => None,
        }
    }

    fn send(&mut self, line: &str) -> Value {
        match &mut self.backend {
            Backend::Headless(headless) => headless.request(line),
            Backend::Gpu(session) => {
                let reply = session.request(line);
                session.game.exchange_jobs();
                reply
            }
            Backend::Closed(error) => json!({"ok": false, "error": error}),
        }
    }

    fn give(&mut self, name: &str, bytes: &[u8]) {
        match &mut self.backend {
            Backend::Gpu(session) => session.file(name, bytes),
            Backend::Headless(headless) => headless.file(name, bytes),
            Backend::Closed(_) => {}
        }
    }
}

impl<G: Game> SmokeSession for Native<'_, '_, G> {
    fn request(&mut self, verb: &str, body: Value) -> Result<Value, Value> {
        match self
            .engine()
            .map(|engine| routed(engine.app.world(), verb, body))
        {
            Some(line) => answered(self.send(&line)),
            None => Err(self.send("")),
        }
    }

    fn step(&mut self, ticks: u32) {
        for line in step_lines(ticks) {
            let reply = self.send(&line);
            if reply["ok"] != true {
                self.faults.push(fault("step", &reply));
                return;
            }
        }
    }

    fn frame(&mut self) -> bool {
        let Backend::Gpu(session) = &mut self.backend else {
            return false;
        };
        self.clock += FRAME_MS;
        if let Err(error) = session.game.frame(self.clock) {
            self.faults.push(fault("frame", &error));
        }
        true
    }

    fn source(&mut self, document: &str) {
        self.source = Some(document.to_owned());
        let loaded = match &mut self.backend {
            Backend::Gpu(session) => session.play(document),
            Backend::Headless(headless) => headless.source(document),
            Backend::Closed(_) => return,
        };
        match loaded {
            Ok(()) => self.loaded = true,
            Err(_) if self.loaded => {}
            Err(error) => self.faults.push(fault("source", &error)),
        }
    }

    fn idle(&mut self, milliseconds: f64) {
        idle(milliseconds);
    }

    fn joined(&mut self, list: TypeId, each: &mut dyn FnMut(&dyn Any)) {
        if let Some(engine) = self.engine() {
            joined(engine.app.world(), list, each);
        }
    }

    fn members(&mut self, place: &str) -> Vec<&'static str> {
        self.engine()
            .map(|engine| members(engine.app.world(), place))
            .unwrap_or_default()
    }

    fn slot(&mut self, filled: TypeId, each: &mut dyn FnMut(&dyn Any)) {
        if let Some(engine) = self.engine() {
            slot(engine.app.world(), filled, each);
        }
    }

    fn without(&mut self, bricks: &[&'static str]) {
        let listed = without(&self.bricks, bricks);
        self.backend = Backend::Closed(Value::Null);
        self.backend = match self.runner.open(listed) {
            Ok(backend) => backend,
            Err(error) => Backend::Closed(serde_json::to_value(&error).unwrap_or(Value::Null)),
        };
        self.clock = 0.0;
        for (name, bytes) in self.files.clone() {
            self.give(&name, &bytes);
        }
        if let Some(document) = self.source.clone() {
            self.source(&document);
        }
    }

    fn file(&mut self, name: &str, bytes: &[u8]) {
        self.files.insert(name.to_owned(), bytes.to_vec());
        self.give(name, bytes);
    }

    fn artifact(&mut self, path: &str) -> Option<Vec<u8>> {
        std::fs::read(path).ok()
    }

    fn decode_image(&mut self, path: &str, bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), Value> {
        engine_worker::decode_image(path, bytes)
            .map(|image| (image.width, image.height, image.pixels))
            .map_err(|error| serde_json::to_value(&error).unwrap_or(Value::Null))
    }
}
