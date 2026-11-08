use std::any::{Any, TypeId};
use std::collections::BTreeMap;

use bevy_ecs::world::World;
use engine_core::{BrickList, ErrorCode, SmokeFailure, SmokeSession};
use serde_json::{Value, json};

use super::{
    Build, FRAME_MS, Headless, SmokeEntry, SmokeRecord, answered, built, fault, routed, select,
    step_lines,
};
use crate::{Game, Result};

pub trait GpuSession {
    fn world(&self) -> &World;
    fn send(&mut self, line: &str, late: &dyn Fn() -> Option<Value>) -> Value;
    fn frame(&mut self, timestamp: f64) -> Result<()>;
    fn play(&mut self, document: &str) -> Result<()>;
    fn file(&mut self, name: &str, bytes: &[u8]);
}

pub trait SmokeHost<G: Game> {
    type Gpu: GpuSession;
    fn open(
        &mut self,
        bricks: BrickList,
        build: Build<'_, G>,
    ) -> impl Future<Output = Result<Self::Gpu>>;
    fn rebuild(
        &mut self,
        gpu: Self::Gpu,
        bricks: BrickList,
        build: Build<'_, G>,
    ) -> Result<Self::Gpu>;
    fn artifact(&mut self, path: &str) -> Option<Vec<u8>>;
}

pub async fn run<G: Game, H: SmokeHost<G>>(
    host: &mut H,
    names: &str,
    limit_ms: f64,
    entries: &[SmokeEntry],
    bricks: &BrickList,
    build: Build<'_, G>,
    report: &mut dyn FnMut(&SmokeRecord),
) -> Result<bool> {
    let selected = select(entries, names)?;
    let mut runner = Runner {
        host,
        bricks,
        build,
        gpu: None,
        limit_ms,
        deadline: crate::host::now_ms() + limit_ms,
    };
    let mut passed = true;
    for entry in selected {
        let record = runner.record(entry).await;
        passed &= record.ok;
        report(&record);
    }
    Ok(passed)
}

struct Runner<'a, G: Game, H: SmokeHost<G>> {
    host: &'a mut H,
    bricks: &'a BrickList,
    build: Build<'a, G>,
    gpu: Option<bool>,
    limit_ms: f64,
    deadline: f64,
}

enum Backend<S, G: Game> {
    Gpu(Box<S>),
    Headless(Box<Headless<G>>),
    Closed(Value),
}

impl<'a, G: Game, H: SmokeHost<G>> Runner<'a, G, H> {
    fn late(&self) -> Option<Value> {
        (crate::host::now_ms() >= self.deadline).then(|| {
            super::json(
                &crate::Error::new(ErrorCode::CommandTimeout, "$.limit_ms")
                    .at_most(self.limit_ms as u64),
            )
        })
    }

    async fn record(&mut self, entry: &SmokeEntry) -> SmokeRecord {
        if entry.gpu() && self.gpu == Some(false) {
            return SmokeRecord::skipped(entry);
        }
        if let Some(late) = self.late() {
            return SmokeRecord::failed(entry, 0.0, fault("limit_ms", &late));
        }
        let started = crate::host::now_ms();
        let elapsed = || crate::host::now_ms() - started;
        let mut faults = Vec::new();
        let first = match entry.restart {
            Some(restart) => {
                let bricks = self.bricks.clone();
                let mut session = match self.session(entry, bricks).await {
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
        let mut session = match self.session(entry, bricks).await {
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

    async fn session(
        &mut self,
        entry: &SmokeEntry,
        bricks: BrickList,
    ) -> Result<Option<Harness<'_, 'a, G, H>>> {
        let backend = self.open(bricks.clone()).await?;
        if entry.gpu() && matches!(backend, Backend::Headless(_)) {
            return Ok(None);
        }
        Ok(Some(Harness {
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

    async fn open(&mut self, bricks: BrickList) -> Result<Backend<H::Gpu, G>> {
        if self.gpu != Some(false) {
            match self.host.open(bricks.clone(), self.build).await {
                Ok(gpu) => {
                    self.gpu = Some(true);
                    return Ok(Backend::Gpu(Box::new(gpu)));
                }
                Err(error) if self.gpu == Some(true) => return Err(error),
                Err(_) => {}
            }
        }
        let (engine, game) = built(bricks, self.build)?;
        self.gpu = Some(false);
        Ok(Backend::Headless(Box::new(Headless::new(engine, game))))
    }
}

struct Harness<'r, 'a, G: Game, H: SmokeHost<G>> {
    runner: &'r mut Runner<'a, G, H>,
    backend: Backend<H::Gpu, G>,
    bricks: BrickList,
    source: Option<String>,
    files: BTreeMap<String, Vec<u8>>,
    clock: f64,
    loaded: bool,
    faults: Vec<SmokeFailure>,
}

impl<G: Game, H: SmokeHost<G>> Harness<'_, '_, G, H> {
    fn world(&self) -> Option<&World> {
        match &self.backend {
            Backend::Gpu(gpu) => Some(gpu.world()),
            Backend::Headless(headless) => Some(headless.engine.app.world()),
            Backend::Closed(_) => None,
        }
    }

    fn close_when_late(&mut self) {
        if let Some(late) = self.runner.late() {
            self.backend = Backend::Closed(late);
        }
    }

    fn send(&mut self, line: &str) -> Value {
        self.close_when_late();
        let runner = &*self.runner;
        let reply = match &mut self.backend {
            Backend::Gpu(gpu) => gpu.send(line, &|| runner.late()),
            Backend::Headless(headless) => headless.request(line),
            Backend::Closed(error) => json!({"ok": false, "error": error}),
        };
        self.close_when_late();
        reply
    }

    fn give(&mut self, name: &str, bytes: &[u8]) {
        match &mut self.backend {
            Backend::Gpu(gpu) => gpu.file(name, bytes),
            Backend::Headless(headless) => headless.file(name, bytes),
            Backend::Closed(_) => {}
        }
    }
}

impl<G: Game, H: SmokeHost<G>> SmokeSession for Harness<'_, '_, G, H> {
    fn request(&mut self, verb: &str, body: Value) -> std::result::Result<Value, Value> {
        match self.world().map(|world| routed(world, verb, body)) {
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
        self.close_when_late();
        let Backend::Gpu(gpu) = &mut self.backend else {
            return false;
        };
        self.clock += FRAME_MS;
        if let Err(error) = gpu.frame(self.clock) {
            self.faults.push(fault("frame", &error));
        }
        true
    }

    fn source(&mut self, document: &str) {
        self.source = Some(document.to_owned());
        let loaded = match &mut self.backend {
            Backend::Gpu(gpu) => gpu.play(document),
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
        super::idle(milliseconds);
    }

    fn joined(&mut self, list: TypeId, each: &mut dyn FnMut(&dyn Any)) {
        if let Some(world) = self.world() {
            super::joined(world, list, each);
        }
    }

    fn members(&mut self, place: &str) -> Vec<&'static str> {
        self.world()
            .map(|world| super::members(world, place))
            .unwrap_or_default()
    }

    fn slot(&mut self, slot: TypeId, each: &mut dyn FnMut(&dyn Any)) {
        if let Some(world) = self.world() {
            super::slot(world, slot, each);
        }
    }

    fn without(&mut self, bricks: &[&'static str]) {
        let listed = super::without(&self.bricks, bricks);
        let build = self.runner.build;
        let rebuilt = match std::mem::replace(&mut self.backend, Backend::Closed(Value::Null)) {
            Backend::Gpu(gpu) => self
                .runner
                .host
                .rebuild(*gpu, listed, build)
                .map(|gpu| Backend::Gpu(Box::new(gpu)))
                .map_err(|error| super::json(&error)),
            Backend::Headless(_) => built(listed, build)
                .map(|(engine, game)| Backend::Headless(Box::new(Headless::new(engine, game))))
                .map_err(|error| super::json(&error)),
            Backend::Closed(error) => Err(error),
        };
        match rebuilt {
            Ok(backend) => self.backend = backend,
            Err(error) => {
                self.faults.push(fault("without", &error));
                self.backend = Backend::Closed(error);
                return;
            }
        }
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
        self.runner.host.artifact(path)
    }

    fn decode_image(
        &mut self,
        path: &str,
        bytes: &[u8],
    ) -> std::result::Result<(u32, u32, Vec<u8>), Value> {
        engine_worker::decode_image(path, bytes)
            .map(|image| (image.width, image.height, image.pixels))
            .map_err(|error| super::json(&error))
    }
}
