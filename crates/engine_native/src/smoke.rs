use std::any::{Any, TypeId};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use engine::Game;
use engine::bricks::BrickList;
use engine::smokes::{
    Build, FRAME_MS, Headless, LIST, SmokeEntry, SmokeFailure, SmokeRecord, SmokeSession, answered,
    engine_of, fault, idle, joined, listed, routed, select, step_lines,
};
use serde_json::Value;

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
}

impl<G: Game> Runner<'_, G> {
    fn record(&mut self, entry: &SmokeEntry) -> SmokeRecord {
        if entry.gpu() && self.gpu == Some(false) {
            return SmokeRecord::skipped(entry);
        }
        let started = Instant::now();
        let elapsed = || started.elapsed().as_secs_f64() * 1000.0;
        let mut faults = Vec::new();
        let first = match entry.restart {
            Some(restart) => {
                let mut session = match self.session(entry, self.bricks.clone()) {
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
        let mut session = match self.session(entry, entry.bricks(self.bricks)) {
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
    ) -> engine::Result<Option<Native<G>>> {
        let backend = self.open(bricks)?;
        if entry.gpu() && matches!(backend, Backend::Headless(_)) {
            return Ok(None);
        }
        Ok(Some(Native {
            backend,
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
                    eprintln!(
                        "{}",
                        serde_json::json!({"event": "gpu_unavailable", "error": error})
                    );
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

struct Native<G: Game> {
    backend: Backend<G>,
    clock: f64,
    loaded: bool,
    faults: Vec<SmokeFailure>,
}

impl<G: Game> Native<G> {
    fn send(&mut self, line: &str) -> Value {
        match &mut self.backend {
            Backend::Headless(headless) => headless.request(line),
            Backend::Gpu(session) => {
                let reply = session.request(line);
                session.game.exchange_jobs();
                reply
            }
        }
    }
}

impl<G: Game> SmokeSession for Native<G> {
    fn request(&mut self, verb: &str, body: Value) -> Result<Value, Value> {
        let line = match &self.backend {
            Backend::Headless(headless) => routed(headless.engine.app.world(), verb, body),
            Backend::Gpu(session) => routed(session.game.host().engine().app.world(), verb, body),
        };
        answered(self.send(&line))
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
        let loaded = match &mut self.backend {
            Backend::Gpu(session) => session.play(document),
            Backend::Headless(headless) => headless.source(document),
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
        let world = match &self.backend {
            Backend::Headless(headless) => headless.engine.app.world(),
            Backend::Gpu(session) => session.game.host().engine().app.world(),
        };
        joined(world, list, each);
    }

    fn file(&mut self, name: &str, bytes: &[u8]) {
        match &mut self.backend {
            Backend::Gpu(session) => session.file(name, bytes),
            Backend::Headless(headless) => headless.file(name, bytes),
        }
    }
}
