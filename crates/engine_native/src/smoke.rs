use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use engine::smoke::{SmokeEntry, SmokeFailure, SmokeSession};
use engine::smokes::{
    Build, FRAME_MS, Headless, LIST, SmokeRecord, answered, fault, listed, request_line, select,
    step_lines,
};
use engine::{App, Game};
use serde_json::Value;

use crate::dev::DevSession;
use crate::{HostOptions, Offscreen};

const SIZE: (u32, u32) = (960, 540);

pub type Attach<'a, G> = &'a dyn Fn(DevSession<G>) -> DevSession<G>;

pub fn run<G: Game>(
    options: &HostOptions,
    names: &str,
    entries: &[SmokeEntry],
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
    build: Build<'a, G>,
    attach: Attach<'a, G>,
    gpu: Option<bool>,
}

enum Backend<G: Game> {
    Gpu(Box<DevSession<G>>),
    Headless(Box<Headless<G>>),
}

impl<'a, G: Game> Runner<'a, G> {
    fn record(&mut self, entry: &SmokeEntry) -> SmokeRecord {
        if entry.gpu && self.gpu == Some(false) {
            return SmokeRecord::skipped(entry);
        }
        let started = Instant::now();
        let elapsed = || started.elapsed().as_secs_f64() * 1000.0;
        let backend = match self.open(None, &[]) {
            Ok(backend) => backend,
            Err(error) => return SmokeRecord::failed(entry, elapsed(), fault("session", &error)),
        };
        if entry.gpu && matches!(backend, Backend::Headless(_)) {
            return SmokeRecord::skipped(entry);
        }
        let mut session = Native {
            runner: self,
            backend,
            source: None,
            clock: 0.0,
            faults: Vec::new(),
        };
        let outcome = entry.run_in(&mut session);
        let faults = session.faults;
        SmokeRecord::of(entry, elapsed(), outcome, faults)
    }

    fn open(&mut self, source: Option<&str>, without: &[&str]) -> engine::Result<Backend<G>> {
        if self.gpu != Some(false) {
            let size = self.options.size.unwrap_or(SIZE);
            let build = self.build;
            match Offscreen::launch(size, self.options.gpu, &self.options.jobs, || {
                build(source, without)
            }) {
                Ok(mut game) => {
                    self.options.settings.apply(game.host_mut());
                    self.gpu = Some(true);
                    let mut session = DevSession::new(game, self.artifacts());
                    if let Some(document) = source {
                        session.play(document);
                    }
                    return Ok(Backend::Gpu(Box::new((self.attach)(session))));
                }
                Err(error) if self.gpu == Some(true) => return Err(error),
                Err(_) => {}
            }
        }
        let (engine, game) = (self.build)(source, without)?;
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
    source: Option<String>,
    clock: f64,
    faults: Vec<SmokeFailure>,
}

impl<G: Game> Native<'_, '_, G> {
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

impl<G: Game> SmokeSession for Native<'_, '_, G> {
    fn app(&mut self) -> &mut App {
        match &mut self.backend {
            Backend::Gpu(session) => &mut session.game.host_mut().engine_mut().app,
            Backend::Headless(headless) => &mut headless.engine.app,
        }
    }

    fn request(&mut self, verb: &str, body: Value) -> Result<Value, Value> {
        answered(self.send(&request_line(verb, body)))
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

    fn gpu(&self) -> bool {
        matches!(self.backend, Backend::Gpu(_))
    }

    fn restart(&mut self, without: &[&str]) {
        match self.runner.open(self.source.as_deref(), without) {
            Ok(backend) => {
                self.backend = backend;
                self.clock = 0.0;
            }
            Err(error) => self.faults.push(fault("restart", &error)),
        }
    }

    fn source(&mut self, document: &str) {
        self.source = Some(document.to_owned());
        if let Backend::Gpu(session) = &mut self.backend {
            session.play(document);
        }
    }
}
