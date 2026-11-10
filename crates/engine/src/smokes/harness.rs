use bevy_ecs::world::World;
use engine_core::{BrickList, ErrorCode, SmokeFailure, SmokeSession};
use serde_json::{Value, json};

use super::{Build, SmokeEntry, SmokeRecord, answered, fault, routed, select, step_line};
use crate::{Game, Result};

pub trait GpuSession {
    fn world(&self) -> &World;
    fn send(&mut self, line: &str, late: &dyn Fn() -> Option<Value>) -> Value;
    fn play(&mut self, document: &str) -> Result<()>;
}

pub trait SmokeHost<G: Game> {
    type Gpu: GpuSession;
    fn open(
        &mut self,
        bricks: BrickList,
        build: Build<'_, G>,
    ) -> impl Future<Output = Result<Self::Gpu>>;
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
    limit_ms: f64,
    deadline: f64,
}

impl<G: Game, H: SmokeHost<G>> Runner<'_, G, H> {
    fn late(&self) -> Option<Value> {
        (crate::host::now_ms() >= self.deadline).then(|| {
            serde_json::to_value(
                crate::Error::new(ErrorCode::CommandTimeout, "$.limit_ms")
                    .at_most(self.limit_ms as u64),
            )
            .unwrap_or(Value::Null)
        })
    }

    async fn record(&mut self, entry: &SmokeEntry) -> SmokeRecord {
        if let Some(late) = self.late() {
            return SmokeRecord::failed(entry, 0.0, fault("limit_ms", &late));
        }
        let started = crate::host::now_ms();
        let elapsed = || crate::host::now_ms() - started;
        let gpu = match self.host.open(entry.list(self.bricks), self.build).await {
            Ok(gpu) => gpu,
            Err(error) => return SmokeRecord::failed(entry, elapsed(), fault("session", &error)),
        };
        let mut session = Harness {
            runner: self,
            gpu: Some(gpu),
            closed: Value::Null,
            faults: Vec::new(),
        };
        let outcome = (entry.run)(&mut session);
        SmokeRecord::of(entry, elapsed(), outcome, session.faults)
    }
}

struct Harness<'r, 'a, G: Game, H: SmokeHost<G>> {
    runner: &'r Runner<'a, G, H>,
    gpu: Option<H::Gpu>,
    closed: Value,
    faults: Vec<SmokeFailure>,
}

impl<G: Game, H: SmokeHost<G>> Harness<'_, '_, G, H> {
    fn close_when_late(&mut self) {
        if let Some(late) = self.runner.late() {
            self.gpu = None;
            self.closed = late;
        }
    }

    fn send(&mut self, line: &str) -> Value {
        self.close_when_late();
        let runner = self.runner;
        let reply = match &mut self.gpu {
            Some(gpu) => gpu.send(line, &|| runner.late()),
            None => json!({"ok": false, "error": self.closed}),
        };
        self.close_when_late();
        reply
    }
}

impl<G: Game, H: SmokeHost<G>> SmokeSession for Harness<'_, '_, G, H> {
    fn request(&mut self, verb: &str, body: Value) -> std::result::Result<Value, Value> {
        match self.gpu.as_ref().map(|gpu| routed(gpu.world(), verb, body)) {
            Some(line) => answered(self.send(&line)),
            None => Err(self.send("")),
        }
    }

    fn step(&mut self, ticks: u32) {
        let reply = self.send(&step_line(ticks));
        if reply["ok"] != true {
            self.faults.push(fault("step", &reply));
        }
    }

    fn source(&mut self, document: &str) {
        let Some(gpu) = &mut self.gpu else {
            return;
        };
        if let Err(error) = gpu.play(document) {
            self.faults.push(fault("source", &error));
        }
    }
}
