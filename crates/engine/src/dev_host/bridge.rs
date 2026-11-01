use bevy_ecs::prelude::Entity;
use engine_core::{Error, ErrorCode};
use engine_dev::results::{
    CaptureResult, FramesResult, GameResult, KeptArtifact, ProbeResult, WatchResult, result,
};
use engine_dev::{
    DrawState, Effect, EntityFacts, ExplainQuestion, HostContext, Outcome, ProfileRequest, Step,
    WatchRequest, Watcher,
};
use engine_registry::BrickCosts;
use engine_scene::SceneFiles;
use engine_settings::Expected;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::host::policy::BENCHMARK_LIMIT_MS;
use crate::host::{AdapterStatus, Distribution, HostEnvironment, KNOWN_DIFFERENCES, now_ms};
use crate::{Game, GameHost};

#[derive(Debug)]
pub(crate) enum Pending {
    Paused,
    Frames {
        count: u32,
        interval: f64,
        ran: u32,
        show: bool,
    },
    Shown {
        reply: Value,
    },
    Game {
        name: String,
    },
    Reload {
        expected_revision: String,
        revision: String,
    },
    Benchmark(Box<BenchmarkRun>),
    Probe {
        frame: Value,
        waits: u32,
    },
    DeviceLoss {
        waits: u32,
    },
    Profile(Box<ProfileRun>),
    Explain {
        id: String,
        entity: Entity,
        slot: u32,
        waits: u32,
    },
    ExplainScene,
    Watch(Box<WatchRun>),
}

const LOSS_WAITS: u32 = 120;

const PROBE_WAITS: u32 = 120;

const READ_BACK_WAITS: u32 = 120;

const EXPLAIN_FRAMES: u32 = 2;

pub(crate) struct ProfileRun {
    request: ProfileRequest,
    ran: u32,
    frames: Vec<BrickCosts>,
}

impl std::fmt::Debug for ProfileRun {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProfileRun")
            .field("request", &self.request)
            .field("ran", &self.ran)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub(crate) struct WatchRun {
    watcher: Watcher,
    ran: u32,
    streaming: bool,
}

#[derive(Debug)]
pub(crate) struct BenchmarkRun {
    frame: Value,
    mode: String,
    warmup: u64,
    samples: u64,
    started: f64,
    view: (u32, u32),
    previous: Option<f64>,
    unwaited: Option<Value>,
    ran: u64,
    intervals: Vec<f64>,
    measurements: Vec<Value>,
    initial: Value,
    end: Value,
}

type Stopped = Error;

fn query(name: &str) -> Value {
    json!({ "verb": "query", "name": name })
}

fn control(name: &str, args: Value) -> Value {
    json!({ "verb": "control", "name": name, "args": args })
}

fn shown(alpha: Option<&Value>) -> Value {
    match alpha {
        Some(alpha) => control("present", json!({ "alpha": alpha })),
        None => control("render", json!({})),
    }
}

fn parsed<T: DeserializeOwned>(control: &Map<String, Value>) -> Result<T, Error> {
    let mut fields = control.clone();
    fields.remove("op");
    engine_settings::parse_value(Value::Object(fields))
}

fn failed_frame(code: ErrorCode, frame: u32, frames: u32, failed: &str) -> Error {
    Error::new(code, "$").got(json!({ "frame": frame, "frames": frames, "failed": failed }))
}

fn view(context: &HostContext, interrupted: bool) -> Value {
    json!({
        "width": context.width,
        "height": context.height,
        "paused": context.paused,
        "interrupted": interrupted,
    })
}

impl BenchmarkRun {
    fn waited(&mut self, now: f64) -> Result<bool, Stopped> {
        if now - self.started >= BENCHMARK_LIMIT_MS as f64 {
            return Err(Error::new(ErrorCode::BenchmarkTimeout, "$")
                .at_most(BENCHMARK_LIMIT_MS)
                .got(now - self.started));
        }
        if let (Some(previous), Some(stats)) = (self.previous, self.unwaited.take())
            && self.ran > self.warmup
        {
            let mut fields = Vec::new();
            numbers(&stats, "", &mut fields);
            if let Some((field, value)) = fields
                .iter()
                .find(|(_, value)| !value.is_finite() || *value < 0.0)
            {
                let mut got = Map::new();
                got.insert(field.clone(), json!(value));
                return Err(Error::new(ErrorCode::InvalidMeasurement, "$")
                    .expected(Expected::Finite)
                    .got(got));
            }
            self.intervals.push((now - previous).max(0.0));
            self.measurements.push(stats);
        }
        self.previous = Some(now);
        Ok(self.ran >= self.warmup + self.samples)
    }

    fn report(
        &self,
        now: f64,
        after: &Value,
        environment: &HostEnvironment,
        timing: &str,
    ) -> Result<Value, Stopped> {
        let stage = |name: &str| -> Vec<f64> {
            self.measurements
                .iter()
                .map(|stats| stats[name].as_f64().unwrap_or(0.0))
                .collect()
        };
        let totals: Vec<f64> = self
            .measurements
            .iter()
            .map(|stats| {
                ["simulation_ms", "extraction_ms", "render_ms"]
                    .iter()
                    .map(|name| stats[*name].as_f64().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        let sampled: Vec<Vec<(String, f64)>> = self
            .measurements
            .iter()
            .map(|stats| {
                let mut fields = Vec::new();
                numbers(stats, "", &mut fields);
                fields
            })
            .collect();
        let mut ranges = Map::new();
        for (path, _) in sampled.first().into_iter().flatten() {
            let values: Option<Vec<f64>> = sampled
                .iter()
                .map(|fields| {
                    fields
                        .iter()
                        .find(|(field, _)| field == path)
                        .map(|(_, value)| *value)
                })
                .collect();
            if let Some(values) = values {
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                ranges.insert(path.clone(), json!({"min": min, "max": max}));
            }
        }
        Ok(json!({
            "version": 1, "mode": self.mode, "warmup": self.warmup, "samples": self.samples,
            "elapsed_ms": now - self.started,
            "frame_interval_ms": distribution(&self.intervals)?,
            "cpu_ms": {
                "simulation": distribution(&stage("simulation_ms"))?,
                "extraction": distribution(&stage("extraction_ms"))?,
                "render": distribution(&stage("render_ms"))?,
                "total": distribution(&totals)?,
            },
            "rendering": {"first": self.measurements.first(), "last": self.measurements.last(), "ranges": ranges},
            "observation": {"start": self.initial["observation"], "end": self.end},
            "assets": {"before": self.initial["result"], "after": after["result"]},
            "environment": environment,
            "timing": timing,
        }))
    }
}

impl<G: Game> GameHost<G> {
    pub fn dev_request(&mut self, line: &str, context: &HostContext) -> Step {
        self.dev_pending = None;
        if line.len() > engine_dev::MAX_REQUEST_BYTES {
            return self.dev_refused(
                Error::new(ErrorCode::RequestTooLarge, "$.command")
                    .at_most(engine_dev::MAX_REQUEST_BYTES as u64)
                    .got(line.len()),
            );
        }
        let command = match serde_json::from_str::<Value>(line) {
            Ok(command @ Value::Object(_)) if command["verb"].is_string() => command,
            _ => {
                return self.dev_refused(
                    Error::new(ErrorCode::InvalidRequest, "$.command").expected(Expected::Object),
                );
            }
        };
        let reply = self.dev_invoke(&command, context);
        let control = (reply["ok"] == true)
            .then(|| reply["result"]["control"].as_object().cloned())
            .flatten();
        match control {
            Some(control) => self.dev_control(&control, context),
            None => self.dev_done(reply, context),
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub fn dev_resume(&mut self, outcome: Outcome, context: &HostContext) -> Step {
        let Some(pending) = self.dev_pending.take() else {
            return self.dev_refused(
                Error::new(ErrorCode::InvalidRequest, "$").expected(Expected::Running),
            );
        };
        match pending {
            Pending::Paused => {
                let status = self.dev_invoke(&query("status"), context);
                self.dev_done(status, context)
            }
            Pending::Frames {
                count,
                interval,
                ran,
                show,
            } => {
                if let Some(failed) = outcome.failed {
                    return self.dev_refused(failed_frame(
                        ErrorCode::RenderFailed,
                        ran + 1,
                        count,
                        &failed,
                    ));
                }
                let ran = ran + 1;
                if ran < count {
                    let pending = Pending::Frames {
                        count,
                        interval,
                        ran,
                        show,
                    };
                    let timestamp = f64::from(ran) * interval;
                    return self.dev_effect(pending, Effect::Frame { timestamp });
                }
                let mut reply = self.dev_invoke(&query("status"), context);
                reply["result"] = result::frames(FramesResult {
                    frames: count,
                    interval_ms: interval,
                    temporary: true,
                    artifact: None,
                });
                if show {
                    return self.dev_effect(Pending::Shown { reply }, Effect::Show);
                }
                self.dev_done(reply, context)
            }
            Pending::Shown { mut reply } => {
                if let Some(failed) = outcome.failed {
                    return self.dev_refused(Error::new(ErrorCode::CaptureFailed, "$").got(failed));
                }
                if let Some(artifact) = outcome.artifact {
                    let observation = &reply["observation"];
                    let kept = KeptArtifact {
                        path: artifact.path,
                        mime_type: artifact.mime_type,
                        bytes: artifact.bytes,
                        sha256: artifact.sha256,
                        frame: observation["frame"].as_str().map(str::to_owned),
                        tick: observation["tick"].as_str().map(str::to_owned),
                    };
                    reply["result"]["artifact"] =
                        serde_json::to_value(kept).expect("result_serializes");
                }
                self.dev_done(reply, context)
            }
            Pending::Game { name } => {
                if let Some(names) = outcome.missing {
                    return self.dev_refused(
                        Error::new(ErrorCode::UnsupportedCommand, "$.args.name")
                            .got(&name)
                            .variants(names),
                    );
                }
                if let Some(failed) = outcome.failed {
                    return self
                        .dev_refused(Error::new(ErrorCode::GameRule, "$.args.args").got(failed));
                }
                let mut rendered = self.dev_invoke(&control("render", json!({})), context);
                if rendered["ok"] == true {
                    rendered["result"] = result::game(GameResult {
                        game: name,
                        result: outcome.result.unwrap_or(Value::Null),
                    });
                }
                self.dev_done(rendered, context)
            }
            Pending::Reload { .. } => {
                if outcome.missing.is_some() {
                    return self.dev_refused(
                        Error::new(ErrorCode::UnsupportedCommand, "$.name").got("reload"),
                    );
                }
                let refused = Error::new(ErrorCode::SourceIo, "$");
                self.dev_refused(match outcome.failed {
                    Some(failed) => refused.got(failed),
                    None => refused.expected(Expected::Present),
                })
            }
            Pending::Benchmark(run) => self.dev_benchmark(run, &outcome, context),
            Pending::Probe { frame, waits } => {
                if let Some(failed) = outcome.failed {
                    return self.dev_refused(Error::new(ErrorCode::RenderFailed, "$").got(failed));
                }
                let reply = self.dev_invoke(&frame, context);
                if reply["ok"] != true {
                    return self.dev_done(reply, context);
                }
                self.dev_probe(frame, waits, context)
            }
            Pending::DeviceLoss { waits } => self.dev_device_lost(waits, context),
            Pending::Profile(run) => self.dev_profile(run, &outcome, context),
            Pending::Explain {
                id,
                entity,
                slot,
                waits,
            } => {
                if let Some(failed) = outcome.failed {
                    return self.dev_refused(Error::new(ErrorCode::RenderFailed, "$").got(failed));
                }
                self.dev_explain_wait((id, entity), slot, waits, context)
            }
            Pending::ExplainScene => {
                if outcome.missing.is_some() {
                    return self.dev_refused(
                        Error::new(ErrorCode::UnsupportedCommand, "$.about").got("scene"),
                    );
                }
                let refused = Error::new(ErrorCode::SourceIo, "$");
                self.dev_refused(match outcome.failed {
                    Some(failed) => refused.got(failed),
                    None => refused.expected(Expected::Present),
                })
            }
            Pending::Watch(run) => self.dev_watch(run, &outcome, context),
        }
    }

    fn dev_device_lost(&mut self, waits: u32, context: &HostContext) -> Step {
        let reply = self.dev_invoke(&control("render", json!({})), context);
        if reply["ok"] != true {
            return self.dev_done(reply, context);
        }
        if waits >= LOSS_WAITS {
            return self.dev_refused(
                Error::new(ErrorCode::RenderFailed, "$")
                    .at_most(u64::from(LOSS_WAITS))
                    .got(waits),
            );
        }
        self.dev_effect(
            Pending::DeviceLoss { waits: waits + 1 },
            Effect::Wait { within_ms: 1000.0 },
        )
    }

    fn dev_probe_start(&mut self, control: &Map<String, Value>, context: &HostContext) -> Step {
        let pixel = |field: &str| control.get(field).cloned().unwrap_or(Value::Null);
        let armed = self.dev_invoke(
            &self::control("probe_pixel", json!({ "x": pixel("x"), "y": pixel("y") })),
            context,
        );
        if armed["ok"] != true {
            return self.dev_done(armed, context);
        }
        let frame = shown(control.get("alpha"));
        for _ in 0..2 {
            let reply = self.dev_invoke(&frame, context);
            if reply["ok"] != true {
                return self.dev_done(reply, context);
            }
        }
        self.dev_probe(frame, 0, context)
    }

    fn dev_probe(&mut self, frame: Value, waits: u32, context: &HostContext) -> Step {
        let mut probed = self.dev_invoke(&query("probed_pixel"), context);
        if probed["ok"] != true {
            return self.dev_done(probed, context);
        }
        if probed["result"]["probing"] != true {
            let probe = match probed["result"]["probe"].take() {
                Value::Object(probe) => probe,
                Value::Null
                | Value::Bool(_)
                | Value::Number(_)
                | Value::String(_)
                | Value::Array(_) => {
                    return self.dev_refused(
                        Error::new(ErrorCode::RenderFailed, "$.probe").expected(Expected::Present),
                    );
                }
            };
            probed["result"] = result::probe(ProbeResult {
                probe,
                entity: None,
            });
            return self.dev_done(probed, context);
        }
        if waits >= PROBE_WAITS {
            return self.dev_refused(
                Error::new(ErrorCode::RenderFailed, "$")
                    .at_most(u64::from(PROBE_WAITS))
                    .got(waits),
            );
        }
        self.dev_effect(
            Pending::Probe {
                frame,
                waits: waits + 1,
            },
            Effect::Wait { within_ms: 1000.0 },
        )
    }

    pub fn dev_reloaded(
        &mut self,
        source: &str,
        files: Option<&SceneFiles>,
        context: &HostContext,
    ) -> Step {
        match self.dev_pending.take() {
            Some(Pending::Reload {
                expected_revision,
                revision,
            }) => {
                let reply = self.dev_reload(source, &expected_revision, &revision, files);
                let mut reply: Value = serde_json::from_str(&reply).expect("reply_json");
                if reply["ok"] == true {
                    let rendered = self.dev_invoke(&control("render", json!({})), context);
                    if rendered["ok"] != true {
                        return self.dev_done(rendered, context);
                    }
                    reply["observation"] = rendered["observation"].clone();
                }
                self.dev_done(reply, context)
            }
            Some(Pending::ExplainScene) => {
                let explained =
                    self.dev_tools()
                        .explain_scene(self.engine().app.world(), source, files);
                let mut reply = self.dev_invoke(&query("status"), context);
                if reply["ok"] == true {
                    reply["result"] = result::explain(explained);
                }
                self.dev_done(reply, context)
            }
            Some(_) | None => self.dev_refused(
                Error::new(ErrorCode::InvalidRequest, "$").expected(Expected::Running),
            ),
        }
    }

    pub fn dev_refusal(&mut self, error: &Error) -> Value {
        let status = engine_dev::named(engine_dev::Verb::Query, "status", json!({}));
        let status: Value =
            serde_json::from_str(&self.dev_command(&status, 1, 1, true)).expect("reply_json");
        json!({"ok": false, "observation": status["observation"], "error": error})
    }

    fn dev_invoke(&mut self, command: &Value, context: &HostContext) -> Value {
        let mut command = command.clone();
        command["version"] = json!(engine_dev::PROTOCOL_VERSION);
        let reply = self.dev_command(
            &command.to_string(),
            context.width,
            context.height,
            context.paused,
        );
        serde_json::from_str(&reply).expect("reply_json")
    }

    fn dev_control(&mut self, control: &Map<String, Value>, context: &HostContext) -> Step {
        let op = control
            .get("op")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let text = |field: &str| {
            control
                .get(field)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        match op {
            "pause" | "resume" => self.dev_effect(
                Pending::Paused,
                Effect::Pause {
                    paused: op == "pause",
                },
            ),
            "frames" => {
                let count = control
                    .get("count")
                    .and_then(Value::as_u64)
                    .and_then(|count| u32::try_from(count).ok())
                    .unwrap_or(1);
                let interval = control
                    .get("interval_ms")
                    .and_then(Value::as_f64)
                    .unwrap_or_default();
                let show = control.get("capture") == Some(&json!(true));
                self.suspend();
                let pending = Pending::Frames {
                    count,
                    interval,
                    ran: 0,
                    show,
                };
                self.dev_effect(pending, Effect::Frame { timestamp: 0.0 })
            }
            "capture" => self.dev_capture(control, context),
            "probe" => self.dev_probe_start(control, context),
            "jobs" => {
                let workers = control
                    .get("workers")
                    .and_then(Value::as_u64)
                    .map_or(u32::MAX, |workers| {
                        u32::try_from(workers).unwrap_or(u32::MAX)
                    });
                let working = self.job_workers();
                if workers > working {
                    return self.dev_refused(
                        Error::new(ErrorCode::OutOfRange, "$.args.workers")
                            .range(0.0, f64::from(working))
                            .got(workers),
                    );
                }
                self.set_job_workers(workers);
                let status = self.dev_invoke(&query("status"), context);
                self.dev_done(status, context)
            }
            "debug_device_loss" => {
                self.lose_device();
                self.dev_effect(
                    Pending::DeviceLoss { waits: 0 },
                    Effect::Wait { within_ms: 1000.0 },
                )
            }
            "game" => {
                let name = text("name");
                let args = control
                    .get("args")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                let pending = Pending::Game { name: name.clone() };
                self.dev_effect(pending, Effect::Game { name, args })
            }
            "reload" => {
                let pending = Pending::Reload {
                    expected_revision: text("expected_revision"),
                    revision: text("revision"),
                };
                self.dev_effect(pending, Effect::Source)
            }
            "benchmark" => self.dev_benchmark_start(control, context),
            "profile" => match parsed::<ProfileRequest>(control) {
                Ok(request) => {
                    self.suspend();
                    let run = ProfileRun {
                        request,
                        ran: 0,
                        frames: Vec::new(),
                    };
                    self.dev_effect(
                        Pending::Profile(Box::new(run)),
                        Effect::Frame { timestamp: 0.0 },
                    )
                }
                Err(error) => self.dev_refused(error),
            },
            "explain" => match parsed::<ExplainQuestion>(control) {
                Ok(question) => self.dev_explain_start(question, context),
                Err(error) => self.dev_refused(error),
            },
            "watch" => match parsed::<WatchRequest>(control) {
                Ok(request) => self.dev_watch_start(request, context),
                Err(error) => self.dev_refused(error),
            },
            _ => self.dev_refused(Error::new(ErrorCode::UnsupportedCommand, "$.name").got(op)),
        }
    }

    fn dev_capture(&mut self, control: &Map<String, Value>, context: &HostContext) -> Step {
        let mut reply = self.dev_invoke(&shown(control.get("alpha")), context);
        if reply["ok"] != true {
            return self.dev_done(reply, context);
        }
        let digest = if control.get("digest") == Some(&json!(true)) {
            let digested = self.dev_invoke(&query("digest"), context);
            if digested["ok"] != true {
                return self.dev_done(digested, context);
            }
            Some(digested["result"].clone())
        } else {
            None
        };
        reply["result"] = result::capture(CaptureResult {
            temporary: control.get("alpha").map(|_| true),
            rendered: true,
            artifact: None,
            digest: None,
        });
        if let Some(digest) = digest {
            reply["result"]["digest"] = digest;
        }
        self.dev_effect(Pending::Shown { reply }, Effect::Show)
    }

    fn dev_profile(
        &mut self,
        mut run: Box<ProfileRun>,
        outcome: &Outcome,
        context: &HostContext,
    ) -> Step {
        if let Some(failed) = &outcome.failed {
            return self.dev_refused(failed_frame(
                ErrorCode::RenderFailed,
                run.ran + 1,
                run.request.frames,
                failed,
            ));
        }
        let Some(costs) = self
            .engine()
            .app
            .world()
            .get_resource::<BrickCosts>()
            .cloned()
        else {
            return self
                .dev_refused(Error::new(ErrorCode::ProfilingUnavailable, "$").got("BrickCosts"));
        };
        run.frames.push(costs);
        run.ran += 1;
        if run.ran < run.request.frames {
            let timestamp = f64::from(run.ran) * run.request.interval_ms;
            return self.dev_effect(Pending::Profile(run), Effect::Frame { timestamp });
        }
        let (profiled, contents) = engine_dev::profiled(&run.frames, &run.request);
        let mut reply = self.dev_invoke(&query("status"), context);
        if reply["ok"] != true {
            return self.dev_done(reply, context);
        }
        reply["result"] = result::profile(profiled);
        self.dev_effect(
            Pending::Shown { reply },
            Effect::Save {
                name: "profile".to_owned(),
                contents,
            },
        )
    }

    fn dev_explain_start(&mut self, question: ExplainQuestion, context: &HostContext) -> Step {
        match question {
            ExplainQuestion::Frame { top, detail } => {
                let stats = self.stats_value();
                let explained =
                    self.dev_tools()
                        .explain_frame(self.engine().app.world(), &stats, top, detail);
                match explained {
                    Ok(explained) => {
                        let mut reply = self.dev_invoke(&query("status"), context);
                        if reply["ok"] == true {
                            reply["result"] = result::explain(explained);
                        }
                        self.dev_done(reply, context)
                    }
                    Err(error) => self.dev_refused(error),
                }
            }
            ExplainQuestion::Scene {} => self.dev_effect(Pending::ExplainScene, Effect::Source),
            ExplainQuestion::Entity { entity: id, .. } => {
                let entity = match self.dev_tools().explained_entity(
                    self.engine().app.world(),
                    self.dev_scene(),
                    &id,
                ) {
                    Ok(entity) => entity,
                    Err(error) => return self.dev_refused(error),
                };
                let readable = self
                    .engine()
                    .app
                    .world()
                    .get_resource::<engine_core::Detail>()
                    .is_some_and(|detail| detail.enabled && detail.fade_seconds > 0.0);
                let Some(slot) = self.engine().render_slot(entity).filter(|_| readable) else {
                    let slot = self.engine().render_slot(entity);
                    return self.dev_explained((id, entity), slot, DrawState::Unread, context);
                };
                self.read_back(slot);
                self.suspend();
                self.dev_effect(
                    Pending::Explain {
                        id,
                        entity,
                        slot,
                        waits: 1,
                    },
                    Effect::Frame { timestamp: 0.0 },
                )
            }
        }
    }

    fn dev_explain_wait(
        &mut self,
        (id, entity): (String, Entity),
        slot: u32,
        waits: u32,
        context: &HostContext,
    ) -> Step {
        match self.read_back(slot) {
            Some((level, 0)) if waits >= EXPLAIN_FRAMES => {
                let drawn = level.map_or(DrawState::Culled, DrawState::Level);
                self.dev_explained((id, entity), Some(slot), drawn, context)
            }
            Some(_) | None if waits >= READ_BACK_WAITS => {
                self.dev_explained((id, entity), Some(slot), DrawState::Unread, context)
            }
            Some(_) | None => self.dev_effect(
                Pending::Explain {
                    id,
                    entity,
                    slot,
                    waits: waits + 1,
                },
                Effect::Frame { timestamp: 0.0 },
            ),
        }
    }

    fn dev_explained(
        &mut self,
        (id, entity): (String, Entity),
        slot: Option<u32>,
        drawn: DrawState,
        context: &HostContext,
    ) -> Step {
        let stats = self.stats_value();
        let facts = EntityFacts {
            slot,
            drawn,
            frustum_culling: engine_dev::frustum_culling(self.engine().app.world()),
        };
        let explained = self.dev_tools().explain_entity(
            self.engine().app.world(),
            (&id, entity),
            facts,
            &stats,
        );
        let mut reply = self.dev_invoke(&query("status"), context);
        if reply["ok"] == true {
            reply["result"] = result::explain(explained);
        }
        self.dev_done(reply, context)
    }

    fn dev_watch_start(&mut self, request: WatchRequest, context: &HostContext) -> Step {
        let mut run = Box::new(WatchRun {
            watcher: Watcher::new(request),
            ran: 0,
            streaming: true,
        });
        let records = match self.dev_watch_read(&mut run.watcher, context) {
            Ok(records) => records,
            Err(refused) => return self.dev_done(refused, context),
        };
        self.suspend();
        self.dev_effect(Pending::Watch(run), Effect::Stream { records })
    }

    fn dev_watch(
        &mut self,
        mut run: Box<WatchRun>,
        outcome: &Outcome,
        context: &HostContext,
    ) -> Step {
        if let Some(failed) = &outcome.failed {
            return self.dev_refused(failed_frame(
                ErrorCode::RenderFailed,
                run.ran + 1,
                run.watcher.frames(),
                failed,
            ));
        }
        if !run.streaming {
            run.ran += 1;
            let records = match self.dev_watch_read(&mut run.watcher, context) {
                Ok(records) => records,
                Err(refused) => return self.dev_done(refused, context),
            };
            if !records.is_empty() {
                run.streaming = true;
                return self.dev_effect(Pending::Watch(run), Effect::Stream { records });
            }
        }
        if run.ran < run.watcher.frames() {
            run.streaming = false;
            let timestamp = f64::from(run.ran) * run.watcher.interval_ms();
            return self.dev_effect(Pending::Watch(run), Effect::Frame { timestamp });
        }
        let mut reply = self.dev_invoke(&query("status"), context);
        if reply["ok"] == true {
            reply["result"] = result::watch(WatchResult {
                frames: run.ran,
                interval_ms: run.watcher.interval_ms(),
                records: run.watcher.sent(),
                temporary: true,
            });
        }
        self.dev_done(reply, context)
    }

    fn dev_watch_read(
        &mut self,
        watcher: &mut Watcher,
        context: &HostContext,
    ) -> Result<Vec<Value>, Value> {
        let mut components = Vec::new();
        for (index, watched) in watcher.watched().to_vec().into_iter().enumerate() {
            let mut reply = self.dev_invoke(
                &json!({
                    "verb": "inspect",
                    "entity": watched.entity,
                    "component": watched.component,
                }),
                context,
            );
            if reply["ok"] != true {
                if let Some(path) = reply["error"]["path"].as_str() {
                    let field = path.strip_prefix("$.").unwrap_or_default().to_owned();
                    reply["error"]["path"] = json!(format!("$.entities[{index}].{field}"));
                }
                return Err(reply);
            }
            components.push(reply["result"]["entity"]["components"][&watched.component].clone());
        }
        let world = self.engine().app.world();
        let frame = world
            .get_resource::<engine_core::FrameTime>()
            .map(|time| time.frame.to_string());
        let tick = world
            .get_resource::<engine_core::FixedTime>()
            .map(|time| time.tick.to_string());
        let stats = self.stats_value();
        Ok(watcher.changes((frame, tick), &stats, components))
    }

    fn stats_value(&self) -> Value {
        serde_json::from_str(&self.stats()).expect("stats_json")
    }

    fn dev_effect(&mut self, pending: Pending, effect: Effect) -> Step {
        self.dev_pending = Some(pending);
        Step::Effect(effect)
    }

    fn dev_done(&mut self, mut reply: Value, context: &HostContext) -> Step {
        self.dev_pending = None;
        if reply["ok"] == true {
            let rendering = self.stats_value();
            let observation = reply["observation"].clone();
            let mut result = reply["result"]
                .as_object_mut()
                .map(std::mem::take)
                .unwrap_or_default();
            result.insert("observation".into(), observation);
            result.insert("paused".into(), json!(context.paused));
            result.insert("source".into(), json!(context.source));
            result.insert("rendering".into(), rendering);
            reply["result"] = Value::Object(result);
        }
        Step::Reply(reply)
    }

    fn dev_refused(&mut self, error: Error) -> Step {
        self.dev_pending = None;
        Step::Reply(self.dev_refusal(&error))
    }

    fn dev_benchmark_start(&mut self, control: &Map<String, Value>, context: &HostContext) -> Step {
        if context.width == 0 || context.height == 0 {
            return self.dev_refused(
                Error::new(ErrorCode::BenchmarkInterrupted, "$").got(view(context, false)),
            );
        }
        let started = now_ms();
        let mode = control
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let frame = match mode.as_str() {
            "step" => self::control("step", json!({ "ticks": 1 })),
            "present" => self::control("present", json!({ "alpha": 0.5 })),
            _ => self::control("render", json!({})),
        };
        let number = |field: &str| control.get(field).and_then(Value::as_u64).unwrap_or(0);
        let initial = self.dev_invoke(&query("assets"), context);
        let cleared = self.dev_invoke(&self::control("input", json!({})), context);
        if let Some(refused) = [&initial, &cleared]
            .into_iter()
            .find(|reply| reply["ok"] != true)
        {
            return self.dev_refused(stopped(refused));
        }
        let run = BenchmarkRun {
            frame,
            mode,
            warmup: number("warmup"),
            samples: number("samples"),
            started,
            view: (context.width, context.height),
            previous: None,
            unwaited: None,
            ran: 0,
            intervals: Vec::new(),
            measurements: Vec::new(),
            initial,
            end: Value::Null,
        };
        let within_ms = BENCHMARK_LIMIT_MS as f64;
        self.dev_effect(
            Pending::Benchmark(Box::new(run)),
            Effect::Wait { within_ms },
        )
    }

    fn dev_benchmark(
        &mut self,
        mut run: Box<BenchmarkRun>,
        outcome: &Outcome,
        context: &HostContext,
    ) -> Step {
        if let Some(failed) = &outcome.failed {
            return self.dev_refused(Error::new(ErrorCode::BenchmarkFailed, "$").got(failed));
        }
        if outcome.interrupted || !context.paused || (context.width, context.height) != run.view {
            return self.dev_refused(
                Error::new(ErrorCode::BenchmarkInterrupted, "$")
                    .got(view(context, outcome.interrupted)),
            );
        }
        let done = match run.waited(outcome.now.unwrap_or_else(now_ms)) {
            Ok(done) => done,
            Err(error) => return self.dev_refused(error),
        };
        if done {
            return self.dev_benchmark_report(&run, context);
        }
        let reply = self.dev_invoke(&run.frame, context);
        if reply["ok"] != true {
            return self.dev_refused(stopped(&reply));
        }
        run.end = reply["observation"].clone();
        run.unwaited = Some(self.stats_value());
        run.ran += 1;
        let within_ms = (BENCHMARK_LIMIT_MS as f64 - (now_ms() - run.started)).max(1.0);
        self.dev_effect(Pending::Benchmark(run), Effect::Wait { within_ms })
    }

    fn dev_benchmark_report(&mut self, run: &BenchmarkRun, context: &HostContext) -> Step {
        let after = self.dev_invoke(&query("assets"), context);
        if after["ok"] != true {
            return self.dev_refused(stopped(&after));
        }
        let environment = HostEnvironment {
            canvas_width: run.view.0,
            canvas_height: run.view.1,
            device_pixel_ratio: context.device_pixel_ratio,
            user_agent: context.user_agent.clone(),
            adapter: AdapterStatus::of(self.adapter()),
            known_differences: KNOWN_DIFFERENCES,
        };
        match run.report(now_ms(), &after, &environment, &context.timing) {
            Ok(benchmark) => {
                let reply = json!({"ok": true, "version": engine_dev::PROTOCOL_VERSION,
                    "observation": run.end, "result": {"benchmark": benchmark, "temporary": true}});
                self.dev_done(reply, context)
            }
            Err(error) => self.dev_refused(error),
        }
    }
}

fn stopped(reply: &Value) -> Stopped {
    match serde_json::from_value::<Error>(reply["error"].clone()) {
        Ok(error) => Error {
            path: "$".to_owned(),
            ..error
        },
        Err(_) => Error::new(ErrorCode::BenchmarkFailed, "$"),
    }
}

fn numbers(value: &Value, prefix: &str, out: &mut Vec<(String, f64)>) {
    if let Value::Object(fields) = value {
        for (key, value) in fields {
            match value {
                Value::Number(number) => {
                    out.push((
                        format!("{prefix}{key}"),
                        number.as_f64().unwrap_or(f64::NAN),
                    ));
                }
                Value::Object(_) => numbers(value, &format!("{prefix}{key}."), out),
                Value::Null | Value::Bool(_) | Value::String(_) | Value::Array(_) => {}
            }
        }
    }
}

fn distribution(values: &[f64]) -> Result<Value, Stopped> {
    Distribution::of(values)
        .map(|distribution| json!(distribution))
        .ok_or_else(|| {
            Error::new(ErrorCode::InvalidMeasurement, "$").expected(if values.is_empty() {
                Expected::NonEmpty
            } else {
                Expected::Finite
            })
        })
}
