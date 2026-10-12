//! The development bridge every host serves requests through, the page's
//! (`web/src/engine/dev.ts`) and the native session's: requests checked and
//! handed to the engine, and the controls the engine hands back carried out
//! step by step ([`Step`]), the host doing only what it alone can
//! ([`Effect`]): pausing its frames, waiting for its next frame, running
//! and showing frames, the game's own controls and the authored scene's
//! files. Replies are finished with what every one carries, refusals carry
//! the engine's observation, and benchmarks sample frame by frame
//! ([`BenchmarkRun`]).

use engine_core::ErrorCode;
use engine_dev::{Effect, HostContext, Outcome, Step};
use engine_scene::SceneFiles;
use serde_json::{Map, Value, json};

use crate::host::policy::BENCHMARK_LIMIT_MS;
use crate::host::{AdapterStatus, Distribution, HostEnvironment, KNOWN_DIFFERENCES, now_ms};
use crate::{Game, GameHost};

/// Why a benchmark stops where its view never showed, and where it
/// changed while it ran.
const NEEDS_VIEW: &str = "benchmark needs a visible, nonzero paused canvas";
const VIEW_CHANGED: &str = "visibility, canvas size or paused state changed during benchmark";

/// The development request under way: what it waits for from its host.
#[derive(Debug)]
pub(crate) enum Pending {
    /// The host pausing or resuming its frames.
    Paused,
    /// Ordinary frames `interval` milliseconds apart, `ran` of `count` so
    /// far; the last one shown where `show`.
    Frames {
        count: u32,
        interval: f64,
        ran: u32,
        show: bool,
    },
    /// The host showing the frame `reply` drew.
    Shown { reply: Value },
    /// The game's control `name` running.
    Game { name: String },
    /// The authored scene's files being read, for a reload from
    /// `expected_revision` to `revision`.
    Reload {
        expected_revision: String,
        revision: String,
    },
    /// A benchmark sampling.
    Benchmark(Box<BenchmarkRun>),
}

/// A benchmark under way: frames of its mode the engine runs, each paired
/// with the host's wait for its next frame (the page's next animation
/// frame, the native host's GPU) and timed from the end of the wait before
/// to the end of its own.
#[derive(Debug)]
pub(crate) struct BenchmarkRun {
    /// The frame command each sample runs.
    frame: Value,
    mode: String,
    warmup: u64,
    samples: u64,
    started: f64,
    // The view it runs on, which must stay.
    view: (u32, u32),
    // When the wait before the next frame ended.
    previous: Option<f64>,
    // The statistics of the frame run and not yet waited for.
    unwaited: Option<Value>,
    ran: u64,
    intervals: Vec<f64>,
    measurements: Vec<Value>,
    // The assets reply before the frames.
    initial: Value,
    // The last frame's observation.
    end: Value,
}

/// Why a benchmark stopped: its code and message.
type Stopped = (ErrorCode, String);

impl BenchmarkRun {
    /// The host's wait for the last frame ended at `now`: its interval and
    /// statistics join the samples past the warm-up. Whether every frame
    /// ran.
    fn waited(&mut self, now: f64) -> Result<bool, Stopped> {
        if now - self.started >= BENCHMARK_LIMIT_MS as f64 {
            return Err((
                ErrorCode::BenchmarkTimeout,
                "benchmark exceeded ten seconds; reduce samples or warmup; gameplay may have advanced"
                    .into(),
            ));
        }
        if let (Some(previous), Some(stats)) = (self.previous, self.unwaited.take())
            && self.ran > self.warmup
        {
            let mut fields = Vec::new();
            numbers(&stats, "", &mut fields);
            if let Some((field, _)) = fields
                .iter()
                .find(|(_, value)| !value.is_finite() || *value < 0.0)
            {
                return Err((
                    ErrorCode::InvalidMeasurement,
                    format!("invalid {field} statistic"),
                ));
            }
            self.intervals.push((now - previous).max(0.0));
            self.measurements.push(stats);
        }
        self.previous = Some(now);
        Ok(self.ran >= self.warmup + self.samples)
    }

    /// The report of the frames sampled, with the assets `after` them, the
    /// host they ran on and its `timing` (what its frame intervals
    /// include), at `now`.
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
        // Every statistic that reads a number in each sample, with its range.
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
    /// Development request `line`, as an agent sent it (JSON), on the host
    /// `context` describes: refused where it is too large or no object
    /// naming its `op`, else the engine's reply, or the first effect of the
    /// control it names. A request under way before it is dropped.
    pub fn dev_request(&mut self, line: &str, context: &HostContext) -> Step {
        self.dev_pending = None;
        if line.len() > engine_dev::MAX_REQUEST_BYTES {
            return self.dev_refused(
                ErrorCode::RequestTooLarge,
                "$.command",
                &format!("command exceeds {} bytes", engine_dev::MAX_REQUEST_BYTES),
            );
        }
        let command = match serde_json::from_str::<Value>(line) {
            Ok(command @ Value::Object(_)) if command["op"].is_string() => command,
            _ => {
                return self.dev_refused(
                    ErrorCode::InvalidRequest,
                    "$.command",
                    "expected an object with an op string",
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

    /// The request under way resumed with the `outcome` of the effect its
    /// host carried out, on the host `context` describes: the next step.
    pub fn dev_resume(&mut self, outcome: Outcome, context: &HostContext) -> Step {
        let Some(pending) = self.dev_pending.take() else {
            return self.dev_refused(
                ErrorCode::InvalidRequest,
                "$",
                "no development request is under way",
            );
        };
        match pending {
            Pending::Paused => {
                let status = self.dev_invoke(&json!({"op": "status"}), context);
                self.dev_done(status, context)
            }
            Pending::Frames {
                count,
                interval,
                ran,
                show,
            } => {
                if let Some(message) = outcome.failed {
                    return self.dev_refused(
                        ErrorCode::RenderFailed,
                        "$",
                        &format!("frame {} of {count} failed: {message}", ran + 1),
                    );
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
                let mut reply = self.dev_invoke(&json!({"op": "status"}), context);
                reply["result"] =
                    json!({"frames": count, "interval_ms": interval, "temporary": true});
                if show {
                    return self.dev_effect(Pending::Shown { reply }, Effect::Show);
                }
                self.dev_done(reply, context)
            }
            Pending::Shown { mut reply } => {
                if let Some(message) = outcome.failed {
                    return self.dev_refused(ErrorCode::CaptureFailed, "$", &message);
                }
                if let Some(artifact) = outcome.artifact {
                    let observation = &reply["observation"];
                    let kept = json!({
                        "path": artifact.path, "mime_type": artifact.mime_type,
                        "bytes": artifact.bytes, "sha256": artifact.sha256,
                        "frame": observation["frame"], "tick": observation["tick"],
                    });
                    reply["result"]["artifact"] = kept;
                }
                self.dev_done(reply, context)
            }
            Pending::Game { name } => {
                if let Some(names) = outcome.missing {
                    let names = if names.is_empty() {
                        "none".to_owned()
                    } else {
                        names.join(", ")
                    };
                    return self.dev_refused(
                        ErrorCode::UnsupportedCommand,
                        "$.name",
                        &format!("this game has no control {name}; it has {names}"),
                    );
                }
                if let Some(message) = outcome.failed {
                    return self.dev_refused(ErrorCode::GameRule, "$.args", &message);
                }
                let mut rendered = self.dev_invoke(&json!({"op": "render"}), context);
                if rendered["ok"] == true {
                    rendered["result"] =
                        json!({"game": name, "result": outcome.result.unwrap_or(Value::Null)});
                }
                self.dev_done(rendered, context)
            }
            Pending::Reload { .. } => {
                if outcome.missing.is_some() {
                    return self.dev_refused(
                        ErrorCode::UnsupportedCommand,
                        "$.op",
                        "this game has no authored scene reload adapter",
                    );
                }
                let message = outcome.failed.unwrap_or_else(|| {
                    "the host handed the reload no scene (dev_reloaded)".to_owned()
                });
                self.dev_refused(ErrorCode::SourceIo, "$", &message)
            }
            Pending::Benchmark(run) => self.dev_benchmark(run, &outcome, context),
        }
    }

    /// The reload under way resumed with the authored scene's saved
    /// `source` and the `files` it references ([`Effect::Source`]), on the
    /// host `context` describes: the reloaded scene rendered, or why the
    /// reload was refused.
    pub fn dev_reloaded(
        &mut self,
        source: &str,
        files: Option<&SceneFiles>,
        context: &HostContext,
    ) -> Step {
        let Some(Pending::Reload {
            expected_revision,
            revision,
        }) = self.dev_pending.take()
        else {
            return self.dev_refused(ErrorCode::InvalidRequest, "$", "no reload is under way");
        };
        let reply = self.dev_reload(source, &expected_revision, &revision, files);
        let mut reply: Value = serde_json::from_str(&reply).expect("development replies are JSON");
        if reply["ok"] == true {
            let rendered = self.dev_invoke(&json!({"op": "render"}), context);
            if rendered["ok"] != true {
                return self.dev_done(rendered, context);
            }
            reply["observation"] = rendered["observation"].clone();
        }
        self.dev_done(reply, context)
    }

    /// A refusal of `code` at `path` its host decided, with the engine's
    /// observation; a request under way goes on.
    pub fn dev_refusal(&mut self, code: ErrorCode, path: &str, message: &str) -> Value {
        // Status reads neither the view nor whether the host is paused.
        let status = json!({"op": "status", "version": engine_dev::PROTOCOL_VERSION});
        let status: Value =
            serde_json::from_str(&self.dev_command(&status.to_string(), 1, 1, true))
                .expect("development replies are JSON");
        json!({"ok": false, "observation": status["observation"],
            "error": {"code": code, "path": path, "message": message}})
    }

    /// The engine's reply to `command` (an object) at the protocol's
    /// version, on the host `context` describes.
    fn dev_invoke(&mut self, command: &Value, context: &HostContext) -> Value {
        let mut command = command.clone();
        command["version"] = json!(engine_dev::PROTOCOL_VERSION);
        let reply = self.dev_command(
            &command.to_string(),
            context.width,
            context.height,
            context.paused,
        );
        serde_json::from_str(&reply).expect("development replies are JSON")
    }

    /// Starts carrying out `control`, which the engine handed back
    /// validated with its defaults: its first effect, or why it cannot.
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
                // As play after a pause: the first frame advances no time,
                // and nothing is held.
                self.suspend();
                let pending = Pending::Frames {
                    count,
                    interval,
                    ran: 0,
                    show,
                };
                self.dev_effect(pending, Effect::Frame { timestamp: 0.0 })
            }
            "capture" => {
                let command = match control.get("alpha") {
                    Some(alpha) if control.get("gpu") == Some(&json!(true)) => {
                        json!({"op": "present", "alpha": alpha, "gpu": true})
                    }
                    Some(alpha) => json!({"op": "present", "alpha": alpha}),
                    None => json!({"op": "render"}),
                };
                let reply = self.dev_invoke(&command, context);
                if reply["ok"] != true {
                    return self.dev_done(reply, context);
                }
                self.dev_effect(Pending::Shown { reply }, Effect::Show)
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
            _ => self.dev_refused(
                ErrorCode::UnsupportedCommand,
                "$.op",
                &format!("this host cannot carry out {op}"),
            ),
        }
    }

    /// A step waiting for its host's `effect`, after which the request
    /// goes on as `pending` says.
    fn dev_effect(&mut self, pending: Pending, effect: Effect) -> Step {
        self.dev_pending = Some(pending);
        Step::Effect(effect)
    }

    /// The request's `reply`, finished where it succeeded: its result with
    /// the observation again, whether the host is paused, the authored
    /// scene's source and the last frame's statistics (`rendering`).
    fn dev_done(&mut self, mut reply: Value, context: &HostContext) -> Step {
        self.dev_pending = None;
        if reply["ok"] == true {
            let rendering: Value =
                serde_json::from_str(&self.stats()).expect("frame statistics are JSON");
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

    /// A refusal of `code` at `path`, ending the request under way.
    fn dev_refused(&mut self, code: ErrorCode, path: &str, message: &str) -> Step {
        self.dev_pending = None;
        Step::Reply(self.dev_refusal(code, path, message))
    }

    /// Starts benchmark `control` on the view `context` describes: the
    /// assets before it and the input batch cleared, then a wait for the
    /// host's frame.
    fn dev_benchmark_start(&mut self, control: &Map<String, Value>, context: &HostContext) -> Step {
        if context.width == 0 || context.height == 0 {
            return self.dev_refused(ErrorCode::BenchmarkInterrupted, "$", NEEDS_VIEW);
        }
        let started = now_ms();
        let mode = control
            .get("mode")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let frame = match mode.as_str() {
            "step" => json!({"op": "step", "ticks": 1}),
            "present" => json!({"op": "present", "alpha": 0.5}),
            _ => json!({"op": "render"}),
        };
        let number = |field: &str| control.get(field).and_then(Value::as_u64).unwrap_or(0);
        let initial = self.dev_invoke(&json!({"op": "assets"}), context);
        let cleared = self.dev_invoke(&json!({"op": "input"}), context);
        if let Some(refused) = [&initial, &cleared]
            .into_iter()
            .find(|reply| reply["ok"] != true)
        {
            let (code, message) = stopped(refused);
            return self.dev_refused(code, "$", &message);
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

    /// Benchmark `run` resumed once its host's wait ended as `outcome`
    /// says: its next frame and wait, or its report once every frame ran.
    fn dev_benchmark(
        &mut self,
        mut run: Box<BenchmarkRun>,
        outcome: &Outcome,
        context: &HostContext,
    ) -> Step {
        if let Some(message) = &outcome.failed {
            return self.dev_refused(ErrorCode::BenchmarkFailed, "$", message);
        }
        if outcome.interrupted || !context.paused || (context.width, context.height) != run.view {
            let message = if run.previous.is_none() {
                NEEDS_VIEW
            } else {
                VIEW_CHANGED
            };
            return self.dev_refused(ErrorCode::BenchmarkInterrupted, "$", message);
        }
        let done = match run.waited(outcome.now.unwrap_or_else(now_ms)) {
            Ok(done) => done,
            Err((code, message)) => return self.dev_refused(code, "$", &message),
        };
        if done {
            return self.dev_benchmark_report(&run, context);
        }
        let reply = self.dev_invoke(&run.frame, context);
        if reply["ok"] != true {
            let (code, message) = stopped(&reply);
            return self.dev_refused(code, "$", &message);
        }
        run.end = reply["observation"].clone();
        run.unwaited =
            Some(serde_json::from_str(&self.stats()).expect("frame statistics are JSON"));
        run.ran += 1;
        let within_ms = (BENCHMARK_LIMIT_MS as f64 - (now_ms() - run.started)).max(1.0);
        self.dev_effect(Pending::Benchmark(run), Effect::Wait { within_ms })
    }

    /// Benchmark `run`'s reply once every frame ran: its report, with the
    /// assets after it and the host it ran on.
    fn dev_benchmark_report(&mut self, run: &BenchmarkRun, context: &HostContext) -> Step {
        let after = self.dev_invoke(&json!({"op": "assets"}), context);
        if after["ok"] != true {
            let (code, message) = stopped(&after);
            return self.dev_refused(code, "$", &message);
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
            Err((code, message)) => self.dev_refused(code, "$", &message),
        }
    }
}

/// The code and message of the engine's refusal `reply`, as a benchmark
/// stops with them.
fn stopped(reply: &Value) -> Stopped {
    let code = reply["error"]["code"]
        .as_str()
        .and_then(ErrorCode::named)
        .unwrap_or(ErrorCode::BenchmarkFailed);
    let message = reply["error"]["message"]
        .as_str()
        .unwrap_or("runtime rejected benchmark frame")
        .to_owned();
    (code, message)
}

/// Every number in `value` by its path, such as `terrain.patches`.
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

/// The spread of measured `values`, as the engine computes it on every
/// host.
fn distribution(values: &[f64]) -> Result<Value, Stopped> {
    Distribution::of(values)
        .map(|distribution| json!(distribution))
        .ok_or_else(|| {
            (
                ErrorCode::InvalidMeasurement,
                "measurements must be nonempty, finite and nonnegative".into(),
            )
        })
}
