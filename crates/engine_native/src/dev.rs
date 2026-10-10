//! The development protocol natively, over standard input and output: one
//! JSON command a line in, one JSON reply a line out, run between frames as
//! the page runs them (`web/src/engine/dev.ts`). The engine answers each
//! command, or hands back a control the host carries out: pause and resume,
//! benchmarks, captures (PNG files), the game's own controls and scene
//! reloads. Profiling builds also take the agent's `profile`: a deep
//! profile of the frames run meanwhile (see `deep.rs`), CPU samples of
//! every thread included unless `cpu_samples` is false, written as files.
//! Sessions draw offscreen and start paused, as the page's development
//! sessions do; a first line says the session is ready.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use engine::{Game, GameHost};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::Offscreen;
use crate::files;

// The interval of frames while the session runs.
const FRAME: Duration = Duration::from_micros(16_667);
// The longest a benchmark samples.
const BENCHMARK_LIMIT: Duration = Duration::from_secs(10);

/// A control a game registers for the `game` command: what it does, and
/// how, on the host and the command's `args`.
pub struct GameControl<G: Game> {
    pub description: String,
    #[allow(clippy::type_complexity)]
    pub run: Box<dyn FnMut(&mut GameHost<G>, &Map<String, Value>) -> Result<Value, String>>,
}

/// A development session of a game drawn offscreen.
pub struct DevSession<G: Game> {
    game: Offscreen<G>,
    controls: BTreeMap<String, GameControl<G>>,
    parameters: BTreeMap<String, String>,
    // The authored scene's file, which reloads read again.
    source: Option<PathBuf>,
    // Where captures go.
    artifacts: PathBuf,
    captures: u32,
    paused: bool,
    started: Instant,
    instance: String,
    // The engine panicked: the session is unusable after.
    panic: Option<String>,
}

impl<G: Game> DevSession<G> {
    /// A session of `game`, writing captures into `artifacts`.
    pub fn new(game: Offscreen<G>, artifacts: PathBuf) -> Self {
        // Distinct per session: the time and the process.
        let mut hash = Sha256::new();
        hash.update(format!(
            "{:?} {}",
            std::time::SystemTime::now(),
            std::process::id()
        ));
        let instance = hash.finalize()[..16]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Self {
            game,
            controls: BTreeMap::new(),
            parameters: BTreeMap::new(),
            source: None,
            artifacts,
            captures: 0,
            paused: true,
            started: Instant::now(),
            instance,
            panic: None,
        }
    }

    /// The session with the game's control `name`.
    #[must_use]
    pub fn control(
        mut self,
        name: &str,
        description: &str,
        run: impl FnMut(&mut GameHost<G>, &Map<String, Value>) -> Result<Value, String> + 'static,
    ) -> Self {
        self.controls.insert(
            name.to_owned(),
            GameControl {
                description: description.to_owned(),
                run: Box::new(run),
            },
        );
        self
    }

    /// The session with the options the program read, with what each does.
    #[must_use]
    pub fn parameters(mut self, parameters: &[(&str, &str)]) -> Self {
        self.parameters = parameters
            .iter()
            .map(|&(name, what)| (name.to_owned(), what.to_owned()))
            .collect();
        self
    }

    /// The session of the authored scene at `path`, which reloads read
    /// again.
    #[must_use]
    pub fn source(mut self, path: &Path) -> Self {
        self.source = Some(path.to_owned());
        self
    }

    /// Serves commands from standard input until it closes.
    pub fn run(mut self) -> engine::Result<()> {
        let lines = read_lines();
        let size = self.game.size();
        self.reply(&json!({
            "ready": true,
            "version": engine::dev::PROTOCOL_VERSION,
            "instance": self.instance,
            "build": option_env!("ENGINE_COMMIT"),
            "size": [size.0, size.1],
            "adapter": adapter(self.game.host()),
            "profiling": cfg!(feature = "profiling"),
            "controls": self.controls.iter().map(|(name, control)| (name.clone(), json!(control.description))).collect::<Map<_, _>>(),
            "parameters": self.parameters,
        }));
        let mut next = Instant::now();
        loop {
            let wait = if self.paused {
                Duration::from_millis(50)
            } else {
                next.saturating_duration_since(Instant::now())
            };
            match lines.recv_timeout(wait) {
                Ok(line) => {
                    let reply = self.request(&line);
                    self.reply(&reply);
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
            if self.panic.is_none() && !self.paused && Instant::now() >= next {
                next = Instant::now() + FRAME;
                let timestamp = self.now();
                if let Err(error) = self.game.frame(timestamp) {
                    eprintln!("The frame failed: {error}");
                }
            }
            self.game.exchange_jobs();
        }
    }

    fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }

    fn reply(&self, reply: &Value) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{reply}");
        let _ = out.flush();
    }

    /// The reply to the command line `line`.
    fn request(&mut self, line: &str) -> Value {
        if let Some(panic) = &self.panic {
            return json!({"ok": false, "error": {"code": "runtime_panicked", "path": "$", "message": panic}});
        }
        if line.len() > engine::dev::MAX_REQUEST_BYTES {
            return self.error(
                "request_too_large",
                "$.command",
                &format!("command exceeds {} bytes", engine::dev::MAX_REQUEST_BYTES),
            );
        }
        let command = match serde_json::from_str::<Value>(line) {
            Ok(Value::Object(command)) if command.get("op").is_some_and(Value::is_string) => {
                command
            }
            _ => {
                return self.error(
                    "invalid_request",
                    "$.command",
                    "expected an object with an op string",
                );
            }
        };
        let handled =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.dispatch(&command)));
        match handled {
            Ok(reply) => reply,
            Err(panic) => {
                let message = panic
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "the engine panicked".into());
                self.panic = Some(message.clone());
                json!({"ok": false, "error": {"code": "runtime_panicked", "path": "$", "message": message}})
            }
        }
    }

    /// Runs `command`: the engine checks it, pause rules included, and runs
    /// it, or returns the control the host carries out, validated with its
    /// defaults.
    fn dispatch(&mut self, command: &Map<String, Value>) -> Value {
        let op = command["op"].as_str().unwrap_or_default().to_owned();
        // The agent's own capture, which the engine's protocol leaves to
        // hosts.
        if op == "profile" {
            return self.profile(command);
        }
        let mut reply = self.invoke(command);
        let control = (reply["ok"] == true)
            .then(|| reply["result"]["control"].as_object().cloned())
            .flatten();
        if let Some(control) = control {
            reply = match op.as_str() {
                "pause" => {
                    self.paused = true;
                    self.invoke_op("status")
                }
                "resume" => {
                    self.paused = false;
                    self.invoke_op("status")
                }
                "benchmark" => self.benchmark(&control),
                "capture" => self.capture(&control),
                "game" => self.game_control(&control),
                "reload" => self.reload(&control),
                _ => {
                    return self.error(
                        "unsupported_command",
                        "$.op",
                        &format!("this host cannot carry out {op}"),
                    );
                }
            };
            if reply["ok"] != true {
                return reply;
            }
        }
        if op == "status" && reply["ok"] == true {
            reply["result"]["adapter"] = adapter(self.game.host());
        }
        self.finish(reply)
    }

    /// The engine's reply to `command`.
    fn invoke(&mut self, command: &Map<String, Value>) -> Value {
        let mut command = command.clone();
        command.insert("version".into(), json!(engine::dev::PROTOCOL_VERSION));
        let (width, height) = self.game.size();
        let paused = self.paused;
        let reply = self.game.host_mut().dev_command(
            &Value::Object(command).to_string(),
            width,
            height,
            paused,
        );
        serde_json::from_str(&reply).expect("development replies are JSON")
    }

    fn invoke_op(&mut self, op: &str) -> Value {
        let mut command = Map::new();
        command.insert("op".into(), json!(op));
        self.invoke(&command)
    }

    fn error(&mut self, code: &str, path: &str, message: &str) -> Value {
        let mut reply =
            json!({"ok": false, "error": {"code": code, "path": path, "message": message}});
        if self.panic.is_none() {
            reply["observation"] = self.invoke_op("status")["observation"].clone();
        }
        reply
    }

    /// `reply` with what the host adds to every successful one.
    fn finish(&mut self, mut reply: Value) -> Value {
        if reply["ok"] == true {
            let rendering: Value =
                serde_json::from_str(&self.game.host().stats()).expect("frame statistics are JSON");
            let observation = reply["observation"].clone();
            let result = reply["result"]
                .as_object_mut()
                .map(std::mem::take)
                .unwrap_or_default();
            let mut finished = result;
            finished.insert("observation".into(), observation);
            finished.insert("paused".into(), json!(self.paused));
            finished.insert(
                "source".into(),
                json!(self.source.as_ref().map(|path| path.display().to_string())),
            );
            finished.insert("rendering".into(), rendering);
            reply["result"] = Value::Object(finished);
        }
        reply
    }

    /// Renders what `control` asks (a fraction of the tick, or the current
    /// state) and writes it as a PNG file, which the reply names.
    fn capture(&mut self, control: &Map<String, Value>) -> Value {
        let mut command = Map::new();
        match control.get("alpha") {
            Some(alpha) => {
                command.insert("op".into(), json!("present"));
                command.insert("alpha".into(), alpha.clone());
                if control.get("gpu") == Some(&json!(true)) {
                    command.insert("gpu".into(), json!(true));
                }
            }
            None => {
                command.insert("op".into(), json!("render"));
            }
        }
        let mut reply = self.invoke(&command);
        if reply["ok"] != true {
            return reply;
        }
        self.captures += 1;
        let path = self
            .artifacts
            .join(format!("capture-{}.png", self.captures));
        let written = std::fs::create_dir_all(&self.artifacts)
            .map_err(|error| error.to_string())
            .and_then(|()| self.game.save_png(&path).map_err(|error| error.to_string()))
            .and_then(|()| std::fs::read(&path).map_err(|error| error.to_string()));
        match written {
            Ok(bytes) => {
                let observation = reply["observation"].clone();
                reply["result"]["artifact"] = json!({
                    "path": path.display().to_string(),
                    "mime_type": "image/png",
                    "bytes": bytes.len(),
                    "sha256": Sha256::digest(&bytes).iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
                    "frame": observation["frame"],
                    "tick": observation["tick"],
                });
                reply
            }
            Err(message) => {
                json!({"ok": false, "observation": reply["observation"], "error": {"code": "capture_failed", "path": "$", "message": message}})
            }
        }
    }

    /// Runs the game's control `control.name` on `control.args`, then
    /// renders.
    fn game_control(&mut self, control: &Map<String, Value>) -> Value {
        let name = control["name"].as_str().unwrap_or_default().to_owned();
        let args = control
            .get("args")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let Some(game) = self.controls.get_mut(&name) else {
            let names: Vec<&str> = self.controls.keys().map(String::as_str).collect();
            let names = if names.is_empty() {
                "none".to_owned()
            } else {
                names.join(", ")
            };
            return self.error(
                "unsupported_command",
                "$.name",
                &format!("this game has no control {name}; it has {names}"),
            );
        };
        let result = match (game.run)(self.game.host_mut(), &args) {
            Ok(result) => result,
            Err(message) => return self.error("game_rule", "$.args", &message),
        };
        let mut rendered = self.invoke_op("render");
        if rendered["ok"] == true {
            rendered["result"] = json!({"game": name, "result": result});
        }
        rendered
    }

    /// Reads the authored scene again with its files and reloads it at the
    /// revisions `control` names, then renders.
    fn reload(&mut self, control: &Map<String, Value>) -> Value {
        let Some(path) = self.source.clone() else {
            return self.error(
                "unsupported_command",
                "$.op",
                "this game has no authored scene reload adapter",
            );
        };
        let loaded = std::fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))
            .and_then(|source| {
                let files =
                    files::scene_files(&path, &source).map_err(|error| error.to_string())?;
                Ok((source, files))
            });
        let (source, files) = match loaded {
            Ok(loaded) => loaded,
            Err(message) => return self.error("source_io", "$", &message),
        };
        let text = |field: &str| control[field].as_str().unwrap_or_default().to_owned();
        let reply = self.game.host_mut().dev_reload(
            &source,
            &text("expected_revision"),
            &text("revision"),
            Some(&files),
        );
        let mut reply: Value = serde_json::from_str(&reply).expect("development replies are JSON");
        if reply["ok"] == true {
            let rendered = self.invoke_op("render");
            if rendered["ok"] != true {
                return rendered;
            }
            reply["observation"] = rendered["observation"].clone();
        }
        reply
    }

    /// Runs frames back to back for `duration_ms` (100 to 5000, 5000 by
    /// default), so the GPU's clocks stay up as under load, while a capture
    /// of at most `max_events` CPU events and every frame's GPU spans runs,
    /// then pauses and writes the report as a file.
    #[cfg(feature = "profiling")]
    fn profile(&mut self, command: &Map<String, Value>) -> Value {
        if command
            .keys()
            .any(|key| !["op", "duration_ms", "max_events", "cpu_samples"].contains(&key.as_str()))
        {
            return self.error(
                "invalid_request",
                "$",
                "profile accepts duration_ms, max_events, and cpu_samples only",
            );
        }
        let bounded =
            |name: &str, (minimum, maximum): (u64, u64), default: u64| match command.get(name) {
                None => Some(default),
                Some(value) => value
                    .as_u64()
                    .filter(|value| (minimum..=maximum).contains(value)),
            };
        let max = engine::profiling::MAX_EVENTS as u64;
        let (Some(duration), Some(max_events)) = (
            bounded("duration_ms", (100, 5000), 5000),
            bounded("max_events", (256, max), max),
        ) else {
            return self.error(
                "invalid_request",
                "$",
                &format!("duration_ms must be 100..5000 and max_events 256..{max}"),
            );
        };
        if !self.paused {
            return self.error(
                "pause_required",
                "$",
                "pause before starting a controlled profiling capture",
            );
        }
        let before = self.invoke_op("status");
        let cpu_samples = command.get("cpu_samples") != Some(&json!(false));
        let capture = match self.game.start_profile(max_events as usize, cpu_samples) {
            Ok(capture) => capture,
            Err(error) => return self.error("profiling_unavailable", "$", &error.message),
        };
        let started = Instant::now();
        let mut failed = None;
        while started.elapsed() < Duration::from_millis(duration) {
            let timestamp = self.now();
            if let Err(error) = self.game.frame(timestamp) {
                failed = Some(error.to_string());
                break;
            }
        }
        let after = self.invoke_op("status");
        let context = json!({
            "program": "engine_native",
            "adapter": adapter(self.game.host()),
            "size": [self.game.size().0, self.game.size().1],
            "duration_ms": duration,
            "observation_start": before["observation"],
            "observation_end": after["observation"],
            "frame_failed": failed,
        });
        self.captures += 1;
        let stem = format!("profile-{}", self.captures);
        let files = match self
            .game
            .finish_profile(capture, &self.artifacts, &stem, context)
        {
            Ok(files) => files,
            Err(error) => return self.error("capture_failed", "$", &error.message),
        };
        let path = |path: &std::path::Path| path.display().to_string();
        json!({"ok": true, "observation": after["observation"], "result": {
            "temporary": true, "paused": true, "complete": files.complete && failed.is_none(),
            "artifacts": {
                "trace": path(&files.report),
                "firefox": path(&files.firefox),
                "summary": path(&files.summary),
                "cpu_profile": null,
            },
            "note": "The summary lists the hottest functions, lines and instructions per thread, GPU pass and draw costs, what was captured and the deeper layers recommended to set up; the firefox file opens at profiler.firefox.com or ui.perfetto.dev.",
        }})
    }

    #[cfg(not(feature = "profiling"))]
    fn profile(&mut self, _: &Map<String, Value>) -> Value {
        self.error(
            "profiling_unavailable",
            "$",
            "native profiling captures need a build with the profiling feature",
        )
    }

    /// Samples `control.samples` frames of `control.mode` after
    /// `control.warmup`, as the page's benchmark does: each frame's CPU
    /// stages and its interval, the GPU's work included.
    fn benchmark(&mut self, control: &Map<String, Value>) -> Value {
        match self.sample_benchmark(control) {
            Ok(benchmark) => {
                let observation = benchmark["observation"]["end"].clone();
                json!({"ok": true, "version": engine::dev::PROTOCOL_VERSION, "observation": observation,
                    "result": {"benchmark": benchmark, "temporary": true}})
            }
            Err((code, message)) => self.error(&code, "$", &message),
        }
    }

    fn sample_benchmark(
        &mut self,
        control: &Map<String, Value>,
    ) -> Result<Value, (String, String)> {
        let mode = control["mode"].as_str().unwrap_or("render").to_owned();
        let warmup = control["warmup"].as_u64().unwrap_or(0);
        let samples = control["samples"].as_u64().unwrap_or(1);
        let start = Instant::now();
        let invoke = |session: &mut Self, op: Value| -> Result<Value, (String, String)> {
            if start.elapsed() >= BENCHMARK_LIMIT {
                return Err((
                    "benchmark_timeout".into(),
                    "benchmark exceeded ten seconds; reduce samples or warmup; gameplay may have advanced".into(),
                ));
            }
            let Value::Object(command) = op else {
                unreachable!("benchmark commands are objects")
            };
            let reply = session.invoke(&command);
            if reply["ok"] != true {
                return Err((
                    reply["error"]["code"]
                        .as_str()
                        .unwrap_or("benchmark_failed")
                        .to_owned(),
                    reply["error"]["message"]
                        .as_str()
                        .unwrap_or("runtime rejected benchmark frame")
                        .to_owned(),
                ));
            }
            Ok(reply)
        };
        let initial = invoke(self, json!({"op": "assets"}))?;
        invoke(self, json!({"op": "input"}))?;
        let frame = match mode.as_str() {
            "step" => json!({"op": "step", "ticks": 1}),
            "present" => json!({"op": "present", "alpha": 0.5}),
            _ => json!({"op": "render"}),
        };
        let mut previous = Instant::now();
        let mut intervals = Vec::new();
        let mut measurements: Vec<Value> = Vec::new();
        let mut end = Value::Null;
        for index in 0..warmup + samples {
            let reply = invoke(self, frame.clone())?;
            let stats: Value =
                serde_json::from_str(&self.game.host().stats()).expect("frame statistics are JSON");
            // The frame's interval includes its GPU work, as presenting
            // frames' do.
            self.game
                .host()
                .wait_for_gpu()
                .map_err(|error| ("benchmark_failed".to_owned(), error.to_string()))?;
            let now = Instant::now();
            if index >= warmup {
                intervals.push(now.duration_since(previous).as_secs_f64() * 1000.0);
                let mut fields = Vec::new();
                numbers(&stats, "", &mut fields);
                if let Some((field, _)) = fields
                    .iter()
                    .find(|(_, value)| !value.is_finite() || *value < 0.0)
                {
                    return Err((
                        "invalid_measurement".into(),
                        format!("invalid {field} statistic"),
                    ));
                }
                measurements.push(stats);
            }
            previous = now;
            end = reply["observation"].clone();
        }
        let last = invoke(self, json!({"op": "assets"}))?;
        let stage = |name: &str| -> Vec<f64> {
            measurements
                .iter()
                .map(|stats| stats[name].as_f64().unwrap_or(0.0))
                .collect()
        };
        let totals: Vec<f64> = measurements
            .iter()
            .map(|stats| {
                ["simulation_ms", "extraction_ms", "render_ms"]
                    .iter()
                    .map(|name| stats[*name].as_f64().unwrap_or(0.0))
                    .sum()
            })
            .collect();
        // Every statistic that reads a number in each sample, with its range.
        let sampled: Vec<Vec<(String, f64)>> = measurements
            .iter()
            .map(|stats| {
                let mut fields = Vec::new();
                numbers(stats, "", &mut fields);
                fields
            })
            .collect();
        let mut ranges = Map::new();
        if let Some(first) = sampled.first() {
            for (path, _) in first {
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
        }
        let (width, height) = self.game.size();
        Ok(json!({
            "version": 1, "mode": mode, "warmup": warmup, "samples": samples,
            "elapsed_ms": start.elapsed().as_secs_f64() * 1000.0,
            "frame_interval_ms": distribution(&intervals)?,
            "cpu_ms": {
                "simulation": distribution(&stage("simulation_ms"))?,
                "extraction": distribution(&stage("extraction_ms"))?,
                "render": distribution(&stage("render_ms"))?,
                "total": distribution(&totals)?,
            },
            "rendering": {"first": measurements.first(), "last": measurements.last(), "ranges": ranges},
            "observation": {"start": initial["observation"], "end": end},
            "assets": {"before": initial["result"], "after": last["result"]},
            "environment": {"canvas_width": width, "canvas_height": height, "device_pixel_ratio": 1,
                "user_agent": "engine_native", "adapter": adapter(self.game.host())},
            "timing": "CPU timings exclude JSON/stats collection; frame intervals include host work and the GPU's work, waited for after each frame offscreen",
        }))
    }
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
                _ => {}
            }
        }
    }
}

/// The spread of measured `values`, as the page's percentiles: linear
/// interpolation between sorted values.
fn distribution(values: &[f64]) -> Result<Value, (String, String)> {
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err((
            "invalid_measurement".into(),
            "measurements must be nonempty, finite and nonnegative".into(),
        ));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let quantile = |fraction: f64| {
        let index = (sorted.len() - 1) as f64 * fraction;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower as f64)
    };
    Ok(
        json!({"min": sorted[0], "median": quantile(0.5), "p95": quantile(0.95),
        "p99": quantile(0.99), "max": sorted[sorted.len() - 1]}),
    )
}

/// The GPU `host` draws on, as replies name it.
fn adapter<G: Game>(host: &GameHost<G>) -> Value {
    let adapter = host.adapter();
    json!({
        "name": adapter.name, "vendor": adapter.vendor, "device": adapter.device,
        "type": adapter.device_type, "driver": adapter.driver,
        "driver_info": adapter.driver_info, "backend": adapter.backend,
    })
}

/// Standard input's lines, read on a thread of their own so frames run
/// while none arrives.
fn read_lines() -> Receiver<String> {
    let (send, lines) = channel();
    std::thread::Builder::new()
        .name("engine-dev-input".into())
        .spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                if !line.trim().is_empty() && send.send(line).is_err() {
                    break;
                }
            }
        })
        .expect("threads start");
    lines
}
