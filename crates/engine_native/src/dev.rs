//! The development protocol natively, over standard input and output: one
//! JSON command a line in, one JSON reply a line out, run between frames as
//! the page runs them (`web/src/engine/dev.ts`). The engine's development
//! bridge answers each command and carries out the control it names step
//! by step (`engine::dev::Step`); the session does what only it can
//! (`engine::dev::Effect`): pausing its frames, waiting for the GPU,
//! running frames at timestamps of their own, writing the last one drawn
//! as a PNG file, the game's own controls and reading the authored scene
//! again. Profiling builds also take the agent's `profile`: a deep profile
//! of the frames run meanwhile (`engine_profiler`'s), CPU samples of every
//! thread included unless `cpu_samples` is false, each pass's hardware
//! counters where the session counts them (`--counters`), and with
//! `vendor: "pix"` the last frame's per-draw counters replayed through PIX
//! where the session loaded it (`--vendor pix`), written as files. Sessions
//! draw offscreen and start paused, as the page's development sessions do;
//! a first line says the session is ready. Each step's GPU work finishes
//! before the next, so frames read back what earlier frames drew in order,
//! however fast the GPU drew. A panic ends the session's process
//! (`engine_panic`'s hook writes its report on standard error), and the
//! program driving the session answers what follows.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use engine::dev::{Artifact, Effect, HostContext, Outcome, Step};
use engine::scene::SceneFiles;
use engine::{ErrorCode, Game, GameHost};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::files;
#[cfg(feature = "profiling")]
use crate::{Frames, PixRecord, ProfileRecord, Record};
use crate::{Offscreen, Pix};

// The interval of frames while the session runs.
const FRAME: Duration = Duration::from_micros(16_667);
// What a benchmark's frame intervals include here.
const TIMING: &str = "CPU timings exclude JSON/stats collection; frame intervals include host work and the GPU's work, waited for after each frame offscreen";

/// A control a game registers for the `game` command: what it does, and
/// how, on the host and the command's `args`.
pub struct GameControl<G: Game> {
    pub description: String,
    #[expect(clippy::type_complexity, reason = "a stored control callback")]
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
    // PIX's capturer, loaded before the game's device, for profiles asking
    // for its counters.
    pix: Option<Pix>,
    captures: u32,
    paused: bool,
    started: Instant,
    instance: String,
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
            pix: None,
            captures: 0,
            paused: true,
            started: Instant::now(),
            instance,
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

    /// The session with more of the options the program read, with what
    /// each does.
    #[must_use]
    pub fn parameters(
        mut self,
        parameters: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.parameters.extend(
            parameters
                .into_iter()
                .map(|(name, what)| (name.into(), what.into())),
        );
        self
    }

    /// The session with PIX's capturer, loaded before the game's device,
    /// for profiles asking for its counters.
    #[must_use]
    pub fn pix(mut self, pix: Option<Pix>) -> Self {
        self.pix = pix;
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
            "counters": counting(),
            "pix": self.pix.is_some(),
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
            if !self.paused && Instant::now() >= next {
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

    /// The reply to the command line `line`: the engine's bridge runs it,
    /// the session carrying out each effect it hands back once the GPU
    /// finished what the step before submitted, so what frames read back
    /// never depends on how fast the GPU drew. Profiles are the session's
    /// own.
    fn request(&mut self, line: &str) -> Value {
        if line.len() <= engine::dev::MAX_REQUEST_BYTES
            && let Ok(Value::Object(command)) = serde_json::from_str::<Value>(line)
            && command.get("op") == Some(&json!("profile"))
        {
            return self.profile(&command);
        }
        let context = self.context();
        let mut step = self.game.host_mut().dev_request(line, &context);
        loop {
            if let Err(error) = self.game.host().wait_for_gpu() {
                return self.error(ErrorCode::RenderFailed, "$", &error.to_string());
            }
            match step {
                Step::Reply(reply) => return reply,
                Step::Effect(effect) => step = self.carry(effect),
            }
        }
    }

    /// Carries out `effect` for the request under way: its next step.
    fn carry(&mut self, effect: Effect) -> Step {
        let failed = |message: String| Outcome {
            failed: Some(message),
            ..Outcome::default()
        };
        let outcome = match effect {
            Effect::Pause { paused } => {
                self.paused = paused;
                Outcome::default()
            }
            // The GPU finished the frame before: `request` waited for it.
            Effect::Wait { .. } => Outcome {
                now: Some(engine::host::now_ms()),
                ..Outcome::default()
            },
            Effect::Frame { timestamp } => match self.game.frame(timestamp) {
                Ok(()) => Outcome::default(),
                Err(error) => failed(error.to_string()),
            },
            Effect::Show => match self.keep_frame() {
                Ok(artifact) => Outcome {
                    artifact: Some(artifact),
                    ..Outcome::default()
                },
                Err(message) => failed(message),
            },
            Effect::Game { name, args } => match self.controls.get_mut(&name) {
                Some(control) => match (control.run)(self.game.host_mut(), &args) {
                    Ok(result) => Outcome {
                        result: Some(result),
                        ..Outcome::default()
                    },
                    Err(message) => failed(message),
                },
                None => Outcome {
                    missing: Some(self.controls.keys().cloned().collect()),
                    ..Outcome::default()
                },
            },
            Effect::Source => match self.read_source() {
                Ok(Some((source, files))) => {
                    let context = self.context();
                    return self
                        .game
                        .host_mut()
                        .dev_reloaded(&source, Some(&files), &context);
                }
                Ok(None) => Outcome {
                    missing: Some(Vec::new()),
                    ..Outcome::default()
                },
                Err(message) => failed(message),
            },
        };
        let context = self.context();
        self.game.host_mut().dev_resume(outcome, &context)
    }

    /// The session's side of each call to the bridge: its view, whether it
    /// is paused, and what it is.
    fn context(&self) -> HostContext {
        let (width, height) = self.game.size();
        HostContext {
            width,
            height,
            paused: self.paused,
            device_pixel_ratio: 1.0,
            user_agent: "engine_native".to_owned(),
            source: self.source.as_ref().map(|path| path.display().to_string()),
            timing: TIMING.to_owned(),
        }
    }

    /// The authored scene read again with the files it references; none
    /// without one.
    fn read_source(&self) -> Result<Option<(String, SceneFiles)>, String> {
        let Some(path) = &self.source else {
            return Ok(None);
        };
        let source = std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let files = files::scene_files(path, &source).map_err(|error| error.to_string())?;
        Ok(Some((source, files)))
    }

    /// The last frame drawn, written as a PNG file.
    fn keep_frame(&mut self) -> Result<Artifact, String> {
        self.captures += 1;
        let path = self
            .artifacts
            .join(format!("capture-{}.png", self.captures));
        std::fs::create_dir_all(&self.artifacts).map_err(|error| error.to_string())?;
        self.game
            .save_png(&path)
            .map_err(|error| error.to_string())?;
        let bytes = std::fs::read(&path).map_err(|error| error.to_string())?;
        Ok(Artifact {
            path: path.display().to_string(),
            mime_type: "image/png".to_owned(),
            bytes: bytes.len() as u64,
            sha256: Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        })
    }

    /// A refusal of `code` at `path`, with the engine's observation.
    fn error(&mut self, code: ErrorCode, path: &str, message: &str) -> Value {
        self.game.host_mut().dev_refusal(code, path, message)
    }

    /// The engine's status reply.
    #[cfg(feature = "profiling")]
    fn status(&mut self) -> Value {
        let context = self.context();
        match self
            .game
            .host_mut()
            .dev_request(r#"{"op":"status"}"#, &context)
        {
            Step::Reply(reply) => reply,
            Step::Effect(effect) => unreachable!("status names no control, yet {effect:?}"),
        }
    }

    /// Runs frames back to back for `duration_ms` (100 to 5000, 5000 by
    /// default), so the GPU's clocks stay up as under load, while a capture
    /// of at most `max_events` CPU events and every frame's GPU spans runs
    /// (and the counters where the session counts them), then pauses and
    /// writes the report as files; with `vendor: "pix"`, the last frame is
    /// also captured through PIX and replayed for each draw's counters.
    #[cfg(feature = "profiling")]
    fn profile(&mut self, command: &Map<String, Value>) -> Value {
        let options = ["op", "duration_ms", "max_events", "cpu_samples", "vendor"];
        if command.keys().any(|key| !options.contains(&key.as_str())) {
            return self.error(
                ErrorCode::InvalidRequest,
                "$",
                "profile accepts duration_ms, max_events, cpu_samples and vendor only",
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
                ErrorCode::InvalidRequest,
                "$",
                &format!("duration_ms must be 100..5000 and max_events 256..{max}"),
            );
        };
        let pix = match command.get("vendor") {
            None => false,
            Some(vendor) if vendor == "pix" => true,
            Some(_) => return self.error(ErrorCode::InvalidRequest, "$.vendor", "vendor is pix"),
        };
        if pix && self.pix.is_none() {
            return self.error(
                ErrorCode::PixUnavailable,
                "$.vendor",
                "PIX's counters need the session started with --vendor pix, where PIX is installed",
            );
        }
        if !self.paused {
            return self.error(
                ErrorCode::PauseRequired,
                "$",
                "pause before starting a controlled profiling capture",
            );
        }
        let before = self.status();
        self.captures += 1;
        let stem = format!("profile-{}", self.captures);
        let record = Record {
            gpu_capture: None,
            profile: Some(ProfileRecord {
                folder: self.artifacts.clone(),
                stem: stem.clone(),
                max_events: Some(max_events as usize),
                cpu_samples: command.get("cpu_samples") != Some(&json!(false)),
                context: json!({
                    "program": "engine_native",
                    "adapter": adapter(self.game.host()),
                    "size": [self.game.size().0, self.game.size().1],
                    "duration_ms": duration,
                    "observation_start": before["observation"],
                }),
            }),
            pix_counters: pix.then(|| PixRecord {
                folder: self.artifacts.join(format!("{stem}-pix")),
                counters: Vec::new(),
            }),
        };
        let started = self.started;
        let at = |_| started.elapsed().as_secs_f64() * 1000.0;
        let recorded = match self.game.record(
            Frames::For(Duration::from_millis(duration)),
            at,
            &record,
            self.pix.as_ref(),
        ) {
            Ok(recorded) => recorded,
            Err(error) => return self.error(ErrorCode::CaptureFailed, "$", &error.message),
        };
        let after = self.status();
        let files = recorded.profile.unwrap_or_default();
        let pix_file = |name: &str| {
            record
                .pix_counters
                .as_ref()
                .map(|dive| dive.folder.join(name).display().to_string())
        };
        json!({"ok": true, "observation": after["observation"], "result": {
            "temporary": true, "paused": true,
            "complete": files["complete"] == true && recorded.failed.is_none(),
            "frame_failed": recorded.failed.map(|error| error.to_string()),
            "artifacts": {
                "trace": files["report"],
                "firefox": files["firefox"],
                "summary": files["summary"],
                "cpu_profile": null,
                "pix_capture": pix_file("frame.wpix"),
                "pix_events": pix_file("pix-events.csv"),
                "pix_summary": pix_file("pix.summary.json"),
            },
            "note": "The summary lists the hottest functions, lines and instructions per thread, GPU pass and draw costs, each pass's hardware counters where the session counts them, what was captured and the deeper layers recommended to set up; the firefox file opens at profiler.firefox.com or ui.perfetto.dev; pix_summary holds each render-graph node's and draw's counters PIX replayed.",
        }})
    }

    #[cfg(not(feature = "profiling"))]
    fn profile(&mut self, _: &Map<String, Value>) -> Value {
        self.error(
            ErrorCode::ProfilingUnavailable,
            "$",
            "native profiling captures need a build with the profiling feature",
        )
    }
}

/// The GPU `host` draws on, as every host's status names it.
fn adapter<G: Game>(host: &GameHost<G>) -> Value {
    json!(engine::host::AdapterStatus::of(host.adapter()))
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

/// Whether the session counts each pass's hardware counters.
fn counting() -> bool {
    #[cfg(feature = "profiling")]
    return engine_profiler::counters::enabled();
    #[cfg(not(feature = "profiling"))]
    false
}
