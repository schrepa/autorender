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

const FRAME: Duration = Duration::from_micros(16_667);
const TIMING: &str = "offscreen_gpu_waited";

pub struct GameControl<G: Game> {
    pub values: String,
    #[expect(clippy::type_complexity, reason = "a stored control callback")]
    pub run: Box<dyn FnMut(&mut GameHost<G>, &Map<String, Value>) -> Result<Value, String>>,
}

pub struct DevSession<G: Game> {
    game: Offscreen<G>,
    controls: BTreeMap<String, GameControl<G>>,
    parameters: BTreeMap<String, String>,
    source: Option<PathBuf>,
    artifacts: PathBuf,
    pix: Option<Pix>,
    captures: u32,
    paused: bool,
    started: Instant,
    instance: String,
}

impl<G: Game> DevSession<G> {
    pub fn new(game: Offscreen<G>, artifacts: PathBuf) -> Self {
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

    #[must_use]
    pub fn control(
        mut self,
        name: &str,
        values: &str,
        run: impl FnMut(&mut GameHost<G>, &Map<String, Value>) -> Result<Value, String> + 'static,
    ) -> Self {
        self.controls.insert(
            name.to_owned(),
            GameControl {
                values: values.to_owned(),
                run: Box::new(run),
            },
        );
        self
    }

    #[must_use]
    pub fn parameters(
        mut self,
        parameters: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        self.parameters.extend(
            parameters
                .into_iter()
                .map(|(name, values)| (name.into(), values.into())),
        );
        self
    }

    #[must_use]
    pub fn pix(mut self, pix: Option<Pix>) -> Self {
        self.pix = pix;
        self
    }

    #[must_use]
    pub fn source(mut self, path: &Path) -> Self {
        self.source = Some(path.to_owned());
        self
    }

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
            "controls": self.controls.iter().map(|(name, control)| (name.clone(), json!(control.values))).collect::<Map<_, _>>(),
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
                    eprintln!("{}", json!({"event": "frame_failed", "error": error}));
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

    fn request(&mut self, line: &str) -> Value {
        if line.len() <= engine::dev::MAX_REQUEST_BYTES
            && let Ok(Value::Object(command)) = serde_json::from_str::<Value>(line)
            && command.get("verb") == Some(&json!("profile"))
            && let Some(deep) = command.get("deep")
        {
            return match deep {
                Value::Object(deep) => self.profile(deep),
                Value::Null
                | Value::Bool(_)
                | Value::Number(_)
                | Value::String(_)
                | Value::Array(_) => {
                    self.error(ErrorCode::InvalidType, "$.deep", "deep is an object")
                }
            };
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
            Effect::Save { name, contents } => match self.keep_file(&name, &contents) {
                Ok(artifact) => Outcome {
                    artifact: Some(artifact),
                    ..Outcome::default()
                },
                Err(message) => failed(message),
            },
            Effect::Stream { records } => {
                for record in records {
                    self.reply(&json!({ "verb": "watch", "record": record }));
                }
                Outcome::default()
            }
        };
        let context = self.context();
        self.game.host_mut().dev_resume(outcome, &context)
    }

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

    fn read_source(&self) -> Result<Option<(String, SceneFiles)>, String> {
        let Some(path) = &self.source else {
            return Ok(None);
        };
        let source = std::fs::read_to_string(path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let files = files::scene_files(path, &source).map_err(|error| error.to_string())?;
        Ok(Some((source, files)))
    }

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

    fn keep_file(&mut self, name: &str, contents: &Value) -> Result<Artifact, String> {
        self.captures += 1;
        let path = self
            .artifacts
            .join(format!("{name}-{}.json", self.captures));
        std::fs::create_dir_all(&self.artifacts).map_err(|error| error.to_string())?;
        let bytes = serde_json::to_vec(contents).map_err(|error| error.to_string())?;
        std::fs::write(&path, &bytes).map_err(|error| error.to_string())?;
        Ok(Artifact {
            path: path.display().to_string(),
            mime_type: "application/json".to_owned(),
            bytes: bytes.len() as u64,
            sha256: Sha256::digest(&bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        })
    }

    fn error(&mut self, code: ErrorCode, path: &str, message: &str) -> Value {
        self.game.host_mut().dev_refusal(code, path, message)
    }

    #[cfg(feature = "profiling")]
    fn status(&mut self) -> Value {
        let context = self.context();
        match self
            .game
            .host_mut()
            .dev_request(r#"{"verb":"query","name":"status"}"#, &context)
        {
            Step::Reply(reply) => reply,
            Step::Effect(effect) => unreachable!("status_has_no_effect {effect:?}"),
        }
    }

    #[cfg(feature = "profiling")]
    fn profile(&mut self, command: &Map<String, Value>) -> Value {
        let options = ["duration_ms", "max_events", "cpu_samples", "vendor"];
        if let Some(unknown) = command.keys().find(|key| !options.contains(&key.as_str())) {
            return self.error(
                ErrorCode::UnknownField,
                &format!("$.deep.{unknown}"),
                "deep profiles take duration_ms, max_events, cpu_samples and vendor",
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
                ErrorCode::OutOfRange,
                "$.deep",
                &format!("duration_ms must be 100..5000 and max_events 256..{max}"),
            );
        };
        let pix = match command.get("vendor") {
            None => false,
            Some(vendor) if vendor == "pix" => true,
            Some(_) => {
                return self.error(ErrorCode::UnknownVariant, "$.deep.vendor", "vendor is pix");
            }
        };
        if pix && self.pix.is_none() {
            return self.error(
                ErrorCode::PixUnavailable,
                "$.deep.vendor",
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

fn adapter<G: Game>(host: &GameHost<G>) -> Value {
    json!(engine::host::AdapterStatus::of(host.adapter()))
}

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
        .expect("thread_starts");
    lines
}

fn counting() -> bool {
    #[cfg(feature = "profiling")]
    return engine_profiler::counters::enabled();
    #[cfg(not(feature = "profiling"))]
    false
}
