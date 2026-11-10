use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use engine::dev::{Artifact, Effect, HostContext, Outcome, Step};
use engine::scene::{SceneFiles, SceneKinds};
use engine::{Error, ErrorCode, Expected, Game, GameHost};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

#[cfg(feature = "profiling")]
use crate::{Frames, PixRecord, ProfileRecord, Record};
use crate::{Offscreen, Pix};
use crate::{file_error, files};

const FRAME: Duration = Duration::from_micros(16_667);
const TIMING: &str = "offscreen_gpu_waited";
const PROFILE_CAPTURE: &str = "ProfileCapture";

pub struct GameControl<G: Game> {
    pub values: String,
    #[expect(clippy::type_complexity, reason = "a stored control callback")]
    pub run: Box<dyn FnMut(&mut GameHost<G>, &Map<String, Value>) -> Result<Value, String>>,
}

pub struct DevSession<G: Game> {
    game: Offscreen<G>,
    controls: BTreeMap<String, GameControl<G>>,
    parameters: BTreeMap<String, String>,
    document: Option<String>,
    artifacts: PathBuf,
    pix: Option<Pix>,
    captures: u32,
    paused: bool,
    started: Instant,
    instance: String,
}

impl<G: Game> DevSession<G> {
    pub fn new(mut game: Offscreen<G>, artifacts: PathBuf) -> Self {
        game.host_mut().set_preparation(false);
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
            document: None,
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

    pub fn run(mut self) -> engine::Result<()> {
        let lines = read_lines();
        let size = self.game.size();
        self.reply(&json!({
            "ready": true,
            "version": engine::dev::PROTOCOL_VERSION,
            "instance": self.instance,
            "size": [size.0, size.1],
            "adapter": adapter(self.game.host()),
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
                    let (id, line) = request_id(&line);
                    let mut reply = self.request(&line, &|| None);
                    if let Some(id) = id {
                        reply["id"] = id;
                    }
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
            self.follow_source();
        }
    }

    fn follow_source(&mut self) {
        let Some(followed) = self.game.host_mut().follow_source() else {
            return;
        };
        let mut reply = match followed {
            Ok(()) => self.request(r#"{"verb":"query","name":"status"}"#, &|| None),
            Err(error) => self.error(&error),
        };
        reply["event"] = json!("source_changed");
        self.reply(&reply);
    }

    fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }

    fn reply(&self, reply: &Value) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{reply}");
        let _ = out.flush();
    }

    fn request(&mut self, line: &str, late: &dyn Fn() -> Option<Value>) -> Value {
        if let Ok(Value::Object(command)) = serde_json::from_str::<Value>(line) {
            if command.get("verb") == Some(&json!("profile"))
                && let Some(deep) = command.get("deep")
            {
                return match deep {
                    Value::Object(deep) => self.profile(deep),
                    Value::Null
                    | Value::Bool(_)
                    | Value::Number(_)
                    | Value::String(_)
                    | Value::Array(_) => self.error(
                        &Error::new(ErrorCode::InvalidType, "$.deep").expected(Expected::Object),
                    ),
                };
            }
            if command.get("verb") == Some(&json!("describe"))
                && command.get("name") == Some(&json!(PROFILE_CAPTURE))
            {
                return self.describe_profile(&command);
            }
        }
        let context = self.context();
        let mut step = self.game.host_mut().dev_request(line, &context);
        loop {
            if let Err(error) = self.game.host().wait_for_gpu() {
                return self.error(&Error::new(ErrorCode::RenderFailed, "$").got(error));
            }
            match step {
                Step::Reply(reply) => return reply,
                Step::Effect(effect) => match late() {
                    Some(error) => return json!({"ok": false, "error": error}),
                    None => step = self.carry(effect),
                },
            }
        }
    }

    fn carry(&mut self, effect: Effect) -> Step {
        let failed = |error: Error| Outcome {
            failed: Some(error.to_string()),
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
                Err(error) => failed(error),
            },
            Effect::Show => match self.keep_frame() {
                Ok(artifact) => Outcome {
                    artifact: Some(artifact),
                    ..Outcome::default()
                },
                Err(error) => failed(error),
            },
            Effect::Game { name, args } => match self.controls.get_mut(&name) {
                Some(control) => match (control.run)(self.game.host_mut(), &args) {
                    Ok(result) => Outcome {
                        result: Some(result),
                        ..Outcome::default()
                    },
                    Err(error) => Outcome {
                        failed: Some(error),
                        ..Outcome::default()
                    },
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
                Err(error) => failed(error),
            },
            Effect::Save { name, contents } => match self.keep_file(&name, &contents) {
                Ok(artifact) => Outcome {
                    artifact: Some(artifact),
                    ..Outcome::default()
                },
                Err(error) => failed(error),
            },
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
            source: self
                .game
                .host()
                .source()
                .map(|path| path.display().to_string()),
            timing: TIMING.to_owned(),
        }
    }

    fn kinds(&self) -> SceneKinds {
        SceneKinds::of(self.game.host().engine().app.world())
    }

    fn read_source(&mut self) -> Result<Option<(String, SceneFiles)>, Error> {
        if let Some(document) = &self.document {
            let files = engine::smokes::scene_files(document, &self.kinds())?;
            return Ok(Some((document.clone(), files)));
        }
        let Some(path) = self.game.host().source().map(Path::to_owned) else {
            return Ok(None);
        };
        self.game.host_mut().watch_source(&path);
        let source = std::fs::read_to_string(&path)
            .map_err(|error| file_error("$.source", &path, &error))?;
        let files = files::scene_files(&path, &source, &self.kinds())?;
        Ok(Some((source, files)))
    }

    fn keep_frame(&mut self) -> Result<Artifact, Error> {
        self.captures += 1;
        let path = self
            .artifacts
            .join(format!("capture-{}.png", self.captures));
        std::fs::create_dir_all(&self.artifacts)
            .map_err(|error| file_error("$.artifacts", &self.artifacts, &error))?;
        self.game.save_png(&path)?;
        let bytes =
            std::fs::read(&path).map_err(|error| file_error("$.artifact", &path, &error))?;
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

    fn keep_file(&mut self, name: &str, contents: &Value) -> Result<Artifact, Error> {
        self.captures += 1;
        let path = self
            .artifacts
            .join(format!("{name}-{}.json", self.captures));
        std::fs::create_dir_all(&self.artifacts)
            .map_err(|error| file_error("$.artifacts", &self.artifacts, &error))?;
        let bytes = serde_json::to_vec(contents)
            .map_err(|_| Error::new(ErrorCode::InvalidJson, "$.contents"))?;
        std::fs::write(&path, &bytes).map_err(|error| file_error("$.artifact", &path, &error))?;
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

    fn error(&mut self, error: &Error) -> Value {
        self.game.host_mut().dev_refusal(error)
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
                &Error::new(ErrorCode::UnknownField, format!("$.deep.{unknown}"))
                    .got(unknown)
                    .variants(options),
            );
        }
        let bounded =
            |name: &str, (minimum, maximum): (u64, u64), default: u64| match command.get(name) {
                None => Some(default),
                Some(value) => value
                    .as_u64()
                    .filter(|value| (minimum..=maximum).contains(value)),
            };
        let durations = (
            *engine::PROFILE_DURATION_MS.start(),
            *engine::PROFILE_DURATION_MS.end(),
        );
        let events = (
            *engine::PROFILE_EVENTS.start() as u64,
            *engine::PROFILE_EVENTS.end() as u64,
        );
        let out_of_range = |name: &str, (minimum, maximum): (u64, u64)| {
            Error::new(ErrorCode::OutOfRange, format!("$.deep.{name}"))
                .range(minimum as f64, maximum as f64)
                .got(command.get(name))
        };
        let (duration, max_events) = match (
            bounded("duration_ms", durations, durations.1),
            bounded("max_events", events, events.1),
        ) {
            (Some(duration), Some(max_events)) => (duration, max_events),
            (None, _) => return self.error(&out_of_range("duration_ms", durations)),
            (Some(_), None) => return self.error(&out_of_range("max_events", events)),
        };
        let pix = match command.get("vendor") {
            None => false,
            Some(vendor) if vendor == "pix" => true,
            Some(vendor) => {
                return self.error(
                    &Error::new(ErrorCode::UnknownVariant, "$.deep.vendor")
                        .got(vendor)
                        .variants(["pix"]),
                );
            }
        };
        if pix && self.pix.is_none() {
            return self.error(
                &Error::new(ErrorCode::PixUnavailable, "$.deep.vendor")
                    .expected(Expected::Supported)
                    .got("pix"),
            );
        }
        if !self.paused {
            return self
                .error(&Error::new(ErrorCode::PauseRequired, "$").expected(Expected::Paused));
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
            Err(error) => {
                return self.error(&Error::new(ErrorCode::CaptureFailed, "$").got(error));
            }
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
            "frame_failed": recorded.failed,
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
            &Error::new(ErrorCode::ProfilingUnavailable, "$.deep").expected(Expected::Supported),
        )
    }

    #[cfg(feature = "profiling")]
    fn describe_profile(&mut self, command: &Map<String, Value>) -> Value {
        let fields = ["verb", "name"];
        if let Some(unknown) = command.keys().find(|key| !fields.contains(&key.as_str())) {
            return self.error(
                &Error::new(ErrorCode::UnknownField, "$")
                    .got(unknown)
                    .variants(fields),
            );
        }
        let mut reply = self.status();
        if reply["ok"] != true {
            return reply;
        }
        let (durations, events) = (engine::PROFILE_DURATION_MS, engine::PROFILE_EVENTS);
        reply["result"] = json!({
            "name": PROFILE_CAPTURE,
            "kind": "tooling",
            "schema": {"type": "object", "additionalProperties": false, "properties": {
                "deep": {"type": "object", "additionalProperties": false, "properties": {
                    "duration_ms": {
                        "type": "integer",
                        "minimum": durations.start(),
                        "maximum": durations.end(),
                        "default": durations.end(),
                    },
                    "max_events": {
                        "type": "integer",
                        "minimum": events.start(),
                        "maximum": events.end(),
                        "default": events.end(),
                    },
                    "cpu_samples": {"type": "boolean", "default": true},
                    "vendor": {"enum": ["pix"]},
                }},
            }},
            "status": {
                "available": true,
                "native": true,
                "counters": engine_profiler::counters::enabled(),
                "pix": self.pix.is_some(),
            },
        });
        reply
    }

    #[cfg(not(feature = "profiling"))]
    fn describe_profile(&mut self, _: &Map<String, Value>) -> Value {
        self.error(&Error::new(ErrorCode::ProfilingUnavailable, "$").expected(Expected::Supported))
    }
}

impl<G: Game> engine::smokes::GpuSession for DevSession<G> {
    fn world(&self) -> &engine::bevy_ecs::world::World {
        self.game.host().engine().app.world()
    }

    fn send(&mut self, line: &str, late: &dyn Fn() -> Option<Value>) -> Value {
        let reply = self.request(line, late);
        self.game.exchange_jobs();
        reply
    }

    fn play(&mut self, document: &str) -> engine::Result<()> {
        self.document = Some(document.to_owned());
        let files = engine::smokes::scene_files(document, &self.kinds())?;
        self.game.host_mut().reload_scene(document, &files)
    }
}

fn adapter<G: Game>(host: &GameHost<G>) -> Value {
    json!(engine::host::AdapterStatus::of(host.adapter()))
}

fn request_id(line: &str) -> (Option<Value>, String) {
    match serde_json::from_str::<Value>(line) {
        Ok(Value::Object(mut command)) if command.contains_key("id") => {
            let id = command.remove("id");
            (id, Value::Object(command).to_string())
        }
        _ => (None, line.to_owned()),
    }
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
