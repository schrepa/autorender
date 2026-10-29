use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use super::matrix::{Capture, Cell};
use super::sides::Side;
use crate::bench::{self, Scene};
use crate::measure;

pub(super) const SETTINGS: &str = "bench/settings/default.json";

const COMMAND_LIMIT: Duration = Duration::from_secs(240);

const SETTLE_LIMIT: Duration = Duration::from_secs(300);

const QUIET_RENDERS: usize = 3;

const TAIL_RENDERS: usize = 30;

struct Session {
    child: Child,
    input: ChildStdin,
    lines: Receiver<String>,
    legacy: bool,
}

fn legacy(command: &Value) -> Value {
    let mut sent = Map::new();
    let verb = command["verb"].as_str().unwrap_or_default();
    match verb {
        "query" | "control" => {
            sent.insert("op".to_owned(), command["name"].clone());
            if let Some(args) = command["args"].as_object() {
                sent.extend(args.clone());
            }
        }
        _ => {
            sent.insert("op".to_owned(), json!(verb));
            for (key, value) in command.as_object().into_iter().flatten() {
                if key != "verb" {
                    sent.insert(key.clone(), value.clone());
                }
            }
        }
    }
    Value::Object(sent)
}

fn named(verb: &str, name: &str, args: Value) -> Value {
    json!({ "verb": verb, "name": name, "args": args })
}

impl Session {
    fn launch(
        program: &Path,
        tree: &Path,
        arguments: &[String],
        log: &Path,
    ) -> Result<Self, String> {
        let errors = fs::File::create(log).map_err(|error| error.to_string())?;
        let mut child = Command::new(program)
            .args(arguments)
            .current_dir(tree)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(errors)
            .spawn()
            .map_err(|error| format!("cannot start {}: {error}", program.display()))?;
        let input = child.stdin.take().expect("stdio_piped");
        let output = child.stdout.take().expect("stdio_piped");
        let (send, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                let Ok(line) = line else { break };
                if send.send(line).is_err() {
                    break;
                }
            }
        });
        let mut session = Self {
            child,
            input,
            lines,
            legacy: false,
        };
        let ready = session.next(log)?;
        if ready["ready"] != true {
            return Err(format!("the program did not say it is ready: {ready}"));
        }
        let probe = session.request(&named("query", "status", json!({})), log)?;
        session.legacy = probe["ok"] != true && probe["error"]["path"] == "$.command";
        Ok(session)
    }

    fn next(&mut self, log: &Path) -> Result<Value, String> {
        let deadline = Instant::now() + COMMAND_LIMIT;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) => match serde_json::from_str::<Value>(&line) {
                    Ok(value) if value.is_object() => return Ok(value),
                    _ => continue,
                },
                Err(RecvTimeoutError::Timeout) => {
                    return Err(format!(
                        "the program did not answer within {} s; see {}",
                        COMMAND_LIMIT.as_secs(),
                        log.display()
                    ));
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let status = self
                        .child
                        .wait()
                        .map_or_else(|error| error.to_string(), |status| status.to_string());
                    return Err(format!(
                        "the program exited ({status}); see {}",
                        log.display()
                    ));
                }
            }
        }
    }

    fn request(&mut self, command: &Value, log: &Path) -> Result<Value, String> {
        let command = if self.legacy {
            legacy(command)
        } else {
            command.clone()
        };
        writeln!(self.input, "{command}")
            .and_then(|()| self.input.flush())
            .map_err(|error| format!("cannot send {command}: {error}"))?;
        self.next(log)
    }

    fn ok(&mut self, command: &Value, log: &Path) -> Result<Value, String> {
        let reply = self.request(command, log)?;
        if reply["ok"] != true {
            return Err(format!(
                "{} {}: {}",
                command["verb"].as_str().unwrap_or_default(),
                command["name"].as_str().unwrap_or_default(),
                reply["error"]
            ));
        }
        Ok(reply["result"].clone())
    }

    fn close(mut self) {
        drop(self.input);
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Clone)]
pub(super) enum Shot {
    Taken {
        path: PathBuf,
        sha256: String,
        work: Map<String, Value>,
    },
    Unavailable(String),
}

#[derive(Default)]
pub(super) struct Run {
    pub(super) shots: BTreeMap<String, Shot>,
    pub(super) digests: BTreeMap<String, Value>,
    pub(super) unsettled: Vec<String>,
    pub(super) error: Option<String>,
}

fn launch_arguments(
    cell: &Cell,
    side: &Side,
    folder: &Path,
    gpu: &str,
) -> Result<Vec<String>, String> {
    let settings = side.tree.join(SETTINGS).display().to_string();
    let mut arguments = vec!["--settings".to_owned(), settings];
    arguments.extend(bench::arguments(
        &Scene::read(&side.tree, &cell.scene)?.options,
    ));
    for (option, value) in &cell.options {
        arguments.extend([format!("--{option}"), value.clone()]);
    }
    arguments.extend([
        "--workers".into(),
        "off".into(),
        "--gpu".into(),
        gpu.into(),
        "--dev".into(),
        "--dev-artifacts".into(),
        folder.display().to_string(),
    ]);
    Ok(arguments)
}

pub(super) fn run_cell(cell: &Cell, side: &Side, folder: &Path, gpu: &str) -> Run {
    let mut run = Run::default();
    if let Err(error) = fs::create_dir_all(folder) {
        run.error = Some(error.to_string());
        return run;
    }
    let log = folder.join("program.log");
    let launched = launch_arguments(cell, side, folder, gpu).and_then(|arguments| {
        Session::launch(side.program(&cell.project)?, &side.tree, &arguments, &log)
    });
    let mut session = match launched {
        Ok(session) => session,
        Err(error) => {
            run.error = Some(error);
            return run;
        }
    };
    if let Err(error) = drive(&mut session, cell, &log, &mut run) {
        run.error = Some(error);
    }
    session.close();
    run
}

fn drive(session: &mut Session, cell: &Cell, log: &Path, run: &mut Run) -> Result<(), String> {
    if let Some(environment) = &cell.environment {
        let mut args = json!({"merge": true});
        for (section, value) in environment.as_object().into_iter().flatten() {
            args[section] = value.clone();
        }
        session.ok(&named("control", "environment", args), log)?;
    }
    if cell.ticks > 0 {
        session.ok(&named("control", "step", json!({"ticks": cell.ticks})), log)?;
    }
    for view in &cell.views {
        match (&view.camera, view.name.as_str()) {
            (Some(camera), _) => {
                session.ok(&named("control", "camera", camera.clone()), log)?;
            }
            (None, "scene") => {}
            (None, name) => {
                session.ok(&named("control", "scene_view", json!({"name": name})), log)?;
            }
        }
        for capture in &cell.captures {
            let key = format!("{} {}", view.name, capture.label());
            if let Some(frame) = settling_frame(*capture)
                && let Some(changing) = settle(session, &frame, log)?
            {
                run.unsettled.push(format!("{key}: {changing}"));
            }
            let shot = shoot(session, *capture, log)?;
            run.shots.insert(key, shot);
        }
        let digest = session.request(&named("query", "digest", json!({})), log)?;
        let result = &digest["result"];
        run.digests.insert(
            view.name.clone(),
            if digest["ok"] == true {
                json!({
                    "hash": result["hash"],
                    "entities": result["entities"],
                    "components": result["components"],
                    "assets": result["assets"],
                })
            } else {
                Value::Null
            },
        );
    }
    Ok(())
}

fn settling_frame(capture: Capture) -> Option<Value> {
    match capture {
        Capture::Exact => Some(named("control", "render", json!({}))),
        Capture::Gpu(alpha) => Some(named("control", "present", json!({"alpha": alpha}))),
        Capture::Display(_) => None,
    }
}

fn settle(session: &mut Session, frame: &Value, log: &Path) -> Result<Option<String>, String> {
    let started = Instant::now();
    let (mut previous, mut quiet): (Option<Map<String, Value>>, usize) = (None, 0);
    for drawn in 1_u64.. {
        let result = session.ok(frame, log)?;
        let stats = &result["rendering"];
        let busy = stats["loading"] == true || stats["pipelines_compiling"] == true;
        let lag = stats["read_back_age"].as_u64();
        let stale = lag.is_some_and(|lag| lag >= drawn);
        let work = measure::work_counters(stats);
        quiet = if !busy && !stale && previous.as_ref() == Some(&work) {
            quiet + 1
        } else {
            0
        };
        let wanted = QUIET_RENDERS.max(usize::try_from(lag.unwrap_or(0)).unwrap_or(usize::MAX) + 1);
        if quiet >= wanted {
            break;
        }
        if started.elapsed() > SETTLE_LIMIT {
            let changing: Vec<String> = previous
                .iter()
                .flat_map(|previous| {
                    work.iter()
                        .filter(|(name, value)| previous.get(*name) != Some(value))
                        .map(|(name, value)| format!("{name} {value}"))
                })
                .chain(busy.then(|| "loading or compiling".to_owned()))
                .chain(stale.then(|| "read-backs older than settling".to_owned()))
                .collect();
            return Ok(Some(changing.join(", ")));
        }
        previous = Some(work);
        if busy {
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    for _ in 0..TAIL_RENDERS {
        session.ok(frame, log)?;
    }
    Ok(None)
}

fn shoot(session: &mut Session, capture: Capture, log: &Path) -> Result<Shot, String> {
    let result = match capture {
        Capture::Exact => session.ok(&json!({"verb": "capture"}), log)?,
        Capture::Gpu(alpha) => session.ok(&json!({"verb": "capture", "alpha": alpha}), log)?,
        Capture::Display(frames) => {
            let reply = session.request(
                &named(
                    "control",
                    "frames",
                    json!({"count": frames, "interval_ms": 1000.0 / 60.0, "capture": true}),
                ),
                log,
            )?;
            if reply["ok"] != true {
                return Ok(Shot::Unavailable(format!(
                    "the host ran no display frames: {}",
                    reply["error"]["message"].as_str().unwrap_or_default()
                )));
            }
            reply["result"].clone()
        }
    };
    let artifact = &result["artifact"];
    match (artifact["path"].as_str(), artifact["sha256"].as_str()) {
        (Some(path), Some(sha256)) => Ok(Shot::Taken {
            path: PathBuf::from(path),
            sha256: sha256.to_owned(),
            work: measure::work_counters(&result["rendering"]),
        }),
        _ => Err(format!("the capture named no file: {result}")),
    }
}
