//! One run of a cell on one side: the native host's development session
//! over standard input and output, settled and captured view by view.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::time::{Duration, Instant};

use serde_json::{Map, Value, json};

use super::matrix::{Capture, Cell, Workload};
use super::sides::Side;
use crate::measure;

/// The settings bundle every cell runs with: the size, the switches at
/// their defaults, no scanned rocks, no job cache.
const SETTINGS: &str = "bench/settings/default.json";

/// How long one command of a session may take: launching a workload and
/// compiling its pipelines take seconds.
const COMMAND_LIMIT: Duration = Duration::from_secs(240);

/// How long a view may take to settle: the highlands stream for most of a
/// minute.
const SETTLE_LIMIT: Duration = Duration::from_secs(300);

/// Renders in a row whose work counters hold still before a view counts as
/// settled.
const QUIET_RENDERS: usize = 3;

/// Frames after a view settled, so time-sliced work (reflection probe
/// faces, the sky's rows, kept shadow levels) completes its rounds.
const TAIL_RENDERS: usize = 30;

/// A development session of one side's program, over its standard input
/// and output.
struct Session {
    child: Child,
    input: ChildStdin,
    lines: Receiver<String>,
}

impl Session {
    /// Launches `side`'s program with `arguments`, writing what it prints on
    /// standard error into `log`, and waits for its ready line.
    fn launch(side: &Side, arguments: &[String], log: &Path) -> Result<Self, String> {
        let errors = fs::File::create(log).map_err(|error| error.to_string())?;
        let mut child = Command::new(&side.program)
            .args(arguments)
            .current_dir(&side.tree)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(errors)
            .spawn()
            .map_err(|error| format!("cannot start {}: {error}", side.program.display()))?;
        let input = child.stdin.take().expect("piped");
        let output = child.stdout.take().expect("piped");
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
        };
        let ready = session.next(log)?;
        if ready["ready"] != true {
            return Err(format!("the program did not say it is ready: {ready}"));
        }
        Ok(session)
    }

    /// The next reply the program printed, within the command limit.
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

    /// The program's reply to `command`.
    fn request(&mut self, command: &Value, log: &Path) -> Result<Value, String> {
        writeln!(self.input, "{command}")
            .and_then(|()| self.input.flush())
            .map_err(|error| format!("cannot send {command}: {error}"))?;
        self.next(log)
    }

    /// The result of `command`, which must succeed.
    fn ok(&mut self, command: &Value, log: &Path) -> Result<Value, String> {
        let reply = self.request(command, log)?;
        if reply["ok"] != true {
            return Err(format!(
                "{}: {}",
                command["op"].as_str().unwrap_or_default(),
                reply["error"]
            ));
        }
        Ok(reply["result"].clone())
    }

    /// Ends the session: its input closes, and the program ends with it.
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

/// One capture of a run: its file and SHA-256, and its frame's work
/// counters; or why the host could not take it.
#[derive(Clone)]
pub(super) enum Shot {
    Taken {
        path: PathBuf,
        sha256: String,
        work: Map<String, Value>,
    },
    Unavailable(String),
}

/// What one run of a cell on one side gave.
#[derive(Default)]
pub(super) struct Run {
    pub(super) shots: BTreeMap<String, Shot>,
    /// The engine's state digest after each view settled, by view; null
    /// where the host has none.
    pub(super) digests: BTreeMap<String, Value>,
    /// Views that did not settle, with what kept changing.
    pub(super) unsettled: Vec<String>,
    /// Why the run stopped, where it did.
    pub(super) error: Option<String>,
}

/// The program's options for `cell` on `side`, its captures into `folder`.
fn launch_arguments(cell: &Cell, side: &Side, folder: &Path, gpu: &str) -> Vec<String> {
    let path = |relative: &str| side.tree.join(relative).display().to_string();
    let mut arguments = vec!["--settings".to_owned(), path(SETTINGS)];
    match &cell.workload {
        Workload::Scene(scene) => {
            arguments.extend([
                "--scene".into(),
                path(&format!("bench/scenes/{scene}.json")),
            ]);
        }
        Workload::Source(source) => arguments.extend(["--source".into(), path(source)]),
    }
    for (option, value) in &cell.options {
        arguments.extend([format!("--{option}"), value.clone()]);
    }
    arguments.extend([
        // Jobs run in the engine: no worker's timing decides when their
        // results land.
        "--workers".into(),
        "off".into(),
        "--assets".into(),
        side.assets.display().to_string(),
        "--gpu".into(),
        gpu.into(),
        "--dev".into(),
        "--dev-artifacts".into(),
        folder.display().to_string(),
    ]);
    arguments
}

/// Runs `cell` once on `side`, its files into `folder`.
pub(super) fn run_cell(cell: &Cell, side: &Side, folder: &Path, gpu: &str) -> Run {
    let mut run = Run::default();
    if let Err(error) = fs::create_dir_all(folder) {
        run.error = Some(error.to_string());
        return run;
    }
    let log = folder.join("program.log");
    let arguments = launch_arguments(cell, side, folder, gpu);
    let mut session = match Session::launch(side, &arguments, &log) {
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

/// Applies `cell`'s environment, steps it, and settles and captures each
/// view into `run`.
fn drive(session: &mut Session, cell: &Cell, log: &Path, run: &mut Run) -> Result<(), String> {
    if let Some(environment) = &cell.environment {
        let mut command = json!({"op": "environment", "merge": true});
        for (section, value) in environment.as_object().into_iter().flatten() {
            command[section] = value.clone();
        }
        session.ok(&command, log)?;
    }
    if cell.ticks > 0 {
        session.ok(&json!({"op": "step", "ticks": cell.ticks}), log)?;
    }
    for view in &cell.views {
        match (&view.camera, view.name.as_str()) {
            (Some(camera), _) => {
                let mut command = camera.clone();
                command["op"] = "camera".into();
                session.ok(&command, log)?;
            }
            (None, "scene") => {}
            (None, name) => {
                session.ok(
                    &json!({"op": "game", "name": "view", "args": {"name": name}}),
                    log,
                )?;
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
        // The digest's own fields: its reply's result also carries the
        // host's observation, source and frame statistics.
        let digest = session.request(&json!({"op": "digest"}), log)?;
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

/// The frame a view settles with before `capture`: the kind of frame the
/// capture draws (GPU-presented frames cull, choose levels of detail and
/// request virtual texture and geometry pages on the GPU, read back a few
/// frames late); none for display frames, which run as they are.
fn settling_frame(capture: Capture) -> Option<Value> {
    match capture {
        Capture::Exact => Some(json!({"op": "render"})),
        Capture::Cpu(alpha) => Some(json!({"op": "present", "alpha": alpha})),
        Capture::Gpu(alpha) => Some(json!({"op": "present", "alpha": alpha, "gpu": true})),
        Capture::Display(_) => None,
    }
}

/// Draws `frame` until nothing loads or compiles, the counts GPU-culled
/// frames read back describe frames drawn since settling began, and the
/// work counters hold still for a few frames (more than the read-backs lag),
/// then a few more; what kept changing where it did not settle in time.
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

/// Takes `capture` of the settled view.
fn shoot(session: &mut Session, capture: Capture, log: &Path) -> Result<Shot, String> {
    let result = match capture {
        Capture::Exact => session.ok(&json!({"op": "capture"}), log)?,
        Capture::Cpu(alpha) => session.ok(&json!({"op": "capture", "alpha": alpha}), log)?,
        Capture::Gpu(alpha) => {
            session.ok(&json!({"op": "capture", "alpha": alpha, "gpu": true}), log)?
        }
        Capture::Display(frames) => {
            let reply = session.request(
                &json!({"op": "frames", "count": frames, "interval_ms": 1000.0 / 60.0, "capture": true}),
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
