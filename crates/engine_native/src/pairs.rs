use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use engine::{BrickList, Error, ErrorCode, Expected, Game, GameEntry};
use serde_json::{Value, json};

use crate::HostOptions;

const FRAMES: u32 = 3;

const SIZE: (u32, u32) = (320, 180);

const POLL: Duration = Duration::from_millis(20);

const SAID_LINES: usize = 4;

enum Outcome {
    Ran,
    Stopped(Value),
    Panicked(String),
    Hung { limit_ms: u64 },
    Unnamed { status: Option<i32>, said: Vec<String> },
}

impl Outcome {
    fn of(success: bool, status: Option<i32>, said: &str) -> Self {
        if let Some(report) = said
            .lines()
            .find_map(|line| line.strip_prefix(engine_panic::PANIC_PREFIX))
        {
            return Self::Panicked(report.to_owned());
        }
        if success {
            return Self::Ran;
        }
        let named = said
            .lines()
            .rev()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find(|record| record["code"].is_string());
        match named {
            Some(report) => Self::Stopped(report),
            None => {
                let lines: Vec<String> = said.lines().map(str::to_owned).collect();
                let kept = lines.len().saturating_sub(SAID_LINES);
                Self::Unnamed {
                    status,
                    said: lines[kept..].to_vec(),
                }
            }
        }
    }

    fn ok(&self) -> bool {
        match self {
            Self::Ran | Self::Stopped(_) => true,
            Self::Panicked(_) | Self::Hung { .. } | Self::Unnamed { .. } => false,
        }
    }

    fn record(&self) -> Value {
        match self {
            Self::Ran => json!({ "outcome": "ran" }),
            Self::Stopped(report) => json!({ "outcome": "stopped", "report": report }),
            Self::Panicked(report) => json!({ "outcome": "panicked", "report": report }),
            Self::Hung { limit_ms } => json!({ "outcome": "hung", "limit_ms": limit_ms }),
            Self::Unnamed { status, said } => {
                json!({ "outcome": "unnamed", "status": status, "said": said })
            }
        }
    }
}

struct Start {
    brick: &'static str,
    child: Child,
    said: Option<JoinHandle<String>>,
    started: Instant,
}

impl Start {
    fn spawn(program: &Path, brick: &'static str, options: &HostOptions) -> std::io::Result<Self> {
        let (width, height) = options.size.unwrap_or(SIZE);
        let mut command = Command::new(program);
        command
            .args(["--without", brick])
            .args(["--offscreen", &FRAMES.to_string()])
            .args(["--size", &format!("{width}x{height}")]);
        if let Some(scene) = &options.scene {
            command.arg("--scene").arg(scene);
        }
        if let Some(assets) = &options.assets {
            command.arg("--assets").arg(assets);
        }
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;
        let said = child.stderr.take().map(|mut stream| {
            std::thread::spawn(move || {
                let mut text = String::new();
                let _ = stream.read_to_string(&mut text);
                text
            })
        });
        Ok(Self {
            brick,
            child,
            said,
            started: Instant::now(),
        })
    }

    fn settle(&mut self, limit_ms: u64) -> Option<Outcome> {
        let status = match self.child.try_wait() {
            Ok(Some(status)) => status,
            Ok(None) if self.started.elapsed() < Duration::from_millis(limit_ms) => return None,
            Ok(None) => {
                let _ = self.child.kill();
                let _ = self.child.wait();
                return Some(Outcome::Hung { limit_ms });
            }
            Err(error) => {
                return Some(Outcome::Unnamed {
                    status: None,
                    said: vec![error.to_string()],
                });
            }
        };
        let said = self
            .said
            .take()
            .and_then(|reader| reader.join().ok())
            .unwrap_or_default();
        Some(Outcome::of(status.success(), status.code(), &said))
    }

    fn line(&self, title: &str, outcome: &Outcome) -> Value {
        let mut line = json!({
            "pairs": title,
            "without": self.brick,
            "ok": outcome.ok(),
            "ms": self.started.elapsed().as_secs_f64() * 1000.0,
        });
        if let (Some(fields), Value::Object(record)) = (line.as_object_mut(), outcome.record()) {
            fields.extend(record);
        }
        line
    }
}

fn refused(error: Error) -> ExitCode {
    crate::exit(Err(error))
}

fn alone(options: &HostOptions) -> Result<(), Error> {
    let others = [
        ("--offscreen", options.offscreen.is_some()),
        ("--dev", options.dev),
        ("--smoke", options.smoke.is_some()),
        ("--without", !options.without.is_empty()),
        ("--cook", options.cook.is_some()),
        ("--schemas", options.schemas),
        ("--pure", options.pure.is_some()),
    ];
    match others.iter().find(|(_, given)| *given) {
        Some((other, _)) => {
            Err(Error::new(ErrorCode::InvalidArguments, *other).expected(Expected::Absent))
        }
        None => Ok(()),
    }
}

pub fn of_game<G: Game>(options: &HostOptions, at_once: u32, game: GameEntry<G>) -> ExitCode {
    let Some(limit_ms) = options.limit_ms else {
        return refused(
            Error::new(ErrorCode::InvalidArguments, "--limit-ms").expected(Expected::Present),
        );
    };
    if let Err(error) = alone(options) {
        return refused(error);
    }
    let program: PathBuf = match std::env::current_exe() {
        Ok(program) => program,
        Err(error) => return refused(crate::host_error(ErrorCode::SourceIo, "--pairs", error)),
    };
    let bricks: Vec<&'static str> = (game.bricks)(BrickList::default())
        .listed()
        .iter()
        .map(|listed| listed.package())
        .collect();
    let mut waiting = bricks.iter().copied();
    let mut running: Vec<Start> = Vec::new();
    let mut failed: Vec<&'static str> = Vec::new();
    loop {
        while running.len() < at_once as usize {
            let Some(brick) = waiting.next() else {
                break;
            };
            match Start::spawn(&program, brick, options) {
                Ok(start) => running.push(start),
                Err(error) => {
                    let line = json!({
                        "pairs": game.title, "without": brick, "ok": false,
                        "outcome": "unnamed", "said": [error.to_string()],
                    });
                    println!("{line}");
                    failed.push(brick);
                }
            }
        }
        if running.is_empty() {
            break;
        }
        std::thread::sleep(POLL);
        let mut index = 0;
        while index < running.len() {
            let Some(outcome) = running[index].settle(limit_ms) else {
                index += 1;
                continue;
            };
            let start = running.swap_remove(index);
            println!("{}", start.line(game.title, &outcome));
            if !outcome.ok() {
                failed.push(start.brick);
            }
        }
    }
    println!(
        "{}",
        json!({ "pairs": game.title, "starts": bricks.len(), "failed": failed })
    );
    if failed.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(3)
    }
}
