use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};

pub const HELD: (&str, &str) = ("ENGINE_HEAVY_LOCK_HELD", "1");

const FOREIGN_CHROME_WAIT: Duration = Duration::from_secs(15 * 60);
pub(crate) const OWNERLESS: Duration = Duration::from_secs(60);

pub struct HeavyRun {
    taken: Option<PathBuf>,
}

impl HeavyRun {
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub fn hold(label: &str) -> Result<Self, String> {
        if std::env::var(HELD.0).is_ok_and(|held| held == HELD.1) {
            return Ok(Self { taken: None });
        }
        let folder = folder();
        let owner = folder.join("owner.json");
        let queue = queue_folder(&folder);
        let mut ticket: Option<Ticket> = None;
        let mut reported: Option<Instant> = None;
        loop {
            let line = queued(&queue);
            let turn = match &ticket {
                Some(Ticket(mine)) => {
                    line.first().map(String::as_str)
                        == mine.file_name().and_then(|name| name.to_str())
                }
                None => line.is_empty(),
            };
            let made = if turn {
                fs::create_dir(&folder)
            } else {
                Err(std::io::ErrorKind::AlreadyExists.into())
            };
            match made {
                Ok(()) => {
                    let record = json!({
                        "pid": std::process::id(),
                        "label": label,
                        "cwd": std::env::current_dir().map(|cwd| cwd.display().to_string()).unwrap_or_default(),
                        "command": std::env::args().collect::<Vec<_>>(),
                        "started": crate::measure::iso(SystemTime::now()),
                    });
                    fs::write(
                        &owner,
                        serde_json::to_string_pretty(&record).expect("owner_serializes"),
                    )
                    .map_err(|error| format!("cannot write {}: {error}", owner.display()))?;
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => {
                    return Err(format!(
                        "cannot make the heavy-run lock {}: {error}",
                        folder.display()
                    ));
                }
            }
            let held = holder(&owner);
            match held.as_ref().and_then(|held| held["pid"].as_u64()) {
                Some(pid) if !alive(pid) => {
                    if holder(&owner).is_some_and(|again| again["pid"].as_u64() == Some(pid)) {
                        let record = json!({
                            "event": "heavy_lock_taken_over",
                            "pid": pid,
                            "holder": describe(held.as_ref()),
                        });
                        eprintln!("{record}");
                        let _ = fs::remove_dir_all(&folder);
                    }
                    continue;
                }
                Some(_) => {}
                None => {
                    let age = fs::metadata(&folder)
                        .and_then(|metadata| metadata.modified())
                        .ok()
                        .and_then(|modified| modified.elapsed().ok());
                    if age.is_some_and(|age| age > OWNERLESS) {
                        let _ = fs::remove_dir_all(&folder);
                        continue;
                    }
                }
            }
            if ticket.is_none() {
                ticket = Some(Ticket::take(&queue, label)?);
            }
            if reported.is_none_or(|at| at.elapsed() > Duration::from_secs(60)) {
                reported = Some(Instant::now());
                match &held {
                    Some(held) => {
                        let record = json!({
                            "event": "heavy_lock_waiting",
                            "pid": held["pid"],
                            "holder": describe(Some(held)),
                            "since": held["started"],
                        });
                        eprintln!("{record}");
                    }
                    None => {
                        let mine = ticket
                            .as_ref()
                            .and_then(|Ticket(path)| path.file_name())
                            .and_then(|name| name.to_str());
                        let before = queued(&queue)
                            .iter()
                            .position(|name| Some(name.as_str()) == mine)
                            .unwrap_or(0);
                        let record = json!({
                            "event": "heavy_lock_queued",
                            "queued_before": before,
                        });
                        eprintln!("{record}");
                    }
                }
            }
            std::thread::sleep(Duration::from_secs(3));
        }
        drop(ticket);
        let run = Self {
            taken: Some(folder),
        };
        let started = Instant::now();
        let mut reported: Option<Instant> = None;
        loop {
            let foreign = headless_chromes();
            if foreign.is_empty() {
                break;
            }
            if started.elapsed() > FOREIGN_CHROME_WAIT {
                let record = json!({
                    "event": "foreign_chrome_ignored",
                    "pids": foreign,
                    "waited_s": FOREIGN_CHROME_WAIT.as_secs(),
                });
                eprintln!("{record}");
                break;
            }
            if reported.is_none_or(|at| at.elapsed() > Duration::from_secs(60)) {
                reported = Some(Instant::now());
                eprintln!(
                    "{}",
                    json!({"event": "foreign_chrome_waiting", "pids": foreign})
                );
            }
            std::thread::sleep(Duration::from_secs(5));
        }
        Ok(run)
    }
}

impl Drop for HeavyRun {
    fn drop(&mut self) {
        let Some(folder) = self.taken.take() else {
            return;
        };
        if holder(&folder.join("owner.json"))
            .is_some_and(|held| held["pid"].as_u64() == Some(u64::from(std::process::id())))
        {
            let _ = fs::remove_dir_all(&folder);
        }
    }
}

fn folder() -> PathBuf {
    if let Some(folder) = std::env::var_os("ENGINE_HEAVY_LOCK_DIR") {
        return PathBuf::from(folder);
    }
    let names: &[&str] = if cfg!(windows) {
        &["TEMP", "TMP"]
    } else {
        &["TMPDIR", "TMP", "TEMP"]
    };
    names
        .iter()
        .find_map(|name| std::env::var_os(name).filter(|value| !value.is_empty()))
        .map_or_else(std::env::temp_dir, PathBuf::from)
        .join("engine-heavy-run.lock")
}

struct Ticket(PathBuf);

impl Ticket {
    fn take(queue: &Path, label: &str) -> Result<Self, String> {
        fs::create_dir_all(queue).map_err(|error| {
            format!(
                "cannot make the heavy-run queue {}: {error}",
                queue.display()
            )
        })?;
        let since = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |since| since.as_millis());
        let path = queue.join(format!("{since:013}-{}.json", std::process::id()));
        let record = json!({"pid": std::process::id(), "label": label});
        fs::write(&path, record.to_string())
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        Ok(Self(path))
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn queue_folder(folder: &Path) -> PathBuf {
    let mut name = folder.as_os_str().to_owned();
    name.push(".queue");
    PathBuf::from(name)
}

fn queued(queue: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(queue) else {
        return Vec::new();
    };
    let mut line = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(pid) = name
            .strip_suffix(".json")
            .and_then(|stem| stem.split_once('-'))
            .and_then(|(_, pid)| pid.parse::<u64>().ok())
        else {
            continue;
        };
        if pid != u64::from(std::process::id()) && !alive(pid) {
            let _ = fs::remove_file(entry.path());
            continue;
        }
        line.push(name);
    }
    line.sort();
    line
}

fn holder(owner: &Path) -> Option<Value> {
    fs::read_to_string(owner)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
}

fn describe(held: Option<&Value>) -> String {
    let Some(held) = held else {
        return "unknown".into();
    };
    match held["label"].as_str().filter(|label| !label.is_empty()) {
        Some(label) => label.to_owned(),
        None => held["command"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" "),
    }
}

pub(crate) fn alive(pid: u64) -> bool {
    let listed = if cfg!(windows) {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
            .output()
    } else {
        Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "pid="])
            .output()
    };
    match listed {
        Ok(output) if cfg!(windows) => {
            !output.status.success()
                || String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\""))
        }
        Ok(output) => {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout).trim() == pid.to_string()
        }
        Err(_) => true,
    }
}

fn headless_chromes() -> Vec<u32> {
    if cfg!(windows) {
        let output = Command::new("powershell.exe")
            .args([
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | Where-Object { $_.CommandLine -match '--headless' -and $_.CommandLine -notmatch '--type=' } | ForEach-Object { $_.ProcessId }",
            ])
            .output();
        return output
            .map(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .split_whitespace()
                    .filter_map(|pid| pid.parse().ok())
                    .collect()
            })
            .unwrap_or_default();
    }
    let output = Command::new("ps").args(["-A", "-o", "pid=,args="]).output();
    output
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter(|line| {
                    let lower = line.to_lowercase();
                    lower.contains("chrom")
                        && line.contains("--headless")
                        && !line.contains("--type=")
                })
                .filter_map(|line| line.split_whitespace().next()?.parse().ok())
                .collect()
        })
        .unwrap_or_default()
}
