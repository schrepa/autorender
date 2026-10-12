//! The machine's "one heavy run at a time" lock, the one the smokes take
//! (`web/scripts/heavy-run.mjs`): anything that opens a browser, a native
//! engine window or a GPU host (measured runs, equivalence runs, smokes)
//! holds it while it runs, so runs never share the GPU and processors with
//! each other.
//!
//! The lock is the folder `ENGINE_HEAVY_LOCK_DIR`, else
//! `engine-heavy-run.lock` in the temporary folder Node uses, holding
//! `owner.json` (`{pid, label, cwd, command, started}`). A lock whose owner
//! process ended is stale and taken over. After taking it, a holder waits up
//! to 15 minutes while a headless Chrome of another program runs (the
//! factory's workers do not take the lock). A holder's children run with
//! `ENGINE_HEAVY_LOCK_HELD=1` ([`HELD`]) and pass straight through, so runs
//! nest.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};

/// The environment a holder's children run with: they hold the lock
/// through it.
pub const HELD: (&str, &str) = ("ENGINE_HEAVY_LOCK_HELD", "1");

/// How long a holder waits for another program's headless Chrome.
const FOREIGN_CHROME_WAIT: Duration = Duration::from_secs(15 * 60);
/// How long a lock folder may lack its owner file: a holder that stopped
/// between making the folder and writing the file.
const OWNERLESS: Duration = Duration::from_secs(60);

/// The lock, held until dropped: released where this process took it, left
/// alone where a parent holds it.
pub struct HeavyRun {
    taken: Option<PathBuf>,
}

impl HeavyRun {
    /// Takes the lock as `label`, waiting while another process holds it,
    /// then while another program's headless Chrome runs; nothing to take
    /// where a parent holds it (`ENGINE_HEAVY_LOCK_HELD=1`).
    pub fn hold(label: &str) -> Result<Self, String> {
        if std::env::var(HELD.0).is_ok_and(|held| held == HELD.1) {
            return Ok(Self { taken: None });
        }
        let folder = folder();
        let owner = folder.join("owner.json");
        let mut reported: Option<Instant> = None;
        loop {
            match fs::create_dir(&folder) {
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
                        serde_json::to_string_pretty(&record).expect("owners serialize"),
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
                    // Taken over only while the same stale owner holds it.
                    if holder(&owner).is_some_and(|again| again["pid"].as_u64() == Some(pid)) {
                        eprintln!(
                            "heavy-run: taking over a stale lock from pid {pid} ({})",
                            describe(held.as_ref())
                        );
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
            if reported.is_none_or(|at| at.elapsed() > Duration::from_secs(60)) {
                reported = Some(Instant::now());
                eprintln!(
                    "heavy-run: waiting for the heavy-run lock held by pid {} ({}) since {}",
                    held.as_ref()
                        .map_or(Value::Null, |held| held["pid"].clone()),
                    describe(held.as_ref()),
                    held.as_ref()
                        .and_then(|held| held["started"].as_str())
                        .unwrap_or("unknown")
                );
            }
            std::thread::sleep(Duration::from_secs(3));
        }
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
            let pids: Vec<String> = foreign.iter().map(u32::to_string).collect();
            if started.elapsed() > FOREIGN_CHROME_WAIT {
                eprintln!(
                    "heavy-run: foreign headless Chrome {} still running after 15 min; going ahead",
                    pids.join(",")
                );
                break;
            }
            if reported.is_none_or(|at| at.elapsed() > Duration::from_secs(60)) {
                reported = Some(Instant::now());
                eprintln!(
                    "heavy-run: waiting for foreign headless Chrome {} (factory?)",
                    pids.join(",")
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

/// The lock's folder: `ENGINE_HEAVY_LOCK_DIR`, else in the temporary folder
/// as Node finds it (`TEMP`, then `TMP` on Windows; `TMPDIR`, `TMP`, `TEMP`
/// elsewhere), which the scripts' lock uses.
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

/// The lock's owner as its file says, or nothing while it is written or
/// gone.
fn holder(owner: &std::path::Path) -> Option<Value> {
    fs::read_to_string(owner)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
}

/// What an owner runs, as messages name it.
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

/// Whether process `pid` runs; a process whose state cannot be read counts
/// as running.
fn alive(pid: u64) -> bool {
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

/// The process ids of headless Chrome browsers running now: their main
/// processes, not their helpers (`--type=`).
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
