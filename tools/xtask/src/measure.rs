//! `cargo xtask measure`: one measure task of the engine factory, run on the
//! native host and written as the factory's report (format version 2). The
//! command runs from the commit the task's tools come from (`tools_sha`):
//! its scenes, settings, catalog and this code. Only the engine under test
//! comes from each arm's commit: both are built the same way into their own
//! folders, then measured in rounds of one base and one candidate run whose
//! order alternates (base, candidate, candidate, base...), each run a fresh
//! process measured as the workload says (`engine_native`'s measured runs),
//! timed by the host and by this command's own clock from the host's marks,
//! until the task's clean pairs (both runs settled), the deadline, or a run
//! that never settles. A deep profile of each arm and the offline shader
//! report follow where the task wants them. `--extend` adds rounds to an
//! earlier report of the same task with the same builds.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::bench::{self, Assets, hex};

const FORMAT_VERSION: u64 = 2;
/// Clean pairs a task asks for unless it says.
const MIN_PAIRS: u64 = 8;
/// Unsettled runs of one arm in a row after which it never settles.
const NEVER_SETTLES: usize = 4;
/// Frames a deep profile records: about 120 KB of summary for Metro's.
const PROFILE_FRAMES: u32 = 60;
/// Frames the GPU may still draw when the next is submitted, in the
/// pipelined timing (`frame_ms_pipelined_mean`).
const FRAMES_IN_FLIGHT: u32 = 2;
/// What is measured, and how it is built: a profiling build, whose engine
/// times its GPU passes and CPU spans when asked and otherwise not.
const PACKAGE: &str = "stress-lab";
const PROGRAM: &str = "stress-lab-native";
const PROFILE: &str = "profiling";
const FEATURES: &[&str] = &["profiling"];
const TOOL: &str = "cargo xtask measure";

const USAGE: &str = "usage: cargo xtask measure --task TASK.json --profile PROFILE.json --out REPORT.json [--extend REPORT.json] [--formats FORMATS.json]";

struct Options {
    task: PathBuf,
    profile: PathBuf,
    out: PathBuf,
    extend: Option<PathBuf>,
    formats: Option<PathBuf>,
}

fn options(arguments: &[String]) -> Result<Options, String> {
    let mut found: BTreeMap<&str, PathBuf> = BTreeMap::new();
    let mut arguments = arguments.iter();
    while let Some(name) = arguments.next() {
        let key = match name.as_str() {
            "--task" | "--profile" | "--out" | "--extend" | "--formats" => &name[2..],
            _ => return Err(format!("unknown option {name}\n{USAGE}")),
        };
        let value = arguments
            .next()
            .ok_or_else(|| format!("{name} takes a path\n{USAGE}"))?;
        found.insert(key, PathBuf::from(value));
    }
    let mut required = |name: &str| {
        found
            .remove(name)
            .ok_or_else(|| format!("--{name} is required\n{USAGE}"))
    };
    Ok(Options {
        task: required("task")?,
        profile: required("profile")?,
        out: required("out")?,
        extend: found.remove("extend"),
        formats: found.remove("formats"),
    })
}

pub fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}

pub fn git(root: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// A program's output, trimmed, run in `folder`.
pub fn output(folder: &Path, program: &str, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(folder)
        .output()
        .map_err(|error| format!("cannot run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub fn file_hash(path: &Path) -> Result<String, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(hex(&Sha256::digest(&bytes)))
}

/// The time `time` in UTC, as ISO 8601 with milliseconds.
pub fn iso(time: SystemTime) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let (days, seconds) = (since.as_secs() / 86_400, since.as_secs() % 86_400);
    // Days to a civil date (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60,
        since.subsec_millis()
    )
}

/// The time an ISO 8601 date and time (`2026-10-11T02:00:00Z`, with a
/// fraction or an offset) names.
pub fn parse_iso(text: &str) -> Option<SystemTime> {
    let number = |range: std::ops::Range<usize>| text.get(range)?.parse::<i64>().ok();
    let (year, month, day) = (number(0..4)?, number(5..7)?, number(8..10)?);
    let (hour, minute, second) = (number(11..13)?, number(14..16)?, number(17..19)?);
    let mut rest = &text[19..];
    if let Some(fraction) = rest.strip_prefix('.') {
        rest = fraction.trim_start_matches(|c: char| c.is_ascii_digit());
    }
    let offset = match rest {
        "Z" | "z" => 0,
        _ => {
            let sign = match rest.get(..1)? {
                "+" => 1,
                "-" => -1,
                _ => return None,
            };
            sign * (rest.get(1..3)?.parse::<i64>().ok()? * 3600
                + rest.get(4..6)?.parse::<i64>().ok()? * 60)
        }
    };
    // A civil date to days (Howard Hinnant's algorithm).
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    let seconds = days * 86_400 + hour * 3600 + minute * 60 + second - offset;
    Some(UNIX_EPOCH + Duration::from_secs(u64::try_from(seconds).ok()?))
}

/// One arm built: its commit, program and build identity.
struct Arm {
    role: &'static str,
    sha: String,
    program: PathBuf,
    build: Value,
    shader_hash: String,
}

fn program_name() -> String {
    format!("{PROGRAM}{}", std::env::consts::EXE_SUFFIX)
}

/// Makes `sha` available, fetching it from `origin` where it is missing
/// (a pull request's merge commit through its ref, when the task names
/// one).
fn ensure_commit(root: &Path, sha: &str, pull: Option<u64>) -> Result<(), String> {
    let commit = format!("{sha}^{{commit}}");
    if git(root, &["cat-file", "-e", &commit]).is_ok() {
        return Ok(());
    }
    eprintln!("Fetching {sha}.");
    if let Some(pull) = pull {
        let reference = format!("pull/{pull}/merge");
        let _ = git(root, &["fetch", "--no-tags", "origin", &reference]);
    }
    if git(root, &["cat-file", "-e", &commit]).is_err() {
        git(root, &["fetch", "--no-tags", "origin", sha])?;
    }
    git(root, &["cat-file", "-e", &commit]).map(drop)
}

/// DXC's library (Windows): `DXC_PATH`, else `dxcompiler.dll` on the path,
/// else the one Firefox or the newest Chrome ships.
fn dxc() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("DXC_PATH").map(PathBuf::from) {
        return path.is_file().then_some(path);
    }
    let on_path = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|path| std::env::split_paths(&path).collect::<Vec<_>>())
        .map(|folder| folder.join("dxcompiler.dll"));
    let programs = std::env::var_os("ProgramFiles").map(PathBuf::from);
    let firefox = programs
        .iter()
        .map(|programs| programs.join("Mozilla Firefox").join("dxcompiler.dll"));
    let mut chrome: Vec<PathBuf> = programs
        .iter()
        .flat_map(|programs| {
            fs::read_dir(programs.join("Google/Chrome/Application"))
                .into_iter()
                .flatten()
        })
        .flatten()
        .map(|entry| entry.path().join("dxcompiler.dll"))
        .collect();
    chrome.sort();
    on_path
        .chain(firefox)
        .chain(chrome.into_iter().rev())
        .find(|file| file.is_file())
}

/// The source at `sha`, checked out under `target/measure/source/`.
pub fn checkout(root: &Path, sha: &str, pull: Option<u64>) -> Result<PathBuf, String> {
    ensure_commit(root, sha, pull)?;
    let source = root.join("target").join("measure").join("source").join(sha);
    if !source.join("Cargo.toml").is_file() {
        let path = source.display().to_string();
        git(
            root,
            &["worktree", "add", "--detach", "--force", &path, sha],
        )?;
    }
    Ok(source)
}

/// The arm `role` at `sha`: its source checked out under
/// `target/measure/source/`, built into `target/measure/builds/`, where a
/// build of the same commit is reused (its program unchanged).
fn prepare_arm(
    root: &Path,
    role: &'static str,
    sha: &str,
    pull: Option<u64>,
    backend: &str,
) -> Result<Arm, String> {
    let source = checkout(root, sha, pull)?;
    let folder = root.join("target").join("measure").join("builds").join(sha);
    let program = folder.join(program_name());
    let identity = folder.join("build.json");
    let kept = read_json(&identity).ok().filter(|build| {
        program.is_file() && file_hash(&program).ok().as_deref() == build["artifact_hash"].as_str()
    });
    let build = match kept {
        Some(build) => {
            eprintln!("Reusing the {role} build of {sha}.");
            build
        }
        None => build_arm(root, &source, &folder, sha, backend)?,
    };
    let mut build = build;
    // D3D12 compiles the engine's shaders with DXC, beside the program.
    if cfg!(windows) {
        match dxc() {
            Some(library) => {
                let beside = folder.join("dxcompiler.dll");
                if file_hash(&beside).ok() != file_hash(&library).ok() {
                    fs::copy(&library, &beside)
                        .map_err(|error| format!("cannot copy {}: {error}", library.display()))?;
                }
                build["toolchain"]["dxc"] = file_hash(&library)?.into();
            }
            None if backend == "dx12" => {
                return Err("D3D12 runs need DXC's dxcompiler.dll, which Firefox and Chrome ship: set DXC_PATH to it, or put it on the path".into());
            }
            None => {}
        }
    }
    let printed = output(root, &program.display().to_string(), &["--shader-hash"])?;
    let shader_hash = serde_json::from_str::<Value>(&printed)
        .ok()
        .and_then(|value| value["shader_hash"].as_str().map(str::to_owned))
        .ok_or_else(|| format!("the {role} program printed no shader hash: {printed}"))?;
    Ok(Arm {
        role,
        sha: sha.to_owned(),
        program,
        build,
        shader_hash,
    })
}

/// Builds the program from `source` into `folder`, with its symbols, and
/// records the build's identity there.
fn build_arm(
    root: &Path,
    source: &Path,
    folder: &Path,
    sha: &str,
    backend: &str,
) -> Result<Value, String> {
    eprintln!("Building {PACKAGE} at {sha}.");
    // One target folder for every arm: dependencies build once, and each
    // commit's own crates build beside the others'.
    let target = root.join("target").join("measure").join("target");
    let features = FEATURES.join(",");
    let started = Instant::now();
    let status = Command::new("cargo")
        .args(["build", "-p", PACKAGE, "--profile", PROFILE, "--locked"])
        .args(["--features", &features])
        .env("CARGO_TARGET_DIR", &target)
        .env("ENGINE_COMMIT", &sha[..12])
        .current_dir(source)
        .status()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    if !status.success() {
        return Err(format!("building {PACKAGE} at {sha} failed ({status})"));
    }
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
    fs::create_dir_all(folder).map_err(|error| error.to_string())?;
    let built = target.join(PROFILE);
    let program = folder.join(program_name());
    fs::copy(built.join(program_name()), &program)
        .map_err(|error| format!("cannot copy the program: {error}"))?;
    // Windows keeps the symbols beside, which deep profiles read.
    let symbols = format!("{}.pdb", PROGRAM.replace('-', "_"));
    if built.join(&symbols).is_file() {
        fs::copy(built.join(&symbols), folder.join(&symbols))
            .map_err(|error| format!("cannot copy the symbols: {error}"))?;
    }
    let build = json!({
        "sha": sha,
        "profile": PROFILE,
        "target": "native",
        "features": FEATURES,
        "toolchain": {
            "rustc": output(source, "rustc", &["-V"])?,
            "cargo": output(source, "cargo", &["-V"])?,
        },
        "env": {
            "RUSTFLAGS": std::env::var("RUSTFLAGS").unwrap_or_default(),
            "WGPU_BACKEND": backend,
        },
        "artifact_hash": file_hash(&program)?,
        "build_ms": build_ms,
    });
    fs::write(
        folder.join("build.json"),
        serde_json::to_string_pretty(&build).expect("builds serialize"),
    )
    .map_err(|error| error.to_string())?;
    Ok(build)
}

/// What a run of the program printed.
struct HostRun {
    /// When each mark line arrived.
    marks: BTreeMap<String, Instant>,
    /// The measured run's report line, or everything else it printed.
    report: Value,
    stdout: String,
    stderr: String,
    success: bool,
}

/// Runs `program` with `arguments` from the repository, reading its marks
/// as they come.
fn run_host(
    root: &Path,
    program: &Path,
    arguments: &[String],
    environment: &[(&str, String)],
) -> Result<HostRun, String> {
    let mut child = Command::new(program)
        .args(arguments)
        .envs(environment.iter().map(|(name, value)| (*name, value)))
        .current_dir(root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", program.display()))?;
    let mut errors = child.stderr.take().expect("piped");
    let stderr = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = errors.read_to_string(&mut text);
        text
    });
    let (mut marks, mut report, mut stdout) = (BTreeMap::new(), Value::Null, String::new());
    for line in BufReader::new(child.stdout.take().expect("piped")).lines() {
        let at = Instant::now();
        let line = line.map_err(|error| error.to_string())?;
        if let Ok(value) = serde_json::from_str::<Value>(&line) {
            if let Some(name) = value["mark"].as_str() {
                marks.insert(name.to_owned(), at);
                continue;
            }
            if value["report"].is_object() {
                report = value["report"].clone();
                continue;
            }
        }
        stdout.push_str(&line);
        stdout.push('\n');
    }
    let status = child.wait().map_err(|error| error.to_string())?;
    Ok(HostRun {
        marks,
        report,
        stdout,
        stderr: stderr.join().unwrap_or_default(),
        success: status.success(),
    })
}

/// The machine's power state now: `ac`, `battery` or `unknown`, and its
/// power plan's name.
fn power() -> (&'static str, String) {
    let line = |program: &str, arguments: &[&str]| {
        Command::new(program)
            .args(arguments)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    if cfg!(windows) {
        let source = line(
            "powershell",
            &[
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus",
            ],
        );
        let power = match source.as_deref() {
            Some("Online") => "ac",
            Some("Offline") => "battery",
            _ => "unknown",
        };
        // "Power Scheme GUID: ...  (Balanced)"
        let plan = line("powercfg", &["/getactivescheme"])
            .and_then(|text| {
                let start = text.rfind('(')?;
                Some(text[start + 1..].trim_end_matches(')').to_owned())
            })
            .unwrap_or_else(|| "unknown".into());
        (power, plan)
    } else if cfg!(target_os = "macos") {
        let power = match line("pmset", &["-g", "batt"]) {
            Some(text) if text.contains("'AC Power'") => "ac",
            Some(text) if text.contains("'Battery Power'") => "battery",
            _ => "unknown",
        };
        (power, "unknown".into())
    } else {
        let supplies = fs::read_dir("/sys/class/power_supply")
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path());
        let mut power = "unknown";
        for supply in supplies {
            let read = |name: &str| fs::read_to_string(supply.join(name)).unwrap_or_default();
            if read("type").trim() == "Mains" {
                power = if read("online").trim() == "1" {
                    "ac"
                } else {
                    "battery"
                };
            }
        }
        let plan = fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .map(|governor| governor.trim().to_owned())
            .unwrap_or_else(|_| "unknown".into());
        (power, plan)
    }
}

/// The smallest step of the clock runs are timed with, in microseconds.
fn timer_resolution_us() -> f64 {
    let mut smallest = Duration::MAX;
    let mut last = Instant::now();
    for _ in 0..100_000 {
        let now = Instant::now();
        let step = now - last;
        if step > Duration::ZERO {
            smallest = smallest.min(step);
        }
        last = now;
    }
    smallest.as_secs_f64() * 1e6
}

/// The frame statistics as work counters: dotted paths to numbers and
/// booleans, lists as their lengths, what has no value left out.
fn work_counters(stats: &Value) -> Map<String, Value> {
    fn flatten(value: &Value, path: String, out: &mut Map<String, Value>) {
        match value {
            Value::Object(fields) => {
                for (name, value) in fields {
                    let path = if path.is_empty() {
                        name.clone()
                    } else {
                        format!("{path}.{name}")
                    };
                    flatten(value, path, out);
                }
            }
            Value::Array(items) => {
                out.insert(path, items.len().into());
            }
            Value::Number(_) | Value::Bool(_) => {
                out.insert(path, value.clone());
            }
            Value::Null | Value::String(_) => {}
        }
    }
    let mut out = Map::new();
    flatten(stats, String::new(), &mut out);
    for name in bench::NOT_WORK {
        out.remove(*name);
    }
    out
}

/// Whether the catalog has `name` as the kind given (work, or anything
/// else when `work` is false).
fn catalogued(catalog: &[Value], name: &str, work: bool) -> bool {
    catalog
        .iter()
        .find(|entry| bench::covers(entry["name"].as_str().unwrap_or_default(), name))
        .is_some_and(|entry| (entry["kind"] == "work") == work)
}

/// One workload's measurement as the task gives it, with how to run it.
struct Workload {
    name: String,
    feature: String,
    options: Map<String, Value>,
    /// The host's arguments: the workload's, the options', the assets'.
    arguments: Vec<String>,
    method: Value,
    scene_file_hash: String,
    settings_hash: String,
    frames: u64,
    warmup_frames: u64,
}

fn workload(
    root: &Path,
    tools_sha: &str,
    entry: &Value,
    assets: &Assets,
    gpu: &str,
) -> Result<Workload, String> {
    let name = entry["name"].as_str().ok_or("a task workload has a name")?;
    let definition = &entry["workload"];
    if !definition.is_object() {
        return Err(format!(
            "workload {name}: the task carries no definition (`workload`: scene, settings, runtimes, settle, sample, capture)"
        ));
    }
    let hash = |folder: &str, key: &str| -> Result<String, String> {
        let file = definition[key]
            .as_str()
            .ok_or_else(|| format!("workload {name}: `{key}` names its file in bench/{folder}"))?;
        let object = format!("{tools_sha}:bench/{folder}/{file}.json");
        let hash = git(root, &["rev-parse", &object])?;
        let wanted = entry[format!(
            "{key}{}",
            if key == "scene" {
                "_file_hash"
            } else {
                "_hash"
            }
        )]
        .as_str();
        if wanted.is_some_and(|wanted| wanted != hash) {
            return Err(format!(
                "workload {name}: bench/{folder}/{file}.json at {tools_sha} is {hash}, the task says {}",
                wanted.unwrap_or_default()
            ));
        }
        Ok(hash)
    };
    let scene_file_hash = hash("scenes", "scene")?;
    let settings_hash = hash("settings", "settings")?;
    let mut arguments: Vec<String> = definition["runtimes"]["native"]["args"]
        .as_array()
        .ok_or_else(|| format!("workload {name}: runtimes.native.args lists the host's arguments"))?
        .iter()
        .map(|argument| argument.as_str().map(str::to_owned))
        .collect::<Option<_>>()
        .ok_or_else(|| format!("workload {name}: runtimes.native.args are strings"))?;
    let options = match &entry["options"] {
        Value::Object(options) => options.clone(),
        Value::Null => Map::new(),
        _ => return Err(format!("workload {name}: options is an object of switches")),
    };
    for (option, value) in &options {
        let value = match value {
            Value::String(text) => text.clone(),
            Value::Bool(true) => "on".into(),
            Value::Bool(false) => "off".into(),
            other => other.to_string(),
        };
        arguments.extend([format!("--{option}"), value]);
    }
    let folder = assets.folder.display().to_string();
    arguments.extend(["--assets".into(), folder, "--gpu".into(), gpu.into()]);
    let sample = &definition["sample"];
    let count = |key: &str| {
        sample[key]
            .as_u64()
            .ok_or_else(|| format!("workload {name}: sample.{key} is a count"))
    };
    Ok(Workload {
        name: name.to_owned(),
        feature: entry["feature"].as_str().unwrap_or("none").to_owned(),
        options,
        arguments,
        method: json!({
            "settle": definition["settle"],
            "sample": sample,
            "capture": definition["capture"],
            "frames_in_flight": FRAMES_IN_FLIGHT,
        }),
        scene_file_hash,
        settings_hash,
        frames: count("frames")?,
        warmup_frames: count("warmup_frames")?,
    })
}

/// What is the same for every run on this machine.
struct Machine {
    backend: String,
    timer_resolution_us: f64,
    generated_hash: String,
}

/// One measured run of `arm`, its files in `folder` as `stem`.
fn measure_run(
    root: &Path,
    arm: &Arm,
    workload: &Workload,
    machine: &Machine,
    catalog: &[Value],
    folder: &Path,
    stem: &str,
) -> Result<(Value, Vec<Value>), String> {
    let mut method = workload.method.clone();
    let capture = folder.join(format!("{stem}.png"));
    if method["capture"].is_object() {
        method["capture"]["path"] = capture.display().to_string().into();
    }
    let method_path = folder.join(format!("{stem}.method.json"));
    fs::write(&method_path, method.to_string()).map_err(|error| error.to_string())?;
    let mut arguments = workload.arguments.clone();
    arguments.extend(["--measure".into(), method_path.display().to_string()]);
    let (power, power_plan) = power();
    let started_at = iso(SystemTime::now());
    let environment = backend_environment(&machine.backend);
    let run = run_host(root, &arm.program, &arguments, &environment)?;
    let log = folder.join(format!("{stem}.log"));
    fs::write(&log, format!("{}\n{}", run.stdout, run.stderr))
        .map_err(|error| error.to_string())?;
    if !run.success || !run.report.is_object() {
        return Err(format!(
            "the {} run failed ({}): {}",
            arm.role,
            log.display(),
            run.stderr
                .lines()
                .rev()
                .take(3)
                .collect::<Vec<_>>()
                .join(" | ")
        ));
    }
    let report = &run.report;
    let frames = report["frames"].as_u64().unwrap_or(workload.frames);
    let timed = |start: &str, end: &str| -> Result<f64, String> {
        match (run.marks.get(start), run.marks.get(end)) {
            (Some(start), Some(end)) => {
                Ok((*end - *start).as_secs_f64() * 1000.0 / frames.max(1) as f64)
            }
            _ => Err(format!(
                "the {} run printed no {start} and {end} marks",
                arm.role
            )),
        }
    };
    let mut metrics = Map::new();
    for (name, value) in report["metrics"].as_object().into_iter().flatten() {
        if catalogued(catalog, name, false) {
            metrics.insert(name.clone(), value.clone());
        } else {
            eprintln!("Left out {name}: bench/metrics.json at the tools' commit lacks it.");
        }
    }
    metrics.insert(
        "wall:frame_ms_mean".into(),
        json!(timed("serialized-start", "serialized-end")?),
    );
    metrics.insert(
        "frame_ms_pipelined_mean".into(),
        json!(timed("pipelined-start", "pipelined-end")?),
    );
    let mut work = work_counters(&report["work"]);
    work.retain(|name, _| {
        let known = catalogued(catalog, name, true);
        if !known {
            eprintln!(
                "Left out work counter {name}: bench/metrics.json at the tools' commit lacks it."
            );
        }
        known
    });
    let held = report["settled"]["held"] == true;
    let conditions = json!({
        "paced": false,
        "vsync": false,
        "refresh_hz": "unknown",
        "power": power,
        "power_plan": power_plan,
        "gpu_clock_mhz": "unknown",
        "foreign_gpu_load_pct": "unknown",
        "foreign_cpu_load_pct": "unknown",
        "visible": "unknown",
        "timer_resolution_us": machine.timer_resolution_us,
        "gpu_timestamps": report["gpu_timestamps"].as_str().unwrap_or("unknown"),
        "settled": {
            "rule": "workload.settle",
            "after_ms": report["settled"]["after_ms"].as_f64().map_or(json!("unknown"), Value::from),
            "held": held,
        },
    });
    let record = json!({
        "arm": arm.role,
        "started_at": started_at,
        "metrics": metrics,
        "frames": frames,
        "generated_hash": machine.generated_hash,
        "work": work,
        "conditions": conditions,
        "capture": capture.display().to_string(),
    });
    let unavailable = report["unavailable"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|entry| {
            json!({
                "metric": entry["metric"],
                "reason": reason(entry["reason"].as_str().unwrap_or_default()),
            })
        })
        .collect();
    Ok((record, unavailable))
}

/// `text` within the 300 characters the factory's reasons hold.
fn reason(text: &str) -> String {
    const MOST: usize = 300;
    if text.chars().count() <= MOST {
        return text.to_owned();
    }
    text.chars().take(MOST - 3).collect::<String>() + "..."
}

/// The environment runs take: the graphics interface the setup names.
fn backend_environment(backend: &str) -> Vec<(&'static str, String)> {
    if backend.is_empty() {
        Vec::new()
    } else {
        vec![("WGPU_BACKEND", backend.to_owned())]
    }
}

/// A deep profile of `arm` on `workload` into `folder`.
fn profile_run(
    root: &Path,
    arm: &Arm,
    workload: &Workload,
    machine: &Machine,
    folder: &Path,
) -> Value {
    let into = folder.join(format!("profile-{}", arm.role));
    let mut arguments = workload.arguments.clone();
    arguments.extend([
        "--offscreen".into(),
        PROFILE_FRAMES.to_string(),
        "--profile".into(),
        into.display().to_string(),
    ]);
    let mut environment = backend_environment(&machine.backend);
    // The program's symbols are beside it, not where it was built.
    if let Some(symbols) = arm.program.parent() {
        environment.push(("_NT_ALTERNATE_SYMBOL_PATH", symbols.display().to_string()));
    }
    let unavailable = |why: String| json!({"arm": arm.role, "unavailable": reason(&why)});
    let run = match run_host(root, &arm.program, &arguments, &environment) {
        Ok(run) => run,
        Err(error) => return unavailable(error),
    };
    let printed: Value = match serde_json::from_str(&run.stdout) {
        Ok(printed) if run.success => printed,
        _ => {
            return unavailable(format!(
                "the profile run failed: {}",
                run.stderr
                    .lines()
                    .rev()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" | ")
            ));
        }
    };
    let profile = &printed["profile"];
    let file = |key: &str, kind: &str| {
        profile[key]
            .as_str()
            .map(|path| json!({"hash": path, "kind": kind, "tool": "engine_native --profile"}))
    };
    let files: Vec<Value> = [
        file("summary", "deep-summary"),
        file("firefox", "firefox-profile"),
    ]
    .into_iter()
    .flatten()
    .collect();
    if files.is_empty() {
        return unavailable("the profile run wrote no files".into());
    }
    json!({"arm": arm.role, "files": files, "complete": profile["complete"] == true})
}

/// The offline shader report of `arm`, from its own commit's tools.
fn shader_report(root: &Path, arm: &Arm, folder: &Path) -> Value {
    let source = root
        .join("target")
        .join("measure")
        .join("source")
        .join(&arm.sha);
    let into = folder.join(format!("shaders-{}", arm.role));
    let target = root.join("target").join("measure").join("target");
    let into_text = into.display().to_string();
    let status = Command::new("cargo")
        .args(["xtask", "gpu", "--report", &into_text])
        .env("CARGO_TARGET_DIR", &target)
        .current_dir(&source)
        .status();
    let report = into.join("report.json");
    match status {
        Ok(status) if status.success() && report.is_file() => {
            let written = read_json(&report).unwrap_or_default();
            let unavailable: Vec<Value> = written["amd_unavailable"]
                .as_str()
                .map(|reason| json!({"part": "amd", "reason": reason}))
                .into_iter()
                .collect();
            json!({
                "files": [{"hash": report.display().to_string(), "kind": "shader-report", "tool": "cargo xtask gpu --report"}],
                "unavailable": unavailable,
            })
        }
        outcome => json!({
            "files": [],
            "unavailable": [{"part": "report", "reason": reason(&format!("cargo xtask gpu --report at {} failed: {outcome:?}", arm.sha))}],
        }),
    }
}

/// The task's spec and id: the task as the factory gives it (`id`,
/// `spec`), or its spec alone with its `id`.
fn spec_of(task: &Value) -> Result<(u64, &Value), String> {
    let spec = if task["spec"].is_object() {
        &task["spec"]
    } else {
        task
    };
    let id = task["id"]
        .as_u64()
        .or_else(|| task["task"].as_u64())
        .ok_or("the task has no id")?;
    Ok((id, spec))
}

pub fn measure(root: &Path, arguments: &[String]) -> Result<(), String> {
    let options = options(arguments)?;
    let task = read_json(&options.task)?;
    let (task_id, spec) = spec_of(&task)?;
    let profile_start = read_json(&options.profile)?;
    let previous = options.extend.as_deref().map(read_json).transpose()?;
    // The tools are this checkout, which must be the task's tools commit.
    let tools_sha = git(root, &["rev-parse", "HEAD"])?;
    if !git(root, &["status", "--porcelain", "--untracked-files=no"])?.is_empty() {
        return Err("the tools checkout has uncommitted changes: measure from a clean checkout of the task's tools commit".into());
    }
    if let Some(wanted) = spec["tools_sha"]
        .as_str()
        .filter(|wanted| *wanted != tools_sha)
    {
        return Err(format!(
            "the tools come from {wanted}; this checkout is {tools_sha} (git checkout --detach {wanted})"
        ));
    }
    let setup = match &spec["setup"] {
        Value::String(id) => profile_start["setups"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|setup| setup["id"] == *id)
            .cloned()
            .ok_or_else(|| format!("setup {id} is not in the profile's setups"))?,
        setup if setup.is_object() => setup.clone(),
        _ => return Err("the task names no setup".into()),
    };
    if setup["runtime"] != "native" {
        return Err(format!(
            "setup {} runs {}: this command measures native setups",
            setup["id"], setup["runtime"]
        ));
    }
    let backend = match setup["api"].as_str().unwrap_or("unknown") {
        "d3d12" => "dx12",
        "vulkan" => "vulkan",
        "metal" => "metal",
        "gl" => "gl",
        _ => "",
    };
    let gpu_type = profile_start["machine"]["gpus"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|gpu| gpu["model"] == setup["gpu"])
        .and_then(|gpu| gpu["type"].as_str());
    let gpu = if gpu_type == Some("integrated") {
        "integrated"
    } else {
        "discrete"
    };
    let base_sha = spec["base_sha"]
        .as_str()
        .ok_or("the task has no base_sha")?;
    let candidate_sha = spec["sha"].as_str().ok_or("the task has no sha")?;
    let pull = spec["pr"].as_u64().or_else(|| task["pr"].as_u64());
    if previous.as_ref().is_some_and(|previous| {
        previous["task"] != task_id
            || previous["base_sha"] != base_sha
            || previous["candidate_sha"] != candidate_sha
            || previous["tools_sha"] != tools_sha.as_str()
    }) {
        return Err("--extend takes a report of this task, its commits and tools".into());
    }
    let deadline = spec["deadline"].as_str().and_then(parse_iso);
    let pairs = spec["pairs"].as_u64().unwrap_or(MIN_PAIRS);
    let want_profiles = spec["want"]["profiles"] == true;
    let catalog = bench::catalog(root)?;
    let assets = bench::assets(root)?;
    let base = prepare_arm(root, "base", base_sha, pull, backend)?;
    let candidate = prepare_arm(root, "candidate", candidate_sha, pull, backend)?;
    if let Some(previous) = &previous {
        for arm in [&base, &candidate] {
            if previous["builds"][arm.role]["artifact_hash"] != arm.build["artifact_hash"] {
                return Err(format!(
                    "the {} build differs from the report's: an extension measures the same binaries",
                    arm.role
                ));
            }
        }
    }
    let machine = Machine {
        backend: backend.to_owned(),
        timer_resolution_us: timer_resolution_us(),
        generated_hash: assets.generated_hash.clone(),
    };
    let runs_folder = root
        .join("target")
        .join("measure")
        .join("runs")
        .join(task_id.to_string());
    let mut incomplete = false;
    let mut workloads = Vec::new();
    for entry in spec["workloads"].as_array().into_iter().flatten() {
        let workload = workload(root, &tools_sha, entry, &assets, gpu)?;
        let folder = runs_folder.join(format!("{}-{}", workload.name, workload.feature));
        fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
        let earlier = previous.as_ref().and_then(|previous| {
            previous["workloads"].as_array()?.iter().find(|earlier| {
                earlier["name"] == workload.name.as_str()
                    && earlier["feature"] == workload.feature.as_str()
            })
        });
        let mut runs: Vec<Value> = earlier
            .and_then(|earlier| earlier["runs"].as_array().cloned())
            .unwrap_or_default();
        let mut unavailable: Vec<Value> = earlier
            .and_then(|earlier| earlier["unavailable"].as_array().cloned())
            .unwrap_or_default();
        let mut settle = earlier
            .and_then(|earlier| earlier["settle"].as_str())
            .unwrap_or("ok")
            .to_owned();
        let mut size = earlier.map_or(Value::Null, |earlier| earlier["method"]["size"].clone());
        let mut longest_round = Duration::ZERO;
        let clean = |runs: &[Value]| {
            runs.chunks(2)
                .filter(|round| {
                    round.len() == 2
                        && round
                            .iter()
                            .all(|run| run["conditions"]["settled"]["held"] == true)
                })
                .count() as u64
        };
        let unsettled_in_a_row = |runs: &[Value], role: &str| {
            runs.iter()
                .rev()
                .filter(|run| run["arm"] == role)
                .take_while(|run| run["conditions"]["settled"]["held"] != true)
                .count()
        };
        while settle == "ok" && clean(&runs) < pairs {
            if deadline
                .is_some_and(|deadline| SystemTime::now() + longest_round.mul_f64(1.2) > deadline)
            {
                incomplete = true;
                break;
            }
            let round = runs.len() / 2;
            let order = if round.is_multiple_of(2) {
                [&base, &candidate]
            } else {
                [&candidate, &base]
            };
            let started = Instant::now();
            let mut failed = None;
            let mut measured = Vec::new();
            for arm in order {
                let stem = format!("{:03}-{}", runs.len() + measured.len(), arm.role);
                eprintln!(
                    "{} ({}): round {}, {} run.",
                    workload.name,
                    workload.feature,
                    round + 1,
                    arm.role
                );
                match measure_run(root, arm, &workload, &machine, &catalog, &folder, &stem) {
                    Ok((record, missing)) => {
                        for entry in missing {
                            if !unavailable.contains(&entry) {
                                unavailable.push(entry);
                            }
                        }
                        measured.push(record);
                    }
                    Err(error) => {
                        failed = Some(error);
                        break;
                    }
                }
            }
            if let Some(error) = failed {
                // A round is one run of each arm: a lone run stays out.
                eprintln!("{error}");
                unavailable.push(json!({"metric": "*", "reason": reason(&error)}));
                break;
            }
            longest_round = longest_round.max(started.elapsed());
            if size.is_null() {
                size = json!(measured_size(&folder));
            }
            runs.extend(measured);
            if unsettled_in_a_row(&runs, "base") >= NEVER_SETTLES {
                settle = "base-never-settled".into();
            } else if unsettled_in_a_row(&runs, "candidate") >= NEVER_SETTLES {
                settle = "candidate-never-settled".into();
            }
        }
        let mut profiles: Vec<Value> = earlier
            .and_then(|earlier| earlier["profiles"].as_array().cloned())
            .unwrap_or_default();
        if want_profiles && workload.feature != "off" {
            for arm in [&base, &candidate] {
                if !profiles.iter().any(|profile| profile["arm"] == arm.role) {
                    eprintln!("{}: profiling the {} arm.", workload.name, arm.role);
                    profiles.push(profile_run(root, arm, &workload, &machine, &folder));
                }
            }
        }
        workloads.push(json!({
            "name": workload.name,
            "feature": workload.feature,
            "options": workload.options,
            "scene_file_hash": workload.scene_file_hash,
            "settings_hash": workload.settings_hash,
            "method": {
                "runtime": "native",
                "tool": TOOL,
                "frames": workload.frames,
                "warmup_frames": workload.warmup_frames,
                "paced": false,
                "size": if size.is_null() { json!("unknown") } else { size },
            },
            "settle": settle,
            "runs": runs,
            "profiles": profiles,
            "vendor": earlier.map_or(json!([]), |earlier| earlier["vendor"].clone()),
            "unavailable": unavailable,
        }));
    }
    let shaders_changed = base.shader_hash != candidate.shader_hash;
    let static_part = if spec["want"]["shader_report"] == true || shaders_changed {
        let folder = runs_folder.join("static");
        let _ = fs::create_dir_all(&folder);
        Some(json!({"shader_report": {
            "base": shader_report(root, &base, &folder),
            "candidate": shader_report(root, &candidate, &folder),
        }}))
    } else {
        previous
            .as_ref()
            .and_then(|previous| previous.get("static").cloned())
    };
    // The checks last: they build and serve the page, which measuring
    // must not share the machine with. An extension keeps those it has.
    let checks = match previous.as_ref().map(|previous| previous["checks"].clone()) {
        Some(Value::Array(checks)) if !checks.is_empty() => checks,
        _ => {
            let wanted = spec["checks"].as_array().cloned().unwrap_or_default();
            let (checks, cut) =
                crate::checks::run(root, &wanted, &runs_folder.join("checks"), deadline)?;
            incomplete |= cut;
            checks
        }
    };
    let profile_end = read_json(&options.profile)?;
    let mut report = json!({
        "format_version": FORMAT_VERSION,
        "kind": "measure",
        "task": task_id,
        "tools_sha": tools_sha,
        "profile_start": profile_start,
        "profile_end": profile_end,
        "setup": setup["id"],
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "builds": {"base": base.build, "candidate": candidate.build},
        "shader_hash": {"base": base.shader_hash, "candidate": candidate.shader_hash},
        "workloads": workloads,
        "checks": checks,
        "incomplete": incomplete,
    });
    if let Some(static_part) = static_part {
        report["static"] = static_part;
    }
    if let Some(formats) = &options.formats {
        let formats = read_json(formats)?;
        let schema = if formats["schemas"].is_object() {
            &formats["schemas"]["measure"]
        } else {
            &formats["measure"]
        };
        if !schema.is_object() {
            return Err("the formats file holds no measure schema (schemas.measure)".into());
        }
        let problems = crate::formats::problems(schema, &with_file_hashes(&report));
        if !problems.is_empty() {
            return Err(format!(
                "the report does not match the factory's format:\n  {}",
                problems.join("\n  ")
            ));
        }
        eprintln!("The report matches the factory's measure format.");
    }
    fs::write(
        &options.out,
        serde_json::to_string_pretty(&report).expect("reports serialize") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", options.out.display()))?;
    print_summary(&report);
    println!("Wrote {}.", options.out.display());
    Ok(())
}

/// The size of the first capture in `folder`, from its PNG header.
fn measured_size(folder: &Path) -> Option<[u32; 2]> {
    let mut captures: Vec<PathBuf> = fs::read_dir(folder)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "png"))
        .collect();
    captures.sort();
    let bytes = fs::read(captures.first()?).ok()?;
    let word = |at: usize| Some(u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
    Some([word(16)?, word(20)?])
}

/// The report with each local file it names as its SHA-256, as the
/// factory's command line sends it after uploading them.
fn with_file_hashes(report: &Value) -> Value {
    fn hashed(value: Option<&mut Value>) {
        let Some(value) = value else {
            return;
        };
        if let Some(hash) = value
            .as_str()
            .and_then(|path| file_hash(Path::new(path)).ok())
        {
            *value = hash.into();
        }
    }
    // Every list of files at `path` in `value`, where there is one.
    fn files<'a>(value: &'a mut Value, path: &[&str]) -> Vec<&'a mut Value> {
        let mut at = Some(value);
        for key in path {
            at = at.and_then(|value| value.get_mut(*key));
        }
        at.and_then(Value::as_array_mut)
            .map(|files| files.iter_mut().collect())
            .unwrap_or_default()
    }
    let mut report = report.clone();
    for workload in files(&mut report, &["workloads"]) {
        for run in files(workload, &["runs"]) {
            hashed(run.get_mut("capture"));
        }
        for key in ["profiles", "vendor"] {
            for entry in files(workload, &[key]) {
                for file in files(entry, &["files"]) {
                    hashed(file.get_mut("hash"));
                }
            }
        }
    }
    for check in files(&mut report, &["checks"]) {
        for attempt in files(check, &["attempts"]) {
            hashed(attempt.get_mut("log"));
        }
    }
    for arm in ["base", "candidate"] {
        for file in files(&mut report, &["static", "shader_report", arm, "files"]) {
            hashed(file.get_mut("hash"));
        }
    }
    report
}

/// Each workload's clean pairs and its arms' medians of the frame times.
fn print_summary(report: &Value) {
    for workload in report["workloads"].as_array().into_iter().flatten() {
        let runs = workload["runs"].as_array().cloned().unwrap_or_default();
        let median = |role: &str, metric: &str| {
            let mut values: Vec<f64> = runs
                .iter()
                .filter(|run| run["arm"] == role && run["conditions"]["settled"]["held"] == true)
                .filter_map(|run| run["metrics"][metric].as_f64())
                .collect();
            values.sort_by(f64::total_cmp);
            (!values.is_empty()).then(|| values[values.len() / 2])
        };
        let mut line = format!(
            "{} ({}): {} runs, settle {}",
            workload["name"].as_str().unwrap_or_default(),
            workload["feature"].as_str().unwrap_or_default(),
            runs.len(),
            workload["settle"].as_str().unwrap_or_default()
        );
        for metric in [
            "frame_ms_mean",
            "frame_ms_pipelined_mean",
            "wall:frame_ms_mean",
        ] {
            if let (Some(base), Some(candidate)) =
                (median("base", metric), median("candidate", metric))
            {
                line.push_str(&format!(
                    "; {metric} {base:.3} -> {candidate:.3} ({:+.1}%)",
                    (candidate / base - 1.0) * 100.0
                ));
            }
        }
        println!("{line}");
    }
}
