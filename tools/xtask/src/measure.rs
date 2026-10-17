//! `cargo xtask measure`: a local versioned comparison request, run on the
//! native host with caller-owned portable observations. The command
//! runs from the commit the request's tools come from (`tools_sha`): its
//! catalog and this code. Each workload compares two arms, each a commit
//! (both may be the same one), its switches, and the scene, settings and
//! generated files from that commit's own tree, so a change's scene edits
//! and switches reach its own build only. Every commit is built the same way
//! into its own folder, then the arms run in rounds of one base and one
//! candidate run whose order alternates (base, candidate, candidate,
//! base...), each run a fresh process measured as the workload says
//! (`engine_native`'s measured runs), timed by the host and by this
//! command's own clock from the host's marks, until the request's clean pairs
//! (both runs settled), the deadline, or a run that never settles. A deep
//! profile of each arm and the offline shader report follow where the task
//! wants them. `--extend` adds rounds to an earlier report of the same experiment
//! with the same builds. The runs and profiles hold the machine's heavy-run
//! lock ([`crate::heavy`]); the checks after them take it themselves.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::measure_host::run_host;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::bench::{self, hex};

/// Clean pairs a task asks for unless it says.
const MIN_PAIRS: u64 = 8;
/// Unsettled runs of one arm in a row after which it never settles.
const NEVER_SETTLES: usize = 4;
/// Frames a deep profile records: about 120 KB of summary for Metro's.
const PROFILE_FRAMES: u32 = 60;
/// Frames the GPU may still draw when the next is submitted, in the
/// pipelined timing (`frame_ms_pipelined_mean`).
const FRAMES_IN_FLIGHT: u32 = 2;
/// Metrics this command takes from the runs' raw times, never from what a
/// run's program computed: both arms' then come from the same code.
const TOOLS_STATISTICS: &[&str] = &["frame_ms_p95"];
/// What is measured, and how it is built: a profiling build, whose engine
/// times its GPU passes and CPU spans when asked and otherwise not.
const PACKAGE: &str = "stress-lab";
const PROGRAM: &str = "stress-lab-native";
const PROFILE: &str = "profiling";
const FEATURES: &[&str] = &["profiling"];
const TOOL: &str = "cargo xtask measure";

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
    xtask::request_contract::deadline(text)
}

/// One commit built and ready to measure: its checkout, its program and
/// build identity, its composed shaders' hash, and the generated files its
/// own tree makes.
struct Build {
    sha: String,
    source: PathBuf,
    program: PathBuf,
    build: Value,
    shader_hash: String,
    /// The folder holding the generated files (`--assets`), and their
    /// SHA-256.
    assets: String,
    generated_hash: String,
}

pub(crate) fn program_name() -> String {
    format!("{PROGRAM}{}", std::env::consts::EXE_SUFFIX)
}

/// Makes `sha` available, fetching it from `origin` where it is missing
/// (through its pull request's ref, when the request names one).
pub(crate) fn ensure_commit(root: &Path, sha: &str, pull: Option<u64>) -> Result<(), String> {
    let commit = format!("{sha}^{{commit}}");
    if git(root, &["cat-file", "-e", &commit]).is_ok() {
        return Ok(());
    }
    eprintln!("Fetching {sha}.");
    if let Some(pull) = pull {
        let reference = format!("pull/{pull}/head");
        let _ = git(root, &["fetch", "--no-tags", "origin", &reference]);
    }
    if git(root, &["cat-file", "-e", &commit]).is_err() {
        git(root, &["fetch", "--no-tags", "origin", sha])?;
    }
    git(root, &["cat-file", "-e", &commit]).map(drop)
}

/// `text` as a full commit id: 40 hexadecimal digits, as tasks name them.
pub fn commit_id<'a>(text: Option<&'a str>, what: &str) -> Result<&'a str, String> {
    text.filter(|text| text.len() == 40 && text.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or_else(|| format!("{what} is a full commit id (40 hexadecimal digits)"))
}

/// DXC's library (Windows): `DXC_PATH`, else `dxcompiler.dll` on the path,
/// else the one Firefox or the newest Chrome ships.
pub(crate) fn dxc() -> Option<PathBuf> {
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

/// Prepare a leased source, ask Cargo for the current executable, and retain
/// an immutable program bundle plus assets made by that source's generators.
fn prepare_build(
    root: &Path,
    sha: &str,
    pull: Option<u64>,
    backend: &str,
) -> Result<Build, String> {
    let source = crate::checkouts::checkout(root, sha, pull)?;
    let compiled = xtask::builds::native_binary(
        &source,
        &[
            "-p",
            PACKAGE,
            "--bin",
            PROGRAM,
            "--profile",
            PROFILE,
            "--features",
            &FEATURES.join(","),
            "--locked",
        ],
        "native",
        &[("ENGINE_COMMIT", &sha[..12])],
    )?;
    let libraries: Vec<_> = dxc().into_iter().collect();
    if cfg!(windows) && backend == "dx12" && libraries.is_empty() {
        return Err("D3D12 requires dxcompiler.dll; set DXC_PATH".into());
    }
    let bundle = xtask::builds::bundle(&source, &compiled, PROGRAM, &libraries)?;
    let program = bundle.data().join(program_name());
    let build = json!({"sha": sha, "profile": PROFILE, "target": "native", "features": FEATURES,
        "toolchain": {"rustc": output(&source, "rustc", &["-vV"])?, "cargo": output(&source, "cargo", &["-V"])?,
            "dxc": libraries.first().map(|p| file_hash(p)).transpose()?},
        "env": {"RUSTFLAGS": std::env::var("RUSTFLAGS").unwrap_or_default(), "WGPU_BACKEND": backend,
            "ENGINE_REPRODUCIBLE_LINK": if cfg!(target_env = "msvc") { "/Brepro" } else { "toolchain-default" }},
        "artifact_hash": file_hash(&program)?, "build_ms": compiled.elapsed_ms,
        "preparation": {"cargo_fresh": compiled.fresh(), "lock_ms": compiled.wait_ms},
        "bundle_manifest": bundle.manifest()?,
        "bundle_identity": xtask::builds::bundle_identity(&bundle, PROGRAM)?});
    xtask::cache::retain(bundle)?;
    drop(compiled);
    let printed = output(root, &program.display().to_string(), &["--shader-hash"])?;
    let shader_hash = serde_json::from_str::<Value>(&printed)
        .ok()
        .and_then(|value| value["shader_hash"].as_str().map(str::to_owned))
        .ok_or_else(|| format!("the program of {sha} printed no shader hash: {printed}"))?;
    let (assets, generated_hash) = generated(root, &source, sha)?;
    Ok(Build {
        sha: sha.to_owned(),
        source,
        program,
        build,
        shader_hash,
        assets,
        generated_hash,
    })
}

/// The generated files of the commit checked out in `source`, as its own
/// `cargo xtask bench assets` makes them: their folder and SHA-256.
pub(crate) fn generated(root: &Path, source: &Path, sha: &str) -> Result<(String, String), String> {
    eprintln!("Preparing generated files at {sha}.");
    let assets = bench::assets(source)?;
    bench::install(source, &assets.folder)?;
    let _ = root;
    Ok((assets.folder.display().to_string(), assets.generated_hash))
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
pub(crate) fn work_counters(stats: &Value) -> Map<String, Value> {
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

/// One side of a workload's comparison: `base` or `candidate`, its
/// commit's build, and the host's arguments: the workload's, its switches',
/// its commit's generated files'.
struct Side<'a> {
    role: &'static str,
    build: &'a Build,
    arguments: Vec<String>,
}

/// One workload's measurement as the request gives it, with how to run it.
struct Workload<'a> {
    name: String,
    feature: String,
    /// The task's arms, which the report repeats.
    arms: Value,
    sides: [Side<'a>; 2],
    method: Value,
    frames: u64,
    warmup_frames: u64,
}

impl Workload<'_> {
    fn side(&self, role: &str) -> &Side<'_> {
        self.sides
            .iter()
            .find(|side| side.role == role)
            .expect("both roles have a side")
    }
}

/// The commits a task's workloads build: their arms'.
fn arm_commits(spec: &Value) -> Result<Vec<&str>, String> {
    let mut commits = Vec::new();
    for entry in spec["workloads"].as_array().into_iter().flatten() {
        for role in ["base", "candidate"] {
            let at = format!("workload {}'s {role} arm's sha", entry["name"]);
            commits.push(commit_id(entry["arms"][role]["sha"].as_str(), &at)?);
        }
    }
    Ok(commits)
}

fn workload<'a>(
    root: &Path,
    entry: &Value,
    builds: &'a BTreeMap<String, Build>,
    gpu: &str,
) -> Result<Workload<'a>, String> {
    let name = entry["name"].as_str().ok_or("a task workload has a name")?;
    let definition = &entry["workload"];
    if !definition.is_object() {
        return Err(format!(
            "workload {name}: the request carries no definition (`workload`: scene, settings, runtimes, settle, sample, capture)"
        ));
    }
    let runtime: Vec<String> = definition["runtimes"]["native"]["args"]
        .as_array()
        .ok_or_else(|| format!("workload {name}: runtimes.native.args lists the host's arguments"))?
        .iter()
        .map(|argument| argument.as_str().map(str::to_owned))
        .collect::<Option<_>>()
        .ok_or_else(|| format!("workload {name}: runtimes.native.args are strings"))?;
    let mut arms = entry["arms"].clone();
    let side = |role: &'static str, arm: &mut Value| -> Result<Side<'a>, String> {
        let sha = commit_id(arm["sha"].as_str(), "an arm's sha")?.to_owned();
        let build = &builds[&sha];
        // The scene and settings the arm loads from its own commit's tree.
        for (folder, key, field) in [
            ("scenes", "scene", "scene_file_hash"),
            ("settings", "settings", "settings_hash"),
        ] {
            let file = definition[key].as_str().ok_or_else(|| {
                format!("workload {name}: `{key}` names its file in bench/{folder}")
            })?;
            let object = format!("{sha}:bench/{folder}/{file}.json");
            let hash = git(root, &["rev-parse", &object]).map_err(|_| {
                format!("workload {name}: bench/{folder}/{file}.json is not at {sha}")
            })?;
            match arm[field].as_str() {
                Some(wanted) if wanted != hash => {
                    return Err(format!(
                        "workload {name}: the {role} arm's bench/{folder}/{file}.json at {sha} is {hash}, the request says {wanted}"
                    ));
                }
                Some(_) => {}
                None => arm[field] = hash.into(),
            }
        }
        let options = match &arm["options"] {
            Value::Object(options) => options.clone(),
            Value::Null => Map::new(),
            _ => {
                return Err(format!(
                    "workload {name}: the {role} arm's options are switches by name"
                ));
            }
        };
        arm["options"] = Value::Object(options.clone());
        let mut arguments = runtime.clone();
        for (option, value) in &options {
            let value = match value {
                Value::String(text) => text.clone(),
                Value::Bool(true) => "on".into(),
                Value::Bool(false) => "off".into(),
                other => other.to_string(),
            };
            arguments.extend([format!("--{option}"), value]);
        }
        arguments.extend([
            "--assets".into(),
            build.assets.clone(),
            "--gpu".into(),
            gpu.into(),
        ]);
        Ok(Side {
            role,
            build,
            arguments,
        })
    };
    let base = side("base", &mut arms["base"])?;
    let candidate = side("candidate", &mut arms["candidate"])?;
    let sample = &definition["sample"];
    let count = |key: &str| {
        sample[key]
            .as_u64()
            .ok_or_else(|| format!("workload {name}: sample.{key} is a count"))
    };
    Ok(Workload {
        name: name.to_owned(),
        feature: entry["feature"].as_str().unwrap_or("none").to_owned(),
        arms,
        sides: [base, candidate],
        method: json!({
            "settle": definition["settle"],
            "sample": sample,
            "capture": definition["capture"],
            "frames_in_flight": FRAMES_IN_FLIGHT,
        }),
        frames: count("frames")?,
        warmup_frames: count("warmup_frames")?,
    })
}

/// What is the same for every run on this machine.
struct Machine {
    backend: String,
    timer_resolution_us: f64,
    deadline: Option<SystemTime>,
}

/// One measured run of `side`, from its commit's checkout, its files in
/// `folder` as `stem`.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn measure_run(
    side: &Side,
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
    let mut arguments = side.arguments.clone();
    arguments.extend(["--measure".into(), method_path.display().to_string()]);
    let (power, power_plan) = power();
    let started_at = iso(SystemTime::now());
    let environment = backend_environment(&machine.backend);
    let run = run_host(
        &side.build.source,
        &side.build.program,
        &arguments,
        &environment,
        &folder.join(format!("{stem}.raw")),
        machine.deadline,
    )?;
    let log = folder.join(format!("{stem}.log"));
    fs::write(&log, format!("{}\n{}", run.stdout, run.stderr))
        .map_err(|error| error.to_string())?;
    if !run.success || !run.report.is_object() {
        return Err(format!(
            "the {} run failed ({}): {}",
            side.role,
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
                side.role
            )),
        }
    };
    let mut metrics = Map::new();
    for (name, value) in report["metrics"].as_object().into_iter().flatten() {
        if TOOLS_STATISTICS.contains(&name.as_str()) {
            continue;
        }
        if catalogued(catalog, name, false) {
            metrics.insert(name.clone(), value.clone());
        } else {
            eprintln!("Left out {name}: bench/metrics.json at the tools' commit lacks it.");
        }
    }
    // The spread of the serialized frames' times, taken here from each one's
    // time, so both arms' come from the same code.
    let frame_ms: Option<Vec<f64>> = report["frame_ms"]
        .as_array()
        .map(|times| times.iter().filter_map(Value::as_f64).collect());
    let mut unavailable: Vec<Value> = Vec::new();
    match frame_ms.as_deref().and_then(engine::host::Distribution::of) {
        Some(spread) => {
            metrics.insert("frame_ms_p95".into(), json!(spread.p95));
        }
        None => unavailable.push(json!({
            "metric": "frame_ms_p95",
            "reason": reason("the run reported no frame times to take their spread from"),
        })),
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
    let known = |value: Option<f64>| value.map_or(json!("unknown"), |value| json!(round(value)));
    let load = &run.load;
    let gpu_clock_mhz = match load.gpu_clock_mhz {
        [None, None] => json!("unknown"),
        [start, end] => json!({ "start": known(start), "end": known(end) }),
    };
    let conditions = json!({
        "paced": false,
        "vsync": false,
        "refresh_hz": "unknown",
        "power": power,
        "power_plan": power_plan,
        "gpu_clock_mhz": gpu_clock_mhz,
        "foreign_gpu_load_pct": known(load.foreign_gpu_pct),
        "foreign_cpu_load_pct": known(load.foreign_cpu_pct),
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
        "arm": side.role,
        "started_at": started_at,
        "metrics": metrics,
        "frames": frames,
        "generated_hash": side.build.generated_hash,
        "work": work,
        "conditions": conditions,
        "capture": capture.display().to_string(),
    });
    unavailable.extend(
        report["unavailable"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|entry| {
                json!({
                    "metric": entry["metric"],
                    "reason": reason(entry["reason"].as_str().unwrap_or_default()),
                })
            }),
    );
    Ok((record, unavailable))
}

/// `value` to a tenth: what other programs load and clocks vary by more.
fn round(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

/// A compact diagnostic; the full error remains in the raw attempt log.
pub(crate) fn reason(text: &str) -> String {
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

/// A deep profile of `side` on its workload into `folder`.
fn profile_run(side: &Side, machine: &Machine, folder: &Path) -> Value {
    let into = folder.join(format!("profile-{}", side.role));
    let mut arguments = side.arguments.clone();
    arguments.extend([
        "--offscreen".into(),
        PROFILE_FRAMES.to_string(),
        "--profile".into(),
        into.display().to_string(),
    ]);
    let mut environment = backend_environment(&machine.backend);
    // The program's symbols are beside it, not where it was built.
    if let Some(symbols) = side.build.program.parent() {
        environment.push(("_NT_ALTERNATE_SYMBOL_PATH", symbols.display().to_string()));
    }
    let unavailable = |why: String| json!({"arm": side.role, "unavailable": reason(&why)});
    let run = match run_host(
        &side.build.source,
        &side.build.program,
        &arguments,
        &environment,
        &folder.join(format!("profile-{}.raw", side.role)),
        machine.deadline,
    ) {
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
    json!({"arm": side.role, "files": files, "complete": profile["complete"] == true})
}

/// The offline shader report of the `role` commit's `build`, from its own
/// tree's tools.
fn shader_report(root: &Path, role: &str, build: &Build, folder: &Path) -> Value {
    let into = folder.join(format!("shaders-{role}"));
    let into_text = into.display().to_string();
    let status = (|| -> Result<(), String> {
        let helper = xtask::builds::helper(&build.source)?;
        let mut command = Command::new(
            helper
                .data()
                .join(format!("engine-xtask{}", std::env::consts::EXE_SUFFIX)),
        );
        command
            .args(["gpu", "--report", &into_text, "--no-rga"])
            .current_dir(&build.source)
            .env("ENGINE_TOOL_WORKSPACE", &build.source);
        let output = xtask::jobs::run(&mut command, &into.join("generation"), &Default::default())?;
        if !output.status.success() {
            return Err(format!(
                "shader generation failed: {}",
                output.stderr.display()
            ));
        }
        let mut report = read_json(&into.join("report.json"))?;
        // SPIR-V and Naga observations come from each revision's own code;
        // the current compiler/cache layer analyzes those exact bytes.
        if let Some(analyzer) = xtask::analyzer::Analyzer::open(root, true)? {
            for entry in report["entries"]
                .as_array_mut()
                .ok_or("shader report has no entries")?
            {
                if let (Some(spirv), Some(stage), Some(name)) = (
                    entry["spirv"].as_str(),
                    entry["stage"].as_str(),
                    entry["entry_point"].as_str(),
                ) {
                    let stem = format!("{}.{}", entry["module"].as_str().unwrap_or("shader"), name);
                    entry["amd"] = analyzer.report(Path::new(spirv), name, stage, &into, &stem);
                }
            }
            report["amd"] = analyzer.description();
            report["amd_unavailable"] = Value::Null;
            report["cache"] = analyzer.telemetry();
        }
        xtask::files::atomic_json(&into.join("report.json"), &report)
    })();
    let report = into.join("report.json");
    match status {
        Ok(()) if report.is_file() => {
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
            "unavailable": [{"part": "report", "reason": reason(&format!("cargo xtask gpu --report at {} failed: {outcome:?}", build.sha))}],
        }),
    }
}

pub fn measure(root: &Path, arguments: &[String]) -> Result<(), String> {
    measure_with(root, arguments, None)
}

/// [`measure`], the tools taken to be commit `tools` where given (the
/// self-check's commit of the working tree, whose tools run), else this
/// checkout's HEAD, which must be clean.
#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(crate) fn measure_with(
    root: &Path,
    arguments: &[String],
    tools: Option<&str>,
) -> Result<(), String> {
    let execution = xtask::execution::Request::read(arguments, "measure")?;
    let spec = &execution.spec;
    let profile_start = spec["environment"].clone();
    let previous = &execution.previous;
    let tool_identity = xtask::execution::tool_identity(root)?;
    let tools_sha = tools
        .map(str::to_owned)
        .unwrap_or(git(root, &["rev-parse", "HEAD"])?);
    let setup = spec["setup"].clone();
    if previous
        .as_ref()
        .is_some_and(|report| report["tool_identity"] != tool_identity)
    {
        return Err(
            "continuation requires the same actual measurement executable and source state".into(),
        );
    }
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
    let base_sha = commit_id(spec["base_sha"].as_str(), "the request's base_sha")?;
    let candidate_sha = commit_id(spec["sha"].as_str(), "the request's sha")?;
    let pull = None;
    if previous.as_ref().is_some_and(|previous| {
        previous["base_sha"] != base_sha
            || previous["candidate_sha"] != candidate_sha
            || previous["tools_sha"] != tools_sha.as_str()
    }) {
        return Err("--extend takes a report of the same source revisions and tools".into());
    }
    let deadline = spec["deadline"].as_str().and_then(parse_iso);
    let pairs = spec["pairs"].as_u64().unwrap_or(MIN_PAIRS);
    let want_profiles = spec["want"]["profiles"] == true;
    let catalog = bench::catalog(root)?;
    // Every commit the request measures, built once: the report's two, and its
    // workloads' arms (both may be the same commit, switched off against on).
    if previous.is_none() {
        execution.checkpoint("preparation", &json!({"kind":"measure", "tool_identity":tool_identity, "tools_sha":tools_sha, "base_sha":base_sha, "candidate_sha":candidate_sha, "workloads":[], "checks":[], "incomplete":true}))?;
    }
    let mut builds: BTreeMap<String, Build> = BTreeMap::new();
    let mut commits = vec![base_sha, candidate_sha];
    commits.extend(arm_commits(spec)?);
    for sha in commits {
        if !builds.contains_key(sha) {
            let built = prepare_build(root, sha, pull, backend).inspect_err(|error| {
                if let Err(journal) =
                    execution.attempt("failed-build", &json!({"sha": sha, "error": error}))
                {
                    eprintln!("{journal}");
                }
            })?;
            builds.insert(sha.to_owned(), built);
        }
    }
    crate::checkouts::prune(root);
    let build_identities: BTreeMap<_, _> = builds.iter().map(|(sha, build)| (sha, json!({
        "artifact_hash": build.build["artifact_hash"], "bundle": build.build["bundle_identity"],
        "toolchain": build.build["toolchain"], "generated_hash": build.generated_hash
    }))).collect();
    let build_identities = json!(build_identities);
    if let Some(previous) = previous {
        xtask::execution::admit_builds(previous, &build_identities)?;
    }
    // The runs and profiles hold the machine's heavy-run lock: nothing else
    // draws on its GPU meanwhile.
    let lock_start = Instant::now();
    let heavy = crate::heavy::HeavyRun::hold_until(
        &format!("cargo xtask measure (run {})", execution.run_id),
        deadline,
    )?;
    let heavy_lock_ms = lock_start.elapsed().as_millis();
    let machine = Machine {
        backend: backend.to_owned(),
        timer_resolution_us: timer_resolution_us(),
        deadline,
    };
    let runs_folder = execution.folder.clone();
    let mut checkpoint = json!({"kind": "measure", "tools_sha": tools_sha, "tool_identity": tool_identity,
        "profile_start": profile_start, "setup": setup["id"], "base_sha": base_sha, "candidate_sha": candidate_sha,
        "builds": {"base": builds[base_sha].build, "candidate": builds[candidate_sha].build},
        "build_identities": build_identities, "preparation": {"heavy_lock_ms": heavy_lock_ms}, "workloads": [], "checks": [], "incomplete": true});
    execution.checkpoint("runtime", &checkpoint)?;
    let mut incomplete = false;
    let mut workloads = Vec::new();
    for entry in spec["workloads"].as_array().into_iter().flatten() {
        let workload = workload(root, entry, &builds, gpu)?;
        let folder = runs_folder.join(xtask::files::relative(&format!(
            "{}-{}",
            workload.name, workload.feature
        ))?);
        fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
        let earlier = previous.as_ref().and_then(|previous| {
            previous["workloads"].as_array()?.iter().find(|earlier| {
                earlier["name"] == workload.name.as_str()
                    && earlier["feature"] == workload.feature.as_str()
            })
        });
        if earlier.is_some_and(|earlier| earlier["arms"] != workload.arms) {
            return Err(format!(
                "workload {} ({}): the report it extends measured other arms",
                workload.name, workload.feature
            ));
        }
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
                ["base", "candidate"]
            } else {
                ["candidate", "base"]
            };
            let started = Instant::now();
            let mut failed = None;
            let mut measured = Vec::new();
            for role in order {
                let stem = format!("{:03}-{role}", runs.len() + measured.len());
                eprintln!(
                    "{} ({}): round {}, {role} run.",
                    workload.name,
                    workload.feature,
                    round + 1
                );
                let side = workload.side(role);
                match measure_run(side, &workload, &machine, &catalog, &folder, &stem) {
                    Ok((record, missing)) => {
                        for entry in missing {
                            if !unavailable.contains(&entry) {
                                unavailable.push(entry);
                            }
                        }
                        execution.attempt("measurement", &record)?;
                        measured.push(record);
                    }
                    Err(error) => {
                        execution.attempt(
                            "failed-measurement",
                            &json!({"arm": role, "error": error, "complete_pair": false}),
                        )?;
                        failed = Some(error);
                        break;
                    }
                }
            }
            if let Some(error) = failed {
                // A round is one run of each arm: a lone run stays out.
                incomplete = true;
                eprintln!("{error}");
                unavailable.push(json!({"metric": "*", "reason": reason(&error)}));
                break;
            }
            longest_round = longest_round.max(started.elapsed());
            if size.is_null() {
                size = json!(measured_size(&folder));
            }
            runs.extend(measured);
            checkpoint["workloads"] = json!(workloads);
            checkpoint["workloads"].as_array_mut().expect("workload array").push(json!({
                "name": workload.name, "feature": workload.feature, "arms": workload.arms,
                "method": {"runtime": "native", "tool": TOOL, "frames": workload.frames, "warmup_frames": workload.warmup_frames, "paced": false, "size": size},
                "runs": runs, "settle": settle, "profiles": [], "vendor": [], "unavailable": unavailable
            }));
            execution.checkpoint("runtime", &checkpoint)?;
            if unsettled_in_a_row(&runs, "base") >= NEVER_SETTLES {
                settle = "base-never-settled".into();
            } else if unsettled_in_a_row(&runs, "candidate") >= NEVER_SETTLES {
                settle = "candidate-never-settled".into();
            }
        }
        let mut profiles: Vec<Value> = earlier
            .and_then(|earlier| earlier["profiles"].as_array().cloned())
            .unwrap_or_default();
        workloads.push(json!({
            "name": workload.name,
            "feature": workload.feature,
            "arms": workload.arms,
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
        if want_profiles && workload.feature != "off" {
            for side in &workload.sides {
                if !profiles
                    .iter()
                    .any(|profile| profile["arm"] == side.role && profile["complete"] == true)
                {
                    eprintln!("{}: profiling the {} arm.", workload.name, side.role);
                    let profile = profile_run(side, &machine, &folder);
                    execution.attempt("profile", &profile)?;
                    profiles.retain(|old| old["arm"] != side.role);
                    profiles.push(profile);
                    workloads.last_mut().expect("current workload")["profiles"] = json!(profiles);
                    checkpoint["workloads"] = json!(workloads);
                    execution.checkpoint("profiling", &checkpoint)?;
                }
            }
        }
    }
    drop(heavy);
    checkpoint["workloads"] = json!(workloads);
    execution.checkpoint("static-analysis", &checkpoint)?;
    let (base, candidate) = (&builds[base_sha], &builds[candidate_sha]);
    let shaders_changed = base.shader_hash != candidate.shader_hash;
    let static_part = if spec["want"]["shader_report"] == true || shaders_changed {
        let folder = runs_folder.join("static");
        let _ = fs::create_dir_all(&folder);
        Some(json!({"shader_report": {
            "base": shader_report(root, "base", base, &folder),
            "candidate": shader_report(root, "candidate", candidate, &folder),
        }}))
    } else {
        previous
            .as_ref()
            .and_then(|previous| previous.get("static").cloned())
    };
    checkpoint["static"] = static_part.clone().unwrap_or(Value::Null);
    execution.checkpoint("checks", &checkpoint)?;
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
    let profile_end = spec["environment"].clone();
    let mut report = json!({
        "kind": "measure",
        "tool_identity": tool_identity,
        "build_identities": build_identities,
        "preparation": {"heavy_lock_ms": heavy_lock_ms},
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
    execution.checkpoint("complete", &report)?;
    print_summary(&report);
    println!("Wrote {}.", execution.out.display());
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

/// Each workload's clean pairs and its arms' medians of the frame times.
fn print_summary(report: &Value) {
    for workload in report["workloads"].as_array().into_iter().flatten() {
        let runs = workload["runs"].as_array().cloned().unwrap_or_default();
        let median = |role: &str, metric: &str| {
            let values: Vec<f64> = runs
                .iter()
                .filter(|run| run["arm"] == role && run["conditions"]["settled"]["held"] == true)
                .filter_map(|run| run["metrics"][metric].as_f64())
                .collect();
            engine::host::median(&values)
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
