use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use engine_profiler::load::{Load, Monitor};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::bench::{self, Scene, hex};
use crate::process::Project;

const MIN_PAIRS: u64 = 8;
const NEVER_SETTLES: usize = 4;
const PROFILE_FRAMES: u32 = 60;
const FRAMES_IN_FLIGHT: u32 = 2;
const TOOLS_STATISTICS: &[&str] = &["frame_ms_p95"];
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

pub fn iso(time: SystemTime) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let (days, seconds) = (since.as_secs() / 86_400, since.as_secs() % 86_400);
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
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    let seconds = days * 86_400 + hour * 3600 + minute * 60 + second - offset;
    Some(UNIX_EPOCH + Duration::from_secs(u64::try_from(seconds).ok()?))
}

struct Build {
    sha: String,
    files: PathBuf,
    program: PathBuf,
    build: Value,
    shader_hash: String,
}

pub(crate) fn ensure_commit(root: &Path, sha: &str, pull: Option<u64>) -> Result<(), String> {
    let commit = format!("{sha}^{{commit}}");
    if git(root, &["cat-file", "-e", &commit]).is_ok() {
        return Ok(());
    }
    eprintln!("{}", json!({"event": "fetching", "commit": sha}));
    if let Some(pull) = pull {
        let reference = format!("pull/{pull}/head");
        let _ = git(root, &["fetch", "--no-tags", "origin", &reference]);
    }
    if git(root, &["cat-file", "-e", &commit]).is_err() {
        git(root, &["fetch", "--no-tags", "origin", sha])?;
    }
    git(root, &["cat-file", "-e", &commit]).map(drop)
}

pub fn commit_id<'a>(text: Option<&'a str>, what: &str) -> Result<&'a str, String> {
    text.filter(|text| text.len() == 40 && text.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or_else(|| format!("{what} is a full commit id (40 hexadecimal digits)"))
}

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

fn prepare_build(
    root: &Path,
    sha: &str,
    project: &str,
    pull: Option<u64>,
    backend: &str,
) -> Result<Build, String> {
    crate::checkouts::lease(root, sha)?;
    let project = Project::at(root, sha, project)?;
    let folder = root
        .join("target")
        .join("measure")
        .join("builds")
        .join(sha)
        .join(&project.package);
    let program = folder.join(project.program());
    let files = folder.join("tree");
    let kept = read_json(&folder.join("build.json")).ok().filter(|build| {
        program.is_file()
            && file_hash(&program).ok().as_deref() == build["artifact_hash"].as_str()
            && files.is_dir()
    });
    let mut build = match kept {
        Some(kept) => {
            let record = json!({
                "event": "build_reused",
                "package": project.package,
                "commit": sha,
            });
            eprintln!("{record}");
            kept
        }
        None => {
            let tree = crate::checkouts::reused(root, sha, pull)?;
            let build = build_arm(root, tree.path(), &folder, sha, &project, backend)?;
            if files.exists() {
                fs::remove_dir_all(&files)
                    .map_err(|error| format!("cannot remove {}: {error}", files.display()))?;
            }
            for part in crate::checkouts::RUN_FILES {
                crate::checkouts::copy_folder(&tree.path().join(part), &files.join(part))?;
            }
            build
        }
    };
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
        .ok_or_else(|| format!("the program of {sha} printed no shader hash: {printed}"))?;
    Ok(Build {
        sha: sha.to_owned(),
        files,
        program,
        build,
        shader_hash,
    })
}

pub(crate) fn build_arm(
    root: &Path,
    source: &Path,
    folder: &Path,
    sha: &str,
    project: &Project,
    backend: &str,
) -> Result<Value, String> {
    let package = project.package.as_str();
    eprintln!(
        "{}",
        json!({"event": "building", "package": package, "commit": sha})
    );
    let target = root.join("target").join("measure").join("target");
    let features = FEATURES.join(",");
    let started = Instant::now();
    let status = Command::new("cargo")
        .args(["build", "-p", package, "--bin", &project.native])
        .args(["--profile", PROFILE, "--locked", "--features", &features])
        .env("CARGO_TARGET_DIR", &target)
        .env("ENGINE_COMMIT", &sha[..12])
        .current_dir(source)
        .status()
        .map_err(|error| format!("cannot run cargo: {error}"))?;
    if !status.success() {
        return Err(format!("building {package} at {sha} failed ({status})"));
    }
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;
    fs::create_dir_all(folder).map_err(|error| error.to_string())?;
    let built = target.join(PROFILE);
    let program = folder.join(project.program());
    fs::copy(built.join(project.program()), &program)
        .map_err(|error| format!("cannot copy the program: {error}"))?;
    let symbols = format!("{}.pdb", project.native.replace('-', "_"));
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
        serde_json::to_string_pretty(&build).expect("build_serializes"),
    )
    .map_err(|error| error.to_string())?;
    Ok(build)
}

struct HostRun {
    marks: BTreeMap<String, Instant>,
    report: Value,
    load: Load,
    stdout: String,
    stderr: String,
    success: bool,
}

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
    let mut load = Monitor::new(child.id());
    let mut loaded = Load::default();
    let mut errors = child.stderr.take().expect("stdio_piped");
    let stderr = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = errors.read_to_string(&mut text);
        text
    });
    let (mut marks, mut report, mut stdout) = (BTreeMap::new(), Value::Null, String::new());
    for line in BufReader::new(child.stdout.take().expect("stdio_piped")).lines() {
        let at = Instant::now();
        let line = line.map_err(|error| error.to_string())?;
        if let Ok(value) = serde_json::from_str::<Value>(&line) {
            if let Some(name) = value["mark"].as_str() {
                match name {
                    "serialized-start" => load.start(),
                    "pipelined-end" => loaded = load.end(),
                    _ => {}
                }
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
        load: loaded,
        stdout,
        stderr: stderr.join().unwrap_or_default(),
        success: status.success(),
    })
}

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
    for name in engine_checks::bench::NOT_WORK {
        out.remove(*name);
    }
    out
}

fn catalogued(catalog: &[Value], name: &str, work: bool) -> bool {
    catalog
        .iter()
        .find(|entry| {
            engine_checks::bench::covers(entry["name"].as_str().unwrap_or_default(), name)
        })
        .is_some_and(|entry| (entry["kind"] == "work") == work)
}

struct Side<'a> {
    role: &'static str,
    build: &'a Build,
    arguments: Vec<String>,
}

struct Workload<'a> {
    name: String,
    feature: String,
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
            .expect("role_has_side")
    }
}

fn task_project(root: &Path, spec: &Value) -> Result<String, String> {
    let mut projects = BTreeSet::new();
    for entry in spec["workloads"].as_array().into_iter().flatten() {
        let scene = entry["workload"]["scene"].as_str().ok_or_else(|| {
            format!(
                "workload {}: `scene` names its file in bench/scenes",
                entry["name"]
            )
        })?;
        for role in ["base", "candidate"] {
            let at = format!("workload {}'s {role} arm's sha", entry["name"]);
            let sha = commit_id(entry["arms"][role]["sha"].as_str(), &at)?;
            projects.insert(Scene::at(root, sha, scene)?.project);
        }
    }
    let mut projects = projects.into_iter();
    match (projects.next(), projects.next()) {
        (Some(project), None) => Ok(project),
        (None, _) => Err("the task measures no workload: its scenes name the project".into()),
        (Some(first), Some(second)) => Err(format!(
            "the task's scenes name projects {first} and {second}: a task measures one project"
        )),
    }
}

fn scene_arguments(files: &Path, runtime: &[String]) -> Result<Vec<String>, String> {
    let mut arguments = Vec::new();
    let mut given = runtime.iter();
    while let Some(argument) = given.next() {
        if argument != "--scene" {
            arguments.push(argument.clone());
            continue;
        }
        let path = given.next().ok_or("--scene takes a bench scene file")?;
        let name = Path::new(path)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| format!("--scene {path} names no file"))?;
        arguments.extend(bench::arguments(&Scene::read(files, name)?.options));
    }
    Ok(arguments)
}

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
            "workload {name}: the task carries no definition (`workload`: scene, settings, runtimes, settle, sample, capture)"
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
                        "workload {name}: the {role} arm's bench/{folder}/{file}.json at {sha} is {hash}, the task says {wanted}"
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
        let mut arguments = scene_arguments(&build.files, &runtime)?;
        arguments.extend(bench::arguments(&options));
        arguments.extend(["--gpu".into(), gpu.into()]);
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

struct Machine {
    backend: String,
    timer_resolution_us: f64,
}

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
        &side.build.files,
        &side.build.program,
        &arguments,
        &environment,
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
            eprintln!(
                "{}",
                json!({"event": "metric_uncatalogued", "metric": name})
            );
        }
    }
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
                "{}",
                json!({"event": "work_counter_uncatalogued", "counter": name})
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

fn round(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

pub(crate) fn reason(text: &str) -> String {
    const MOST: usize = 300;
    if text.chars().count() <= MOST {
        return text.to_owned();
    }
    text.chars().take(MOST - 3).collect::<String>() + "..."
}

fn backend_environment(backend: &str) -> Vec<(&'static str, String)> {
    if backend.is_empty() {
        Vec::new()
    } else {
        vec![("WGPU_BACKEND", backend.to_owned())]
    }
}

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
    if let Some(symbols) = side.build.program.parent() {
        environment.push(("_NT_ALTERNATE_SYMBOL_PATH", symbols.display().to_string()));
    }
    let unavailable = |why: String| json!({"arm": side.role, "unavailable": reason(&why)});
    let run = match run_host(
        &side.build.files,
        &side.build.program,
        &arguments,
        &environment,
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

fn shader_report(root: &Path, role: &str, build: &Build, folder: &Path) -> Value {
    let into = folder.join(format!("shaders-{role}"));
    let target = root.join("target").join("measure").join("target");
    let into_text = into.display().to_string();
    let status = crate::checkouts::reused(root, &build.sha, None).and_then(|tree| {
        Command::new("cargo")
            .args(["xtask", "gpu", "--report", &into_text])
            .env("CARGO_TARGET_DIR", &target)
            .current_dir(tree.path())
            .status()
            .map_err(|error| format!("cannot run cargo: {error}"))
    });
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
            "unavailable": [{"part": "report", "reason": reason(&format!("cargo xtask gpu --report at {} failed: {outcome:?}", build.sha))}],
        }),
    }
}

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
    measure_with(root, arguments, None)
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
pub(crate) fn measure_with(
    root: &Path,
    arguments: &[String],
    tools: Option<&str>,
) -> Result<(), String> {
    let options = options(arguments)?;
    let task = read_json(&options.task)?;
    let (task_id, spec) = spec_of(&task)?;
    let profile_start = read_json(&options.profile)?;
    let previous = options.extend.as_deref().map(read_json).transpose()?;
    let tools_sha = match tools {
        Some(tools) => tools.to_owned(),
        None if git(root, &["status", "--porcelain", "--untracked-files=no"])?.is_empty() => {
            git(root, &["rev-parse", "HEAD"])?
        }
        None => return Err("the tools checkout has uncommitted changes: measure from a clean checkout of the task's tools commit".into()),
    };
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
    let base_sha = commit_id(spec["base_sha"].as_str(), "the task's base_sha")?;
    let candidate_sha = commit_id(spec["sha"].as_str(), "the task's sha")?;
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
    let catalog = engine_checks::bench::catalog(root)?;
    let project = task_project(root, spec)?;
    let mut builds: BTreeMap<String, Build> = BTreeMap::new();
    let mut commits = vec![base_sha, candidate_sha];
    commits.extend(arm_commits(spec)?);
    for sha in commits {
        if !builds.contains_key(sha) {
            let build = prepare_build(root, sha, &project, pull, backend)?;
            builds.insert(sha.to_owned(), build);
        }
    }
    crate::checkouts::prune(root);
    if let Some(previous) = &previous {
        for (role, sha) in [("base", base_sha), ("candidate", candidate_sha)] {
            if previous["builds"][role]["artifact_hash"] != builds[sha].build["artifact_hash"] {
                return Err(format!(
                    "the {role} build differs from the report's: an extension measures the same binaries"
                ));
            }
        }
    }
    let heavy = crate::heavy::HeavyRun::hold(&format!("cargo xtask measure (task {task_id})"))?;
    let machine = Machine {
        backend: backend.to_owned(),
        timer_resolution_us: timer_resolution_us(),
    };
    let runs_folder = root
        .join("target")
        .join("measure")
        .join("runs")
        .join(task_id.to_string());
    let mut incomplete = false;
    let mut workloads = Vec::new();
    for entry in spec["workloads"].as_array().into_iter().flatten() {
        let workload = workload(root, entry, &builds, gpu)?;
        let folder = runs_folder.join(format!("{}-{}", workload.name, workload.feature));
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
                let record = json!({
                    "event": "measure_run",
                    "workload": workload.name,
                    "feature": workload.feature,
                    "round": round + 1,
                    "role": role,
                });
                eprintln!("{record}");
                let side = workload.side(role);
                match measure_run(side, &workload, &machine, &catalog, &folder, &stem) {
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
            for side in &workload.sides {
                if !profiles.iter().any(|profile| profile["arm"] == side.role) {
                    let record = json!({
                        "event": "profiling_arm",
                        "workload": workload.name,
                        "role": side.role,
                    });
                    eprintln!("{record}");
                    profiles.push(profile_run(side, &machine, &folder));
                }
            }
        }
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
    }
    drop(heavy);
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
    fs::write(
        &options.out,
        serde_json::to_string_pretty(&report).expect("report_serializes") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", options.out.display()))?;
    print_summary(&report);
    let record = json!({"event": "written", "file": options.out.display().to_string()});
    println!("{record}");
    match &options.formats {
        Some(formats) => check_format(&report, formats, "measure"),
        None => Ok(()),
    }
}

pub fn check_format(report: &Value, formats: &Path, kind: &str) -> Result<(), String> {
    let formats = read_json(formats)?;
    let schema = if formats["schemas"].is_object() {
        &formats["schemas"][kind]
    } else {
        &formats[kind]
    };
    if !schema.is_object() {
        return Err(format!(
            "the formats file holds no {kind} schema (schemas.{kind})"
        ));
    }
    let problems = engine_checks::formats::problems(schema, &with_file_hashes(report));
    if !problems.is_empty() {
        return Err(format!(
            "the report does not match the factory's {kind} format:
  {}",
            problems.join(
                "
  "
            )
        ));
    }
    println!("{}", json!({"event": "format_matches", "format": kind}));
    Ok(())
}

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
        let mut metrics = Map::new();
        for metric in [
            "frame_ms_mean",
            "frame_ms_pipelined_mean",
            "wall:frame_ms_mean",
        ] {
            if let (Some(base), Some(candidate)) =
                (median("base", metric), median("candidate", metric))
            {
                let change = json!({
                    "base": base,
                    "candidate": candidate,
                    "change_percent": (candidate / base - 1.0) * 100.0,
                });
                metrics.insert(metric.to_owned(), change);
            }
        }
        let record = json!({
            "event": "workload_summary",
            "workload": workload["name"],
            "feature": workload["feature"],
            "runs": runs.len(),
            "settle": workload["settle"],
            "medians": metrics,
        });
        println!("{record}");
    }
}
