mod build_scripts;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode};

use serde_json::Value;

const KINDS: &[&str] = &["support", "core", "core-internal", "host", "game", "tool"];

const CORE_CRATES: &[&str] = &[
    "engine_ecs",
    "engine_place",
    "engine_gpu_bits",
    "engine_core",
    "engine_render",
    "engine_assets",
    "engine_settings",
    "engine_scene",
    "engine_profile",
    "engine_panic",
];

const CORE_INTERNAL_CRATES: &[&str] = &["engine_registry", "engine_textures"];

const CLOSED: &[(&str, &[&str])] = &[
    (
        "support",
        &[
            "engine_math",
            "engine_codec",
            "engine_sources",
            "engine_command_line",
            "engine_rig",
        ],
    ),
    ("core", CORE_CRATES),
    ("core-internal", CORE_INTERNAL_CRATES),
    (
        "host",
        &["engine", "engine_native", "engine_dev", "engine_profiler"],
    ),
];

struct Member {
    name: String,
    at: String,
    dir: PathBuf,
    kind: Option<String>,
    engine_rules: Value,
    cdylib: bool,
    build: Option<PathBuf>,
    dependencies: Vec<Dependency>,
}

impl Member {
    fn label(&self) -> String {
        format!("{}/Cargo.toml ({})", self.at, self.name)
    }
}

struct Dependency {
    table: String,
    key: String,
    package: String,
    path: Option<PathBuf>,
}

fn main() -> ExitCode {
    let root = match std::env::args_os().nth(1).map(PathBuf::from) {
        Some(path) if path.is_absolute() => normal(&path),
        Some(path) => normal(
            &std::env::current_dir()
                .map(|dir| dir.join(&path))
                .unwrap_or_else(|_| path.clone()),
        ),
        None => normal(&Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")),
    };
    let mut problems = Vec::new();
    let crates = match check(&root, &mut problems) {
        Ok(crates) => crates,
        Err(problem) => {
            problems.push(problem);
            0
        }
    };
    for problem in &problems {
        println!("{problem}");
    }
    if !problems.is_empty() {
        return ExitCode::FAILURE;
    }
    println!(
        "{}",
        serde_json::json!({"event": "kinds_hold", "crates": crates})
    );
    ExitCode::SUCCESS
}

fn check(root: &Path, problems: &mut Vec<String>) -> Result<usize, String> {
    let metadata = metadata(root)?;
    let root = metadata["workspace_root"]
        .as_str()
        .map(|folder| normal(Path::new(folder)))
        .ok_or("cargo metadata names no workspace root")?;
    let members = members(&root, &metadata);
    let folders: BTreeMap<&Path, &Member> = members
        .iter()
        .map(|member| (member.dir.as_path(), member))
        .collect();
    closed(&members, problems);
    implementors(&root, &members, problems)?;
    build_scripts_inside(&root, &members, &folders, problems)?;
    for member in &members {
        check_member(member, &folders, problems);
    }
    Ok(members.len())
}

fn metadata(root: &Path) -> Result<Value, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--locked"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("cargo metadata is not JSON: {error}"))
}

fn members(root: &Path, metadata: &Value) -> Vec<Member> {
    let ids: BTreeSet<&str> = metadata["workspace_members"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let mut members: Vec<Member> = metadata["packages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|package| package["id"].as_str().is_some_and(|id| ids.contains(id)))
        .map(|package| member(root, package))
        .collect();
    members.sort_by(|first, second| first.at.cmp(&second.at));
    members
}

fn member(root: &Path, package: &Value) -> Member {
    let manifest = normal(Path::new(
        package["manifest_path"].as_str().unwrap_or_default(),
    ));
    let dir = manifest.parent().map(Path::to_path_buf).unwrap_or_default();
    let cdylib = package["targets"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|target| target["crate_types"].as_array().into_iter().flatten())
        .any(|crate_type| crate_type == "cdylib");
    let build = package["targets"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|target| {
            target["kind"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|kind| kind == "custom-build")
        })
        .and_then(|target| target["src_path"].as_str())
        .map(|path| normal(Path::new(path)));
    Member {
        name: package["name"].as_str().unwrap_or_default().to_owned(),
        at: relative(root, &dir),
        kind: package["metadata"]["engine"]["kind"]
            .as_str()
            .map(str::to_owned),
        engine_rules: package["metadata"]["engine"]["engine_rules"].clone(),
        cdylib,
        build,
        dependencies: package["dependencies"]
            .as_array()
            .into_iter()
            .flatten()
            .map(dependency)
            .collect(),
        dir,
    }
}

fn dependency(entry: &Value) -> Dependency {
    let kind = entry["kind"].as_str();
    let table = match kind {
        Some("dev") => "dev-dependencies",
        Some("build") => "build-dependencies",
        Some(_) | None => "dependencies",
    };
    let table = match entry["target"].as_str() {
        Some(target) => format!("target.'{target}'.{table}"),
        None => table.to_owned(),
    };
    let package = entry["name"].as_str().unwrap_or_default().to_owned();
    let key = match entry["rename"].as_str() {
        Some(rename) => format!("{rename} ({package})"),
        None => package.clone(),
    };
    Dependency {
        table,
        key,
        package,
        path: entry["path"].as_str().map(|path| normal(Path::new(path))),
    }
}

fn closed(members: &[Member], problems: &mut Vec<String>) {
    for &(kind, names) in CLOSED {
        for name in names {
            match members.iter().find(|member| member.name == *name) {
                None => problems.push(format!(
                    "Cargo.toml: the kind {kind} names {name}, which is no workspace crate"
                )),
                Some(member) if member.kind.as_deref() != Some(kind) => problems.push(format!(
                    "{}: the kind {kind} names this crate, which declares {}",
                    member.label(),
                    member.kind.as_deref().unwrap_or("no kind")
                )),
                Some(_) => {}
            }
        }
    }
}

fn core(member: &Member) -> bool {
    matches!(member.kind.as_deref(), Some("core" | "core-internal"))
}

fn build_scripts_inside(
    root: &Path,
    members: &[Member],
    folders: &BTreeMap<&Path, &Member>,
    problems: &mut Vec<String>,
) -> Result<(), String> {
    for member in members.iter().filter(|member| core(member)) {
        for dependency in &member.dependencies {
            let workspace = dependency
                .path
                .as_deref()
                .and_then(|dir| folders.get(dir).copied());
            if let Some(target) = workspace
                && dependency.table.ends_with("build-dependencies")
            {
                problems.push(format!(
                    "{}: {} {}: a core crate's build script reads no file outside its own crate, and the lint cannot see what {} reads",
                    member.label(),
                    dependency.table,
                    dependency.key,
                    target.name
                ));
            }
        }
        if let Some(script) = &member.build {
            build_scripts::check(
                &member.label(),
                script,
                |path| relative(root, path),
                problems,
            )?;
        }
    }
    Ok(())
}

fn implementors(root: &Path, members: &[Member], problems: &mut Vec<String>) -> Result<(), String> {
    for member in members {
        let kind = member.kind.as_deref().unwrap_or("no kind");
        let source = member.dir.join("src");
        if kind == "game" || !source.is_dir() {
            continue;
        }
        let mut files = Vec::new();
        matching_files(&source, implements_brick, &mut files)?;
        for file in files {
            problems.push(format!(
                "{}: a {kind} crate implements Brick in {}; only a game may",
                member.label(),
                relative(root, &file)
            ));
        }
    }
    Ok(())
}

fn matching_files(
    folder: &Path,
    matches: fn(&str) -> bool,
    found: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let fail = |error: std::io::Error| format!("{}: {error}", folder.display());
    let mut paths = fs::read_dir(folder)
        .map_err(fail)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(fail)?;
    paths.sort();
    for path in paths {
        if path.is_dir() {
            matching_files(&path, matches, found)?;
        } else if path.extension().is_some_and(|extension| extension == "rs")
            && fs::read_to_string(&path)
                .map_err(|error| format!("{}: {error}", path.display()))?
                .lines()
                .any(matches)
        {
            found.push(path);
        }
    }
    Ok(())
}

fn implements_brick(line: &str) -> bool {
    let mut words = line.split_whitespace();
    words.next() == Some("impl")
        && words.next().is_some_and(|path| {
            !path.starts_with("$crate::") && path.rsplit("::").next() == Some("Brick")
        })
        && words.next() == Some("for")
}

fn check_member(member: &Member, folders: &BTreeMap<&Path, &Member>, problems: &mut Vec<String>) {
    let at = member.label();
    let Some(kind) = member.kind.as_deref() else {
        problems.push(format!("{at}: no [package.metadata.engine] kind"));
        return;
    };
    if !KINDS.contains(&kind) {
        problems.push(format!("{at}: unknown kind {kind}"));
        return;
    }
    if let Some((_, names)) = CLOSED.iter().find(|(closed, _)| *closed == kind)
        && !names.contains(&member.name.as_str())
    {
        problems.push(format!(
            "{at}: the kind {kind} is closed and does not name this crate"
        ));
    }
    if member.cdylib && kind != "game" {
        problems.push(format!("{at}: only game crates build a cdylib"));
    }
    if !member.engine_rules.is_null() && (kind != "game" || member.engine_rules != true) {
        problems.push(format!(
            "{at}: engine_rules is true on a game crate or absent"
        ));
    }
    for dependency in &member.dependencies {
        let edge = format!("{at}: {} {}", dependency.table, dependency.key);
        if let Some(target) = dependency.path.as_deref().and_then(|dir| folders.get(dir)) {
            if !allowed(kind, target) {
                let target_kind = target.kind.as_deref().unwrap_or("no kind");
                problems.push(format!(
                    "{edge}: a {kind} crate may not depend on {} ({target_kind})",
                    target.name
                ));
            }
            continue;
        }
        if let Some(reason) = external(kind, &dependency.package) {
            problems.push(format!("{edge}: {reason}"));
        }
    }
}

fn allowed(kind: &str, target: &Member) -> bool {
    let target_kind = target.kind.as_deref().unwrap_or_default();
    match kind {
        "support" => target_kind == "support",
        "core" | "core-internal" => {
            matches!(target_kind, "support" | "core" | "core-internal")
        }
        "host" => matches!(target_kind, "support" | "core" | "core-internal" | "host"),
        "game" => matches!(target_kind, "host" | "core" | "support"),
        "tool" => true,
        _ => false,
    }
}

fn external(kind: &str, package: &str) -> Option<String> {
    match kind {
        "support" if engine_only(package) => Some(format!("{package} is engine-only")),
        "game" if engine_only(package) && !page_bindings(package) => {
            Some(format!("{package} is engine-only"))
        }
        _ => None,
    }
}

fn engine_only(package: &str) -> bool {
    package == "bevy"
        || package.starts_with("bevy_")
        || package.starts_with("wgpu")
        || package == "naga"
        || package == "wesl"
        || page_bindings(package)
}

fn page_bindings(package: &str) -> bool {
    package.starts_with("wasm-bindgen") || package == "web-sys" || package == "js-sys"
}

fn normal(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}
