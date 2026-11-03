use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitCode};

use serde_json::Value;
use toml_edit::{Document, Item};

const KINDS: &[&str] = &[
    "support",
    "core",
    "core-internal",
    "place",
    "brick",
    "list",
    "host",
    "game",
    "tool",
];

const CLOSED: &[(&str, &[&str])] = &[
    (
        "support",
        &[
            "engine_math",
            "engine_codec",
            "engine_heightfield",
            "engine_lodbuild",
            "engine_sources",
        ],
    ),
    (
        "core",
        &[
            "engine_ecs",
            "engine_place",
            "engine_core",
            "engine_render",
            "engine_assets",
            "engine_settings",
            "engine_scene",
            "engine_profile",
            "engine_panic",
        ],
    ),
    ("core-internal", &["engine_registry", "engine_textures"]),
    (
        "host",
        &[
            "engine",
            "engine_native",
            "engine_dev",
            "engine_worker",
            "engine_profiler",
        ],
    ),
];

const PLACE_EXTERNALS: &[&str] = &["glam", "bytemuck", "serde", "serde_json", "schemars"];

struct Member {
    name: String,
    at: String,
    dir: PathBuf,
    manifest: PathBuf,
    kind: Option<String>,
    cdylib: bool,
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
    git: bool,
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
    let workspace = read(&root.join("Cargo.toml"))?;
    let members = members(&root, &metadata);
    let folders: BTreeMap<&Path, &Member> = members
        .iter()
        .map(|member| (member.dir.as_path(), member))
        .collect();
    closed(&members, problems);
    for member in &members {
        check_member(&root, &workspace, member, &folders, problems)?;
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
    Member {
        name: package["name"].as_str().unwrap_or_default().to_owned(),
        at: relative(root, &dir),
        kind: package["metadata"]["engine"]["kind"]
            .as_str()
            .map(str::to_owned),
        cdylib,
        dependencies: package["dependencies"]
            .as_array()
            .into_iter()
            .flatten()
            .map(dependency)
            .collect(),
        dir,
        manifest,
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
        git: entry["source"]
            .as_str()
            .is_some_and(|source| source.starts_with("git+")),
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

fn check_member(
    root: &Path,
    workspace: &Item,
    member: &Member,
    folders: &BTreeMap<&Path, &Member>,
    problems: &mut Vec<String>,
) -> Result<(), String> {
    let at = member.label();
    let Some(kind) = member.kind.as_deref() else {
        problems.push(format!("{at}: no [package.metadata.engine] kind"));
        return Ok(());
    };
    if !KINDS.contains(&kind) {
        problems.push(format!("{at}: unknown kind {kind}"));
        return Ok(());
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
    if matches!(kind, "brick" | "place" | "list" | "game")
        && !forbids_unsafe(&read(&member.manifest)?, workspace)
    {
        problems.push(format!(
            "{at}: a {kind} crate needs [lints.rust] unsafe_code = \"forbid\""
        ));
    }
    let leaf_side = member.at.starts_with("crates/") || member.at.starts_with("tools/");
    let examples = root.join("examples");
    for dependency in &member.dependencies {
        let edge = format!("{at}: {} {}", dependency.table, dependency.key);
        let into_examples = dependency
            .path
            .as_ref()
            .is_some_and(|path| path.starts_with(&examples));
        if leaf_side && into_examples {
            problems.push(format!(
                "{edge}: nothing in crates/ or tools/ depends on examples/"
            ));
            continue;
        }
        if let Some(target) = dependency.path.as_deref().and_then(|dir| folders.get(dir)) {
            if !allowed(member, kind, target) {
                let target_kind = target.kind.as_deref().unwrap_or("no kind");
                problems.push(format!(
                    "{edge}: a {kind} crate may not depend on {} ({target_kind})",
                    target.name
                ));
            }
            continue;
        }
        if matches!(kind, "brick" | "place" | "list") {
            if dependency.path.is_some() {
                problems.push(format!(
                    "{edge}: a {kind} crate has no path dependency outside the workspace"
                ));
            }
            if dependency.git {
                problems.push(format!("{edge}: a {kind} crate has no git dependency"));
            }
        }
        if let Some(reason) = external(kind, &dependency.package) {
            problems.push(format!("{edge}: {reason}"));
        }
    }
    Ok(())
}

fn allowed(member: &Member, kind: &str, target: &Member) -> bool {
    let target_kind = target.kind.as_deref().unwrap_or_default();
    let name = target.name.as_str();
    match kind {
        "support" => target_kind == "support",
        "core" | "core-internal" => {
            matches!(target_kind, "support" | "core" | "core-internal")
        }
        "host" => matches!(target_kind, "support" | "core" | "core-internal" | "host"),
        "place" => target_kind == "support" || matches!(name, "engine_place" | "engine_ecs"),
        "brick" => {
            matches!(target_kind, "support" | "place")
                || matches!(name, "engine_ecs" | "engine_core" | "engine_render")
        }
        "list" => matches!(target_kind, "place" | "brick" | "list") || name == "engine_core",
        "game" => matches!(
            target_kind,
            "host" | "list" | "brick" | "place" | "core" | "support"
        ),
        "tool" => {
            !target.at.starts_with("examples/")
                || game(&member.at).is_some_and(|own| game(&target.at) == Some(own))
        }
        _ => false,
    }
}

fn game(at: &str) -> Option<&str> {
    let name = at
        .strip_prefix("examples/")?
        .split('/')
        .next()
        .filter(|name| !name.is_empty())?;
    at.get(.."examples/".len() + name.len())
}

fn external(kind: &str, package: &str) -> Option<String> {
    match kind {
        "list" => Some(format!(
            "a list crate depends on no external crate ({package})"
        )),
        "place" if !PLACE_EXTERNALS.contains(&package) => Some(format!(
            "a place crate depends only on {} among external crates ({package})",
            PLACE_EXTERNALS.join(", ")
        )),
        "support" if engine_only(package) => Some(format!("{package} is engine-only")),
        "brick" | "game" if engine_only(package) && !page_bindings(package) => {
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

fn forbids_unsafe(manifest: &Item, workspace: &Item) -> bool {
    let lints = manifest.get("lints");
    let inherited = lints
        .and_then(|lints| lints.get("workspace"))
        .and_then(Item::as_bool)
        == Some(true);
    let table = if inherited {
        workspace
            .get("workspace")
            .and_then(|table| table.get("lints"))
    } else {
        lints
    };
    let level = table
        .and_then(|lints| lints.get("rust"))
        .and_then(|rust| rust.get("unsafe_code"));
    let level = level.and_then(|level| level.get("level").unwrap_or(level).as_str());
    level == Some("forbid")
}

fn read(path: &Path) -> Result<Item, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Document::parse(text)
        .map(Document::into_item)
        .map_err(|error| format!("{}: {error}", path.display()))
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
