use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

use toml_edit::{Document, Item};

const KINDS: &[&str] = &["support", "core", "core-internal", "place", "brick", "list", "host", "app", "tool"];

const CLOSED: &[(&str, &[&str])] = &[
    ("support", &["engine_math", "engine_codec", "engine_heightfield", "engine_lodbuild", "engine_sources"]),
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
    ("host", &["engine", "engine_native", "engine_dev", "engine_worker"]),
    ("tool", &["engine_checks", "engine_profiler"]),
];

const PLACE_EXTERNALS: &[&str] = &["glam", "bytemuck", "serde", "serde_json"];

const TABLES: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies", "dev_dependencies", "build_dependencies"];

struct Member {
    name: String,
    at: String,
    dir: PathBuf,
    kind: Option<String>,
    manifest: Item,
}

struct Dependency {
    table: String,
    key: String,
    package: String,
    path: Option<PathBuf>,
    git: bool,
    unresolved: bool,
}

fn main() -> ExitCode {
    let root = match std::env::args_os().nth(1).map(PathBuf::from) {
        Some(path) if path.is_absolute() => normal(&path),
        Some(path) => normal(&std::env::current_dir().map(|dir| dir.join(&path)).unwrap_or_else(|_| path.clone())),
        None => normal(&Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")),
    };
    let mut problems = Vec::new();
    if let Err(problem) = check(&root, &mut problems) {
        problems.push(problem);
    }
    for problem in &problems {
        println!("{problem}");
    }
    if problems.is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}

fn check(root: &Path, problems: &mut Vec<String>) -> Result<(), String> {
    let workspace = read(&root.join("Cargo.toml"))?;
    let members = members(root, &workspace, problems)?;
    let by_name: BTreeMap<&str, &Member> = members.iter().map(|member| (member.name.as_str(), member)).collect();
    for member in &members {
        check_member(root, &workspace, member, &by_name, problems);
    }
    Ok(())
}

fn members(root: &Path, workspace: &Item, problems: &mut Vec<String>) -> Result<Vec<Member>, String> {
    let table = workspace.get("workspace").ok_or_else(|| format!("{}: no [workspace] table", root.join("Cargo.toml").display()))?;
    let excluded: Vec<PathBuf> = strings(table.get("exclude")).iter().map(|entry| normal(&root.join(entry))).collect();
    let mut dirs = Vec::new();
    for pattern in strings(table.get("members")) {
        if let Some(parent) = pattern.strip_suffix("/*") {
            let Ok(read) = fs::read_dir(root.join(parent)) else {
                problems.push(format!("Cargo.toml: the members entry {pattern} names no folder"));
                continue;
            };
            let mut found: Vec<PathBuf> = read.flatten().map(|entry| entry.path()).filter(|path| path.join("Cargo.toml").is_file()).collect();
            found.sort();
            dirs.extend(found);
        } else {
            dirs.push(root.join(&pattern));
        }
    }
    let mut out = Vec::new();
    for dir in dirs {
        let dir = normal(&dir);
        if excluded.contains(&dir) {
            continue;
        }
        let at = relative(root, &dir);
        let manifest = match read(&dir.join("Cargo.toml")) {
            Ok(manifest) => manifest,
            Err(problem) => {
                problems.push(problem);
                continue;
            }
        };
        let Some(name) = manifest.get("package").and_then(|package| package.get("name")).and_then(Item::as_str).map(str::to_owned) else {
            problems.push(format!("{at}: no [package] name"));
            continue;
        };
        let kind = manifest
            .get("package")
            .and_then(|package| package.get("metadata"))
            .and_then(|metadata| metadata.get("engine"))
            .and_then(|engine| engine.get("kind"))
            .and_then(Item::as_str)
            .map(str::to_owned);
        out.push(Member { name, at, dir, kind, manifest });
    }
    Ok(out)
}

fn check_member(root: &Path, workspace: &Item, member: &Member, by_name: &BTreeMap<&str, &Member>, problems: &mut Vec<String>) {
    let at = format!("{} ({})", member.at, member.name);
    let Some(kind) = member.kind.as_deref() else {
        problems.push(format!("{at}: no [package.metadata.engine] kind"));
        return;
    };
    if !KINDS.contains(&kind) {
        problems.push(format!("{at}: unknown kind {kind}"));
        return;
    }
    if let Some((_, names)) = CLOSED.iter().find(|(closed, _)| *closed == kind) {
        let in_tools = kind == "tool" && member.at.starts_with("tools/");
        if !names.contains(&member.name.as_str()) && !in_tools {
            problems.push(format!("{at}: the kind {kind} is closed and does not name this crate"));
        }
    }
    if kind != "app" && crate_types(&member.manifest).iter().any(|crate_type| crate_type == "cdylib") {
        problems.push(format!("{at}: only app crates build a cdylib"));
    }
    if matches!(kind, "brick" | "place" | "list") && !forbids_unsafe(&member.manifest, workspace) {
        problems.push(format!("{at}: a {kind} crate needs [lints.rust] unsafe_code = \"forbid\""));
    }
    let leaf_side = member.at.starts_with("crates/") || member.at.starts_with("tools/");
    let examples = root.join("examples");
    for dependency in dependencies(root, workspace, member) {
        let edge = format!("{at}: {} {}", dependency.table, dependency.key);
        if dependency.unresolved {
            problems.push(format!("{edge}: inherits a dependency the workspace does not declare"));
            continue;
        }
        if leaf_side && dependency.path.as_ref().is_some_and(|path| path.starts_with(&examples)) {
            problems.push(format!("{edge}: nothing in crates/ or tools/ depends on examples/"));
        }
        if dependency.package == "wasm-bindgen" && kind != "app" {
            problems.push(format!("{edge}: only app crates depend on wasm-bindgen"));
        }
        match by_name.get(dependency.package.as_str()) {
            Some(target) => {
                let target_kind = target.kind.as_deref().unwrap_or("no kind");
                if !allowed(kind, target_kind, &target.name, target.at.starts_with("examples/")) {
                    problems.push(format!("{edge}: a {kind} crate may not depend on {} ({target_kind})", target.name));
                }
            }
            None => {
                if let Some(path) = &dependency.path {
                    if path.starts_with(root) {
                        problems.push(format!("{edge}: {} is inside the workspace but not a member", relative(root, path)));
                    } else if matches!(kind, "brick" | "place" | "list") {
                        problems.push(format!("{edge}: a {kind} crate has no path dependency outside the workspace"));
                    }
                }
                if dependency.git && matches!(kind, "brick" | "place" | "list") {
                    problems.push(format!("{edge}: a {kind} crate has no git dependency"));
                }
                if let Some(reason) = external(kind, &dependency.package) {
                    problems.push(format!("{edge}: {reason}"));
                }
            }
        }
    }
}

fn allowed(kind: &str, target_kind: &str, target: &str, target_in_examples: bool) -> bool {
    match kind {
        "support" => target_kind == "support",
        "core" | "core-internal" | "host" => matches!(target_kind, "support" | "core" | "core-internal"),
        "place" => target_kind == "support" || matches!(target, "engine_place" | "engine_ecs"),
        "brick" => matches!(target_kind, "support" | "place") || matches!(target, "engine_ecs" | "engine_core" | "engine_render"),
        "list" => matches!(target_kind, "place" | "brick") || target == "engine_core",
        "app" => matches!(target_kind, "host" | "list" | "core"),
        "tool" => !target_in_examples,
        _ => false,
    }
}

fn external(kind: &str, package: &str) -> Option<String> {
    match kind {
        "list" => Some(format!("a list crate depends on no external crate ({package})")),
        "place" if !PLACE_EXTERNALS.contains(&package) => {
            Some(format!("a place crate depends only on {} among external crates ({package})", PLACE_EXTERNALS.join(", ")))
        }
        "support" | "brick" if engine_only(package) => Some(format!("{package} is engine-only")),
        _ => None,
    }
}

fn engine_only(package: &str) -> bool {
    package == "bevy"
        || package.starts_with("bevy_")
        || package.starts_with("wgpu")
        || package == "naga"
        || package == "wesl"
        || package.starts_with("wasm-bindgen")
        || package == "web-sys"
        || package == "js-sys"
}

fn dependencies(root: &Path, workspace: &Item, member: &Member) -> Vec<Dependency> {
    let mut tables: Vec<(String, &Item)> = Vec::new();
    for table in TABLES {
        if let Some(item) = member.manifest.get(*table) {
            tables.push(((*table).to_owned(), item));
        }
    }
    if let Some(targets) = member.manifest.get("target").and_then(Item::as_table_like) {
        for (cfg, target) in targets.iter() {
            for table in TABLES {
                if let Some(item) = target.get(*table) {
                    tables.push((format!("target.{cfg}.{table}"), item));
                }
            }
        }
    }
    let mut out = Vec::new();
    for (table, item) in tables {
        let Some(entries) = item.as_table_like() else { continue };
        for (key, entry) in entries.iter() {
            out.push(resolve(root, workspace, &member.dir, &table, key, entry));
        }
    }
    out
}

fn resolve(root: &Path, workspace: &Item, dir: &Path, table: &str, key: &str, entry: &Item) -> Dependency {
    let inherited = entry.get("workspace").and_then(Item::as_bool) == Some(true);
    let (source, base) = if inherited {
        (workspace.get("workspace").and_then(|table| table.get("dependencies")).and_then(|dependencies| dependencies.get(key)), root)
    } else {
        (Some(entry), dir)
    };
    let renamed = entry.get("package").and_then(Item::as_str);
    let package = renamed.or_else(|| source.and_then(|item| item.get("package")).and_then(Item::as_str)).unwrap_or(key).to_owned();
    let path = source.and_then(|item| item.get("path")).and_then(Item::as_str).map(|path| normal(&base.join(path)));
    let git = source.is_some_and(|item| item.get("git").is_some());
    Dependency { table: table.to_owned(), key: key.to_owned(), package, path, git, unresolved: source.is_none() }
}

fn crate_types(manifest: &Item) -> Vec<String> {
    strings(manifest.get("lib").and_then(|lib| lib.get("crate-type").or_else(|| lib.get("crate_type"))))
}

fn forbids_unsafe(manifest: &Item, workspace: &Item) -> bool {
    let lints = manifest.get("lints");
    let inherited = lints.and_then(|lints| lints.get("workspace")).and_then(Item::as_bool) == Some(true);
    let table = if inherited { workspace.get("workspace").and_then(|table| table.get("lints")) } else { lints };
    let level = table.and_then(|lints| lints.get("rust")).and_then(|rust| rust.get("unsafe_code"));
    level.and_then(|level| level.as_str().or_else(|| level.get("level").and_then(Item::as_str))) == Some("forbid")
}

fn strings(item: Option<&Item>) -> Vec<String> {
    item.and_then(Item::as_array)
        .map(|array| array.iter().filter_map(|value| value.as_str().map(str::to_owned)).collect())
        .unwrap_or_default()
}

fn read(path: &Path) -> Result<Item, String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Document::parse(text).map(Document::into_item).map_err(|error| format!("{}: {error}", path.display()))
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
