mod build_scripts;
mod draw_kinds;
mod techniques;

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
    "shared",
    "kit",
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
            "engine_lodbuild",
            "engine_sources",
            "engine_command_line",
        ],
    ),
    ("core", engine_registry::CORE_CRATES),
    ("core-internal", engine_registry::CORE_INTERNAL_CRATES),
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

const RINGS: &[(&str, &[&str])] = &[
    (
        "core",
        &[
            "engine_ecs",
            "engine_place",
            "engine_gpu_bits",
            "engine_core",
            "engine_registry",
            "engine_render",
            "engine_assets",
            "engine_textures",
            "engine_settings",
            "engine_scene",
            "engine_profile",
            "engine_panic",
        ],
    ),
    ("rendering", &[]),
    ("simulation", &[]),
    ("world", &[]),
];

const RINGED: &[&str] = &["core", "core-internal", "place", "shared", "kit", "brick"];

const KITS: &[(&str, &[&str])] = &[];

const PLACE_EXTERNALS: &[&str] = &["glam", "bytemuck", "serde", "serde_json", "schemars"];

const SHARED_EXTERNALS: &[&str] = &[
    "glam",
    "bytemuck",
    "serde",
    "serde_json",
    "schemars",
    "parry3d",
    "libm",
];

struct Member {
    name: String,
    at: String,
    dir: PathBuf,
    manifest: PathBuf,
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
    kits(&members, problems);
    placed(&members, problems);
    implementors(&root, &members, problems)?;
    declarations(&root, &members, problems)?;
    technique_free(&root, &members, &folders, problems)?;
    draw_kind_free(&root, &members, problems)?;
    build_scripts_inside(&root, &members, &folders, problems)?;
    for member in &members {
        check_member(&workspace, member, &folders, problems)?;
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

fn kits(members: &[Member], problems: &mut Vec<String>) {
    for &(kit, users) in KITS {
        match members.iter().find(|member| member.name == kit) {
            None => problems.push(format!("Cargo.toml: the kit {kit} is no workspace crate")),
            Some(member) if member.kind.as_deref() != Some("kit") => problems.push(format!(
                "{}: the kits name this crate, which declares {}",
                member.label(),
                member.kind.as_deref().unwrap_or("no kind")
            )),
            Some(_) => {}
        }
        for user in users {
            match members.iter().find(|member| member.name == *user) {
                None => problems.push(format!(
                    "Cargo.toml: the kit {kit} names {user}, which is no workspace crate"
                )),
                Some(member) if member.kind.as_deref() != Some("brick") => problems.push(format!(
                    "{}: the kit {kit} names this crate as a part, which declares {}",
                    member.label(),
                    member.kind.as_deref().unwrap_or("no kind")
                )),
                Some(_) => {}
            }
        }
    }
    for member in members {
        if member.kind.as_deref() == Some("kit") && !KITS.iter().any(|(kit, _)| *kit == member.name)
        {
            problems.push(format!(
                "{}: a kit crate needs a place in the kits list",
                member.label()
            ));
        }
    }
}

fn kit_part(kit: &str, brick: &str) -> bool {
    KITS.iter()
        .any(|&(name, parts)| name == kit && parts.contains(&brick))
}

fn declarations(root: &Path, members: &[Member], problems: &mut Vec<String>) -> Result<(), String> {
    for member in members {
        let source = member.dir.join("src");
        if member.kind.as_deref() != Some("shared") || !source.is_dir() {
            continue;
        }
        let mut files = Vec::new();
        matching_files(&source, declares_place, &mut files)?;
        for file in files {
            problems.push(format!(
                "{}: a shared crate declares a place in {}; places live in place or kit crates",
                member.label(),
                relative(root, &file)
            ));
        }
    }
    Ok(())
}

fn core(member: &Member) -> bool {
    member
        .kind
        .as_deref()
        .is_some_and(|kind| techniques::FREE_KINDS.contains(&kind))
}

fn linked(dependency: &Dependency) -> bool {
    dependency.table.rsplit('.').next() == Some("dependencies")
}

fn linked_by_core<'a>(
    members: &'a [Member],
    folders: &BTreeMap<&Path, &'a Member>,
) -> BTreeSet<&'a str> {
    let mut reached = BTreeSet::new();
    let mut pending: Vec<&Member> = members.iter().filter(|member| core(member)).collect();
    while let Some(member) = pending.pop() {
        for dependency in member
            .dependencies
            .iter()
            .filter(|dependency| linked(dependency))
        {
            let Some(target) = dependency
                .path
                .as_deref()
                .and_then(|dir| folders.get(dir).copied())
            else {
                continue;
            };
            if !core(target) && reached.insert(target.name.as_str()) {
                pending.push(target);
            }
        }
    }
    reached
}

fn technique_free(
    root: &Path,
    members: &[Member],
    folders: &BTreeMap<&Path, &Member>,
    problems: &mut Vec<String>,
) -> Result<(), String> {
    let linked = linked_by_core(members, folders);
    for member in members {
        let label = if core(member) {
            member.label()
        } else if linked.contains(member.name.as_str()) {
            format!("{}, which the core links", member.label())
        } else {
            continue;
        };
        let mut found = Vec::new();
        let source = member.dir.join("src");
        if source.is_dir() {
            techniques::files(&source, &mut found)?;
        }
        found.extend(member.build.iter().cloned());
        techniques::check(&label, &found, |path| relative(root, path), problems)?;
    }
    Ok(())
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

fn draw_kind_free(
    root: &Path,
    members: &[Member],
    problems: &mut Vec<String>,
) -> Result<(), String> {
    for member in members {
        let free = member
            .kind
            .as_deref()
            .is_some_and(|kind| techniques::FREE_KINDS.contains(&kind));
        if free {
            draw_kinds::check(
                &member.label(),
                &member.dir.join("src"),
                |path| relative(root, path),
                problems,
            )?;
        }
    }
    Ok(())
}

fn declares_place(line: &str) -> bool {
    let mut words = line.split_whitespace();
    let implements = words.next() == Some("impl")
        && words
            .next()
            .is_some_and(|path| matches!(path.rsplit("::").next(), Some("Slot" | "List" | "Ask")))
        && words.next() == Some("for");
    implements || line.contains("PlaceDecl::")
}

fn ring(name: &str) -> Option<&'static str> {
    RINGS
        .iter()
        .find(|(_, names)| names.contains(&name))
        .map(|&(ring, _)| ring)
}

fn names(ring: &str, target: &str) -> bool {
    match ring {
        "core" => target == "core",
        "rendering" => matches!(target, "core" | "rendering"),
        "simulation" => matches!(target, "core" | "simulation"),
        "world" => true,
        _ => false,
    }
}

fn placed(members: &[Member], problems: &mut Vec<String>) {
    for &(ring, names) in RINGS {
        for name in names {
            if !members.iter().any(|member| member.name == *name) {
                problems.push(format!(
                    "Cargo.toml: the {ring} ring names {name}, which is no workspace crate"
                ));
            }
        }
    }
    for member in members {
        let kind = member.kind.as_deref().unwrap_or("no kind");
        let rings = RINGS
            .iter()
            .filter(|(_, names)| names.contains(&member.name.as_str()))
            .count();
        let problem = match (RINGED.contains(&kind), rings) {
            (true, 1) | (false, 0) => continue,
            (true, 0) => "sits in no ring",
            (true, _) => "sits in more than one ring",
            (false, _) => "sits in a ring",
        };
        problems.push(format!("{}: a {kind} crate {problem}", member.label()));
    }
}

fn implementors(root: &Path, members: &[Member], problems: &mut Vec<String>) -> Result<(), String> {
    for member in members {
        let kind = member.kind.as_deref().unwrap_or("no kind");
        let source = member.dir.join("src");
        if matches!(kind, "brick" | "game") || !source.is_dir() {
            continue;
        }
        let mut files = Vec::new();
        matching_files(&source, implements_brick, &mut files)?;
        for file in files {
            problems.push(format!(
                "{}: a {kind} crate implements Brick in {}; only a brick in a ring or a game may",
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

fn check_member(
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
    if !member.engine_rules.is_null() && (kind != "game" || member.engine_rules != true) {
        problems.push(format!(
            "{at}: engine_rules is true on a game crate or absent"
        ));
    }
    if matches!(kind, "brick" | "place" | "shared" | "kit" | "list")
        && !forbids_unsafe(&read(&member.manifest)?, workspace)
    {
        problems.push(format!(
            "{at}: a {kind} crate needs [lints.rust] unsafe_code = \"forbid\""
        ));
    }
    for dependency in &member.dependencies {
        let edge = format!("{at}: {} {}", dependency.table, dependency.key);
        if let Some(target) = dependency.path.as_deref().and_then(|dir| folders.get(dir)) {
            if !allowed(member, kind, target) {
                let target_kind = target.kind.as_deref().unwrap_or("no kind");
                problems.push(format!(
                    "{edge}: a {kind} crate may not depend on {} ({target_kind})",
                    target.name
                ));
            }
            if let (Some(own), Some(theirs)) = (ring(&member.name), ring(&target.name))
                && !names(own, theirs)
            {
                problems.push(format!(
                    "{edge}: the {own} ring may not name {} of the {theirs} ring",
                    target.name
                ));
            }
            continue;
        }
        if matches!(kind, "brick" | "place" | "shared" | "kit" | "list") {
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
        "place" => {
            matches!(target_kind, "support" | "place")
                || matches!(name, "engine_place" | "engine_ecs")
        }
        "shared" | "kit" => {
            matches!(target_kind, "support" | "place" | "shared")
                || matches!(name, "engine_place" | "engine_ecs")
        }
        "brick" => {
            matches!(target_kind, "support" | "place" | "shared")
                || (target_kind == "kit" && kit_part(name, &member.name))
                || matches!(name, "engine_ecs" | "engine_core" | "engine_render")
        }
        "list" => {
            matches!(target_kind, "place" | "kit" | "brick" | "list") || name == "engine_core"
        }
        "game" => matches!(
            target_kind,
            "host" | "list" | "brick" | "kit" | "shared" | "place" | "core" | "support"
        ),
        "tool" => target_kind != "brick",
        _ => false,
    }
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
        "shared" | "kit" if !SHARED_EXTERNALS.contains(&package) => Some(format!(
            "a {kind} crate depends only on {} among external crates ({package})",
            SHARED_EXTERNALS.join(", ")
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
