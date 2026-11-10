use std::{
    fs,
    path::{Path, PathBuf},
};

use engine::Engine;
use engine_checks::Repository;
use engine_core::{DrawnTable, ScenePasses};
use engine_place::PlaceKind;
use engine_registry::Registry;
use serde_json::json;

use crate::process::{Client, Project, WASM_TARGET, project_option, read_clients, run};

const USAGE: &str = "usage: cargo xtask new app --project DIR | new brick NAME --project DIR \
                     [--kind gameplay|drawn] [--place PLACE]";

const APP_TEMPLATE: &str = "crates/engine/examples/app_template";

pub(crate) const TEMPLATE_CRATE: &str = "app_template";

const TEMPLATE_KIND: &str = "kind = \"game\"\n";

const TEMPLATE_SCENE: &str = "template.scene.json";

#[derive(Clone, Copy, PartialEq, Eq)]
enum BrickKind {
    Gameplay,
    Drawn,
}

impl BrickKind {
    const ALL: [(Self, &'static str); 2] = [(Self::Gameplay, "gameplay"), (Self::Drawn, "drawn")];

    fn named(given: &str) -> Result<Self, String> {
        Self::ALL
            .iter()
            .find(|(_, name)| *name == given)
            .map(|(kind, _)| *kind)
            .ok_or_else(|| {
                json!({
                    "code": "unknown_variant",
                    "path": "--kind",
                    "got": given,
                    "variants": Self::ALL.iter().map(|(_, name)| *name).collect::<Vec<_>>(),
                })
                .to_string()
            })
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Gameplay => "gameplay",
            Self::Drawn => "drawn",
        }
    }

    const fn template(self) -> &'static str {
        match self {
            Self::Gameplay => "turning",
            Self::Drawn => "glimmer",
        }
    }

    const fn needs(self) -> &'static [&'static str] {
        match self {
            Self::Gameplay => &["engine", "serde"],
            Self::Drawn => &["engine_core", "engine_ecs", "engine_render"],
        }
    }

    fn folder(self) -> String {
        format!("src/bricks/{}", self.template())
    }
}

const RESERVED: [&str; 4] = ["engine", "generated", "face", "place"];

pub fn new(root: &Path, options: &[String]) -> Result<(), String> {
    match options.split_first() {
        Some((kind, rest)) if kind == "app" => app(root, rest),
        Some((kind, rest)) if kind == "brick" => brick(root, rest),
        _ => Err(USAGE.into()),
    }
}

fn pascal(words: &str, separator: char) -> String {
    words
        .split(separator)
        .filter(|word| !word.is_empty())
        .map(|word| word[..1].to_ascii_uppercase() + &word[1..])
        .collect()
}

fn lowercase_words(name: &str, separator: char) -> bool {
    name.starts_with(|character: char| character.is_ascii_lowercase())
        && name.split(separator).all(|word| {
            !word.is_empty()
                && word
                    .chars()
                    .all(|character| character.is_ascii_lowercase() || character.is_ascii_digit())
        })
}

fn write_files(root: &Path, files: &[(PathBuf, String)]) -> Result<(), String> {
    for (path, contents) in files {
        let folder = path
            .parent()
            .ok_or_else(|| format!("{} has no folder", path.display()))?;
        fs::create_dir_all(folder)
            .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
        fs::write(path, contents)
            .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
        let shown = path.strip_prefix(root).unwrap_or(path);
        println!(
            "{}",
            json!({"event": "created", "file": crate::process::slashed(shown)})
        );
    }
    Ok(())
}

fn template_files(base: &Path) -> Result<Vec<(PathBuf, String)>, String> {
    let base = base.to_path_buf();
    let mut found = Vec::new();
    let mut folders = vec![base.clone()];
    while let Some(folder) = folders.pop() {
        let unreadable =
            |error: std::io::Error| format!("cannot read {}: {error}", folder.display());
        for entry in fs::read_dir(&folder).map_err(unreadable)? {
            let path = entry.map_err(unreadable)?.path();
            if path.is_dir() {
                folders.push(path);
                continue;
            }
            let text = fs::read_to_string(&path)
                .map(|text| text.replace("\r\n", "\n"))
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            let inside = path
                .strip_prefix(&base)
                .map_err(|_| format!("{} lies outside {}", path.display(), base.display()))?;
            found.push((inside.to_path_buf(), text));
        }
    }
    found.sort();
    Ok(found)
}

fn app(root: &Path, options: &[String]) -> Result<(), String> {
    let (project, rest) = project_option(options)?;
    let (Some(project), true) = (project, rest.is_empty()) else {
        return Err(USAGE.into());
    };
    let project = project
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_owned();
    let name = project.rsplit('/').next().unwrap_or_default().to_owned();
    if !lowercase_words(&name, '-') || name.starts_with("engine") || name == "xtask" {
        return Err(json!({
            "code": "invalid_name",
            "path": "--project",
            "got": name,
            "expected": "lowercase words of letters and digits joined by dashes, not starting with engine",
        })
        .to_string());
    }
    let wasm = name.replace('-', "_");
    let pascal = pascal(&name, '-');
    let crate_dir = root.join(&project);
    let page = format!("{name}.html");
    let web = root.join("web/src").join(&name);
    if !Project::read(root, &project).is_ok_and(|found| found.package == name) {
        for taken in [crate_dir.clone(), root.join("web").join(&page), web.clone()] {
            if taken.exists() {
                return Err(format!("{} already exists", taken.display()));
            }
        }
    }
    let named = |client: &Client| {
        client.folder != project
            && (client.game.as_deref() == Some(name.as_str()) || client.wasm == wasm)
    };
    if read_clients()?.iter().any(named) {
        return Err(format!("a client named {name} already exists"));
    }
    let mut files = Vec::new();
    let drawn = BrickKind::Drawn;
    for (inside, text) in template_files(&root.join(APP_TEMPLATE))? {
        if inside.starts_with(drawn.folder()) {
            continue;
        }
        let text = if inside == Path::new("Cargo.toml") {
            game_manifest(&text, &name, &page)?
        } else {
            text.replace(&format!("    pub mod {};\n", drawn.template()), "")
                .replace(TEMPLATE_CRATE, &wasm)
                .replace("Template", &pascal)
        };
        files.push((crate_dir.join(inside), text));
    }
    let tool = format!("{name}-xtask");
    files.extend([
        (crate_dir.join("xtask/Cargo.toml"), xtask_manifest(&name)),
        (crate_dir.join("xtask/src/main.rs"), xtask_main(&wasm)),
        (
            crate_dir.join(".cargo/config.toml"),
            format!("[alias]\nxtask = \"run -p {tool} --\"\n"),
        ),
        (root.join("web").join(&page), page_html(&name, &pascal)),
        (web.join("main.ts"), page_entry(&wasm, &pascal)),
        (web.join("worker.ts"), worker_entry(&wasm)),
    ]);
    files.retain(|(path, _)| !path.exists());
    write_files(root, &files)?;
    for member in ["", "/xtask"] {
        join_workspace(root, &format!("{project}{member}"))?;
    }
    join_dependencies(root, &name, &project)?;
    run(root, "cargo", &["run", "-p", &tool, "--", "generate"])?;
    run(
        root,
        "cargo",
        &["check", "-p", &name, "--target", WASM_TARGET],
    )?;
    run(root, "cargo", &["run", "-q", "-p", "kinds"])?;
    crate::authoring::write_generated(root)?;
    println!(
        "{}",
        json!({
            "event": "app_created",
            "project": project,
            "page": format!("web/{page}"),
            "native": format!("{name}-native"),
            "xtask": tool,
        })
    );
    Ok(())
}

fn game_manifest(template: &str, name: &str, page: &str) -> Result<String, String> {
    if !template.contains(TEMPLATE_KIND)
        || !template.contains(&format!("name = \"{TEMPLATE_CRATE}\""))
    {
        return Err(format!(
            "{APP_TEMPLATE}/Cargo.toml no longer declares the crate {TEMPLATE_CRATE} of kind game"
        ));
    }
    let engine = format!(
        "{TEMPLATE_KIND}page = \"{page}\"\ngame = \"{name}\"\nnative = \"{name}-native\"\n"
    );
    Ok(template
        .replace(TEMPLATE_CRATE, name)
        .replace(TEMPLATE_KIND, &engine))
}

fn xtask_manifest(name: &str) -> String {
    format!(
        r#"[package]
name = "{name}-xtask"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[dependencies]
engine_checks.workspace = true
{name} = {{ workspace = true, features = ["dev"] }}

[package.metadata.engine]
kind = "tool"
"#
    )
}

fn xtask_main(wasm: &str) -> String {
    format!(
        r#"use {wasm} as app;

fn main() -> std::process::ExitCode {{
    engine_checks::app::main(env!("CARGO_MANIFEST_DIR"), app::GAME.bricks, app::GAME.build)
}}
"#
    )
}

fn page_html(name: &str, pascal: &str) -> String {
    format!(
        r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{pascal}</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
  </head>
  <body>
    <script type="module" src="/src/{name}/main.ts"></script>
  </body>
</html>
"#
    )
}

fn page_entry(wasm: &str, pascal: &str) -> String {
    format!(
        r#"import {{ playGame }} from "../engine/game";
import JobWorker from "./worker.ts?worker";

playGame({{
  wasm: "{wasm}",
  scene: "{TEMPLATE_SCENE}",
  load: () => import("../../pkg/{wasm}"),
  worker: JobWorker,
  game: bindings => bindings.{pascal}Game,
}});
"#
    )
}

fn worker_entry(wasm: &str) -> String {
    format!(
        r#"import {{ serveJobs }} from "../engine/job-worker";

serveJobs(() => import("../../pkg/{wasm}"));
"#
    )
}

enum Shape {
    Slot { contract: Option<&'static str> },
    List,
}

impl Shape {
    const fn name(&self) -> &'static str {
        match self {
            Self::Slot { .. } => "slot",
            Self::List => "list",
        }
    }
}

struct Claim {
    shape: Shape,
    place: String,
    path: String,
    package: String,
    filler: Option<&'static str>,
}

struct BrickArguments {
    project: String,
    name: String,
    place: Option<String>,
    kind: BrickKind,
}

fn brick_arguments(options: &[String]) -> Result<BrickArguments, String> {
    let (project, rest) = project_option(options)?;
    let (Some(project), Some((name, rest))) = (project, rest.split_first()) else {
        return Err(USAGE.into());
    };
    let (mut place, mut kind) = (None, BrickKind::Gameplay);
    let mut given = rest.iter();
    while let Some(flag) = given.next() {
        match (flag.as_str(), given.next()) {
            ("--place", Some(value)) => place = Some(value.as_str()),
            ("--kind", Some(value)) => kind = BrickKind::named(value)?,
            _ => return Err(USAGE.into()),
        }
    }
    if place.is_some() && kind != BrickKind::Drawn {
        return Err(json!({
            "code": "invalid_arguments",
            "path": "--place",
            "expected": "--kind drawn",
            "got": kind.name(),
        })
        .to_string());
    }
    let named_options = std::iter::once(("NAME", name.as_str()))
        .chain(place.map(|place| ("--place", place)));
    for (path, given) in named_options {
        if !lowercase_words(given, '_') {
            return Err(json!({
                "code": "invalid_name",
                "path": path,
                "got": given,
                "expected": "lowercase words of letters and digits joined by underscores",
            })
            .to_string());
        }
    }
    if RESERVED.contains(&name.as_str()) {
        return Err(json!({
            "code": "invalid_name",
            "path": "NAME",
            "got": name,
            "expected": format!("a prefix no core module uses, not {}", RESERVED.join(", ")),
        })
        .to_string());
    }
    Ok(BrickArguments {
        project,
        name: name.clone(),
        place: place.map(str::to_owned),
        kind,
    })
}

fn brick(root: &Path, options: &[String]) -> Result<(), String> {
    let BrickArguments {
        project,
        name,
        place,
        kind,
    } = brick_arguments(options)?;
    let (name, place) = (&name, place.as_deref());
    let project = project
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_owned();
    let game = game_crate(root, &project)?;
    let module = format!("{project}/src/bricks/{name}");
    if root.join(&module).exists() {
        return Err(format!("{module} already exists"));
    }
    let repository = crate::rules::repository(root)?;
    let engine = crate::bricks::standard_engine().map_err(|error| error.to_string())?;
    let registry = engine
        .app
        .world()
        .get_resource::<Registry>()
        .ok_or("the standard engine holds no brick registry")?;
    taken(root, &project, name, &engine)?;
    let named = pascal(name, '_');
    let claim = match place {
        Some(place) => {
            let found = claim(root, repository, registry, place)?;
            if found.is_none() {
                println!(
                    "{}",
                    json!({"event": "part_undeclared", "part": place, "brick": name})
                );
            }
            found
        }
        None => None,
    };
    let upper = name.to_ascii_uppercase();
    let template = kind.template();
    let template_named = pascal(template, '_');
    let template_upper = template.to_ascii_uppercase();
    let rename = |text: &str| {
        text.replace(template, name)
            .replace(&template_named, &named)
            .replace(&template_upper, &upper)
    };
    let mut files = Vec::new();
    for (inside, text) in template_files(&root.join(APP_TEMPLATE).join(kind.folder()))? {
        let inside: PathBuf = inside
            .components()
            .map(|part| rename(&part.as_os_str().to_string_lossy()))
            .collect();
        let mut text = rename(&text);
        if inside == Path::new("mod.rs") {
            text = registered(text, &named, name, claim.as_ref())?;
        }
        files.push((root.join(&module).join(inside), text));
    }
    if let Some(Claim {
        shape: Shape::Slot {
            contract: Some(contract),
        },
        place,
        ..
    }) = &claim
    {
        files.push((
            root.join(&module)
                .join(format!("shaders/{name}/faces/{place}.wgsl")),
            answers(place, contract),
        ));
    }
    write_files(root, &files)?;
    let listed = declare_module(root, &project, name, &named)?;
    need_dependencies(root, &project, kind, claim.as_ref())?;
    run(root, "cargo", &["check", "-p", &game])?;
    run(
        root,
        "cargo",
        &["check", "-p", &game, "--target", WASM_TARGET],
    )?;
    run(root, "cargo", &["run", "-q", "-p", "kinds"])?;
    println!(
        "{}",
        json!({
            "event": "brick_created",
            "project": project,
            "module": module,
            "brick": format!("{}::bricks::{name}::{named}Brick", game.replace('-', "_")),
            "kind": kind.name(),
            "listed": listed,
            "part": place,
            "claims": claim.as_ref().map(|claim| claim.shape.name()),
            "filled_by": claim.as_ref().and_then(|claim| claim.filler),
        })
    );
    Ok(())
}

fn game_crate(root: &Path, project: &str) -> Result<String, String> {
    let manifest = format!("{project}/Cargo.toml");
    let text = fs::read_to_string(root.join(&manifest))
        .map_err(|error| format!("--project {project}: cannot read {manifest}: {error}"))?;
    let document = toml_edit::Document::parse(text)
        .map_err(|error| format!("{manifest} is not TOML: {error}"))?;
    let package = document.as_item().get("package");
    let kind = package
        .and_then(|package| package.get("metadata"))
        .and_then(|metadata| metadata.get("engine"))
        .and_then(|engine| engine.get("kind"))
        .and_then(toml_edit::Item::as_str);
    if kind != Some("game") {
        return Err(json!({
            "code": "invalid_project",
            "path": "--project",
            "got": project,
            "expected": "a game crate, whose [package.metadata.engine] kind is game",
        })
        .to_string());
    }
    let build = fs::read_to_string(root.join(project).join("build.rs")).unwrap_or_default();
    if !build.contains("engine_sources::emit") {
        return Err(format!(
            "{project}/build.rs does not call engine_sources::emit, which writes a brick module's source"
        ));
    }
    package
        .and_then(|package| package.get("name"))
        .and_then(toml_edit::Item::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("{manifest}: [package] has no name"))
}

fn taken(root: &Path, project: &str, name: &str, engine: &Engine) -> Result<(), String> {
    let world = engine.app.world();
    let registry = world
        .get_resource::<Registry>()
        .ok_or("the standard engine holds no brick registry")?;
    let twice = |path: String, owner: &str| -> Result<(), String> {
        Err(json!({"code": "name_twice", "path": path, "got": owner}).to_string())
    };
    let stats = format!("{name}_stats");
    for key in [name, stats.as_str()] {
        let owner = registry
            .section(key)
            .map(|entry| entry.brick)
            .or_else(|| registry.stats_section(key).map(|entry| entry.brick));
        if let Some(owner) = owner {
            return twice(format!("$.sections.{key}"), owner);
        }
    }
    if let Some(pass) = world
        .get_resource::<ScenePasses>()
        .and_then(|passes| passes.get(name))
    {
        return twice(format!("$.passes.{name}"), pass.brick);
    }
    let kind = world
        .get_resource::<DrawnTable>()
        .is_some_and(|table| table.kinds().iter().any(|decl| decl.kind.name() == name));
    if kind {
        return twice(format!("$.draw_kinds.{name}"), "a registered draw kind");
    }
    let inside = format!("{project}/src/");
    for path in engine_checks::files::sources(root)? {
        if !path.starts_with(&inside) || !path.ends_with(".rs") {
            continue;
        }
        let text = fs::read_to_string(root.join(&path))
            .map_err(|error| format!("cannot read {path}: {error}"))?;
        for key in [name, stats.as_str()] {
            if text.contains(&format!("#[section(\"{key}\")]")) {
                return twice(format!("$.sections.{key}"), &path);
            }
        }
    }
    Ok(())
}

fn registered(
    text: String,
    named: &str,
    name: &str,
    claim: Option<&Claim>,
) -> Result<String, String> {
    let Some(claim) = claim else {
        return Ok(text);
    };
    let line = match &claim.shape {
        Shape::Slot { contract } => {
            let wgsl = match contract {
                Some(_) => format!("Some(shaders::{name}::faces::{}::MODULE)", claim.place),
                None => "None".to_owned(),
            };
            format!(".fill::<{}>({wgsl})", claim.path)
        }
        Shape::List => format!(".join::<{}>(Default::default(), None)", claim.path),
    };
    let anchor = format!(".settings::<{named}>()\n");
    if !text.contains(&anchor) {
        return Err(format!(
            "{APP_TEMPLATE}/{}/mod.rs no longer registers its settings",
            BrickKind::Drawn.folder()
        ));
    }
    Ok(text.replacen(&anchor, &format!("{anchor}            {line}\n"), 1))
}

fn edited(path: &Path, edit: impl FnOnce(&str) -> String) -> Result<(), String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let crlf = text.contains("\r\n");
    let mut changed = edit(&text.replace("\r\n", "\n"));
    if crlf {
        changed = changed.replace('\n', "\r\n");
    }
    fs::write(path, changed).map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn declare_module(root: &Path, project: &str, name: &str, named: &str) -> Result<bool, String> {
    let path = root.join(project).join("src/lib.rs");
    let opened = "pub mod bricks {\n";
    let line = format!("    pub mod {name};\n");
    let brick = format!(".with::<bricks::{name}::{named}Brick>()");
    let listing = "pub fn bricks<L: Listing>(list: L) -> L {\n";
    let mut listed = false;
    edited(&path, |text| {
        let mut text = match text.find(opened) {
            Some(start) => {
                let mut at = start + opened.len();
                while let Some(rest) = text[at..].strip_prefix("    pub mod ") {
                    let end = rest.find('\n').map_or(text.len(), |end| at + end + 13);
                    let known = rest[..rest.find(';').unwrap_or(0)].trim();
                    if known > name {
                        break;
                    }
                    at = end;
                }
                format!("{}{line}{}", &text[..at], &text[at..])
            }
            None => format!("{text}\npub mod bricks {{\n{line}}}\n"),
        };
        if let Some(start) = text.find(listing) {
            let body = start + listing.len();
            if let Some(end) = text[body..].find("\n}\n").map(|end| body + end)
                && text[body..end].trim_end().ends_with(')')
            {
                text.insert_str(end, &format!("\n        {brick}"));
                listed = true;
            }
        }
        text
    })?;
    println!(
        "{}",
        json!({
            "event": if listed { "brick_listed" } else { "brick_unlisted" },
            "file": format!("{project}/src/lib.rs"),
            "list": brick,
        })
    );
    Ok(listed)
}

fn need_dependencies(
    root: &Path,
    project: &str,
    kind: BrickKind,
    claim: Option<&Claim>,
) -> Result<(), String> {
    let path = root.join(project).join("Cargo.toml");
    let mut needs: Vec<&str> = kind.needs().to_vec();
    if let Some(claim) = claim
        && !needs.contains(&claim.package.as_str())
    {
        needs.push(&claim.package);
    }
    let mut added = Vec::new();
    edited(&path, |text| {
        let mut text = text.to_owned();
        let table = "\n[dependencies]\n";
        for need in needs {
            let held = [format!("\n{need}.workspace"), format!("\n{need} =")]
                .iter()
                .any(|line| text.contains(line.as_str()));
            if held {
                continue;
            }
            let at = match text.find(table) {
                Some(at) => at + table.len(),
                None => {
                    text.push_str(table);
                    text.len()
                }
            };
            text.insert_str(at, &format!("{need}.workspace = true\n"));
            added.push(need.to_owned());
        }
        text
    })?;
    for package in added {
        println!(
            "{}",
            json!({"event": "dependency_added", "package": package, "project": project})
        );
    }
    Ok(())
}

fn claim(
    root: &Path,
    repository: &Repository,
    registry: &Registry,
    part: &str,
) -> Result<Option<Claim>, String> {
    let Some(decl) = registry.place(part).copied() else {
        return Ok(None);
    };
    let Some(declared) = declared(root, repository, part)? else {
        return Ok(None);
    };
    let kind = declared.kind.as_str();
    let core = matches!(declared.package.as_str(), "engine_core" | "engine_render");
    if !matches!(kind, "place" | "kit") && !core {
        return Err(format!(
            "{part} is declared in {} (kind {kind}), which a brick cannot depend on",
            declared.package
        ));
    }
    let shape = match decl.kind() {
        PlaceKind::Slot => Shape::Slot {
            contract: decl.wgsl(),
        },
        PlaceKind::List => Shape::List,
        PlaceKind::Component | PlaceKind::Ledger => return Ok(None),
    };
    let filler = match shape {
        Shape::Slot { .. } => registry.filler_named(part),
        Shape::List => None,
    };
    Ok(Some(Claim {
        shape,
        place: part.to_owned(),
        path: declared.path,
        package: declared.package,
        filler,
    }))
}

struct Declared {
    path: String,
    package: String,
    kind: String,
}

fn declared(root: &Path, repository: &Repository, part: &str) -> Result<Option<Declared>, String> {
    let declaration = format!("const NAME: &'static str = \"{part}\";");
    for path in engine_checks::files::sources(root)? {
        if !path.ends_with(".rs") || !path.starts_with("crates/") {
            continue;
        }
        let text = fs::read_to_string(root.join(&path))
            .map_err(|error| format!("cannot read {path}: {error}"))?;
        for marker in ["impl Slot for ", "impl List for "] {
            for (at, _) in text.match_indices(marker) {
                let rest = &text[at + marker.len()..];
                let block = &rest[..rest.find('}').unwrap_or(rest.len())];
                if !block.contains(&declaration) {
                    continue;
                }
                let name: String = rest
                    .chars()
                    .take_while(|character| character.is_alphanumeric() || *character == '_')
                    .collect();
                let Some(package) = repository.own().find(|package| {
                    package
                        .path
                        .as_deref()
                        .is_some_and(|folder| path.starts_with(&format!("{folder}/src/")))
                }) else {
                    continue;
                };
                let module = module_of(&path, package.path.as_deref().unwrap_or_default());
                let inner = if module.is_empty() {
                    name.clone()
                } else {
                    format!("{module}::{name}")
                };
                return Ok(Some(Declared {
                    path: format!("{}::{inner}", package.crate_name()),
                    package: package.name.clone(),
                    kind: package.engine["kind"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                }));
            }
        }
    }
    Ok(None)
}

fn module_of(path: &str, folder: &str) -> String {
    let inside = path
        .strip_prefix(&format!("{folder}/src/"))
        .unwrap_or(path)
        .trim_end_matches(".rs");
    let inside = inside.strip_suffix("/mod").unwrap_or(inside);
    if inside == "lib" {
        return String::new();
    }
    inside.replace('/', "::")
}

fn items(text: &str) -> Vec<&str> {
    let mut items = Vec::new();
    let mut depth = 0_u32;
    let mut start = 0;
    for (at, character) in text.char_indices() {
        let ends = match character {
            '{' => {
                depth += 1;
                false
            }
            '}' => {
                depth = depth.saturating_sub(1);
                depth == 0
            }
            ';' => depth == 0,
            _ => false,
        };
        if ends {
            let item = text[start..=at].trim();
            if !item.is_empty() && item != ";" {
                items.push(item);
            }
            start = at + 1;
        }
    }
    items
}

fn declared_name(item: &str) -> Option<&str> {
    let rest = ["struct ", "const ", "alias ", "override "]
        .into_iter()
        .find_map(|keyword| item.strip_prefix(keyword))?;
    let end = rest
        .find(|character: char| !(character.is_alphanumeric() || character == '_'))
        .unwrap_or(rest.len());
    Some(&rest[..end]).filter(|name| !name.is_empty())
}

fn mentions(text: &str, name: &str) -> bool {
    let word = |character: char| character.is_alphanumeric() || character == '_';
    text.match_indices(name).any(|(at, _)| {
        !text[..at].chars().next_back().is_some_and(word)
            && !text[at + name.len()..].chars().next().is_some_and(word)
    })
}

fn answers(place: &str, contract: &str) -> String {
    let items = items(contract);
    let functions: Vec<&str> = items
        .iter()
        .copied()
        .filter(|item| item.starts_with("fn "))
        .collect();
    let named: Vec<&str> = items
        .iter()
        .filter_map(|item| declared_name(item))
        .filter(|name| functions.iter().any(|function| mentions(function, name)))
        .collect();
    let mut text = String::new();
    match named.as_slice() {
        [] => {}
        [one] => text.push_str(&format!("import package::place::{place}::{one};\n")),
        many => text.push_str(&format!(
            "import package::place::{place}::{{{}}};\n",
            many.join(", ")
        )),
    }
    for import in items.iter().filter(|item| item.starts_with("import ")) {
        text.push_str(import);
        text.push('\n');
    }
    if !text.is_empty() {
        text.push('\n');
    }
    text.push_str(&functions.join("\n\n"));
    text.push('\n');
    text
}

fn join_workspace(root: &Path, project: &str) -> Result<(), String> {
    let path = root.join("Cargo.toml");
    let manifest =
        fs::read_to_string(&path).map_err(|error| format!("cannot read Cargo.toml: {error}"))?;
    let parent = project.rsplit_once('/').map(|(parent, _)| parent);
    let listed = |member: &str| {
        manifest.contains(&format!("\"{member}\""))
            || parent.is_some_and(|parent| manifest.contains(&format!("\"{parent}/*\"")))
    };
    if listed(project) {
        return Ok(());
    }
    let members = "members = [";
    let at = manifest
        .find(members)
        .ok_or("Cargo.toml lists no [workspace] members")?
        + members.len();
    let joined = format!("{}\"{project}\", {}", &manifest[..at], &manifest[at..]);
    fs::write(&path, joined).map_err(|error| format!("cannot write Cargo.toml: {error}"))?;
    println!("{}", json!({"event": "member_added", "member": project}));
    Ok(())
}

fn join_dependencies(root: &Path, package: &str, folder: &str) -> Result<(), String> {
    let path = root.join("Cargo.toml");
    let manifest =
        fs::read_to_string(&path).map_err(|error| format!("cannot read Cargo.toml: {error}"))?;
    if manifest.contains(&format!("\n{package} = ")) {
        return Ok(());
    }
    let table = manifest
        .find("[workspace.dependencies]")
        .ok_or("Cargo.toml has no [workspace.dependencies]")?;
    let at = manifest[table..]
        .find('\n')
        .map_or(manifest.len(), |end| table + end + 1);
    let joined = format!(
        "{}{package} = {{ path = \"{folder}\" }}\n{}",
        &manifest[..at],
        &manifest[at..]
    );
    fs::write(&path, joined).map_err(|error| format!("cannot write Cargo.toml: {error}"))?;
    println!(
        "{}",
        json!({"event": "dependency_added", "package": package, "path": folder})
    );
    Ok(())
}
