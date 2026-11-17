use std::{
    fs,
    path::{Path, PathBuf},
};

use engine::Engine;
use engine_core::{DrawnTable, ScenePasses};
use engine_registry::Registry;
use serde_json::json;

use crate::process::{Client, Project, WASM_TARGET, project_option, read_clients, run};

const USAGE: &str = "usage: cargo xtask new app --project DIR | new brick NAME --project DIR \
                     [--kind gameplay|drawn]";

const APP_TEMPLATE: &str = "crates/engine/examples/app_template";

pub(crate) const TEMPLATE_CRATE: &str = "app_template";

const TEMPLATE_KIND: &str = "kind = \"game\"\n";

const TEMPLATE_SCENE: &str = "template.scene.json";

const TEMPLATE_ONLY: [(&str, &str); 4] = [
    ("use crate::bricks::glimmer::GlimmerBrick;\n", ""),
    (".with::<GlimmerBrick>()", ""),
    (
        r#"  "materials": [
    {
      "id": "disc",
      "color": [1.0, 1.0, 1.0, 1.0],
      "slots": {
        "opacity": "glimmer"
      },
      "alpha_mode": "mask",
      "alpha_cutoff": 0.5,
      "cull": "none",
      "lighting": "unlit"
    }
  ],
"#,
        "",
    ),
    (
        r#"    {
      "id": "spark",
      "drawn": {
        "kind": "glimmer",
        "material": "disc",
        "values": {
          "floats": [0.5]
        }
      },
      "turning": {},
      "transform": {
        "translation": [0.0, 3.5, 0.0]
      }
    },
"#,
        "",
    ),
];

#[derive(Clone, Copy)]
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
            Self::Gameplay => &["engine", "engine_core", "schemars", "serde"],
            Self::Drawn => &["engine_core", "engine_ecs", "engine_render"],
        }
    }

    fn folder(self) -> String {
        format!("src/bricks/{}", self.template())
    }
}

const RESERVED: [&str; 2] = ["engine", "generated"];

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
            TEMPLATE_ONLY
                .iter()
                .fold(text, |text, (only, kept)| text.replace(only, kept))
                .replace(&format!("    pub mod {};\n", drawn.template()), "")
                .replace(TEMPLATE_CRATE, &wasm)
                .replace("Template", &pascal)
        };
        if text.contains(drawn.template()) {
            return Err(format!(
                "{APP_TEMPLATE}/{} still names {} once its template-only parts are removed",
                inside.display(),
                drawn.template()
            ));
        }
        files.push((crate_dir.join(inside), text));
    }
    files.extend([
        (root.join("web").join(&page), page_html(&name, &pascal)),
        (web.join("main.ts"), page_entry(&wasm, &pascal)),
        (web.join("worker.ts"), worker_entry(&wasm)),
    ]);
    files.retain(|(path, _)| !path.exists());
    write_files(root, &files)?;
    join_workspace(root, &project)?;
    join_dependencies(root, &name, &project)?;
    run(
        root,
        "cargo",
        &["check", "-p", &name, "--target", WASM_TARGET],
    )?;
    run(root, "cargo", &["run", "-q", "-p", "kinds"])?;
    crate::authoring::generate(root, &[])?;
    println!(
        "{}",
        json!({
            "event": "app_created",
            "project": project,
            "page": format!("web/{page}"),
            "native": format!("{name}-native"),
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
        .replace(&format!("native = \"{TEMPLATE_CRATE}-native\"\n"), "")
        .replace(TEMPLATE_CRATE, name)
        .replace(TEMPLATE_KIND, &engine))
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

struct BrickArguments {
    project: String,
    name: String,
    kind: BrickKind,
}

fn brick_arguments(options: &[String]) -> Result<BrickArguments, String> {
    let (project, rest) = project_option(options)?;
    let (Some(project), Some((name, rest))) = (project, rest.split_first()) else {
        return Err(USAGE.into());
    };
    let mut kind = BrickKind::Gameplay;
    let mut given = rest.iter();
    while let Some(flag) = given.next() {
        match (flag.as_str(), given.next()) {
            ("--kind", Some(value)) => kind = BrickKind::named(value)?,
            _ => return Err(USAGE.into()),
        }
    }
    if !lowercase_words(name, '_') {
        return Err(json!({
            "code": "invalid_name",
            "path": "NAME",
            "got": name,
            "expected": "lowercase words of letters and digits joined by underscores",
        })
        .to_string());
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
        kind,
    })
}

fn brick(root: &Path, options: &[String]) -> Result<(), String> {
    let BrickArguments {
        project,
        name,
        kind,
    } = brick_arguments(options)?;
    let name = &name;
    let project = project
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_owned();
    let game = game_crate(root, &project)?;
    let module = format!("{project}/src/bricks/{name}");
    if root.join(&module).exists() {
        return Err(format!("{module} already exists"));
    }
    let engine = crate::bricks::standard_engine().map_err(|error| error.to_string())?;
    taken(root, &project, name, &engine)?;
    let named = pascal(name, '_');
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
        files.push((root.join(&module).join(inside), rename(&text)));
    }
    write_files(root, &files)?;
    let listed = declare_module(root, &project, name, &named)?;
    need_dependencies(root, &project, kind)?;
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

fn edited(path: &Path, edit: impl FnOnce(&str) -> String) -> Result<(), String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
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

fn need_dependencies(root: &Path, project: &str, kind: BrickKind) -> Result<(), String> {
    let path = root.join(project).join("Cargo.toml");
    let needs = kind.needs();
    let mut added = Vec::new();
    edited(&path, |text| {
        let mut text = text.to_owned();
        let table = "\n[dependencies]\n";
        for &need in needs {
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
