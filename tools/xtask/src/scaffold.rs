use std::{fs, path::Path};

use engine_checks::Repository;
use engine_place::PlaceKind;
use engine_registry::Registry;
use serde_json::json;

use crate::process::{WASM_TARGET, clients, project_option, run};

const USAGE: &str = "usage: cargo xtask new app --project DIR | new brick PART NAME";

const APP_TEMPLATE: &str = "crates/engine/examples/app_template.rs";

const APP_NATIVE: &str = "crates/engine_native/examples/app_native.rs";

const APP_XTASK: &str = "crates/engine_checks/examples/app_xtask.rs";

const TEMPLATE_GAME: &str = "#[path = \"../../engine/examples/app_template.rs\"]\nmod app;";

const TEMPLATE_LIST: &str = "pub fn bricks<L: Listing>(list: L) -> L {\n    \
                             engine_standard::standard(list).with::<FlyCameraBrick>()\n}\n";

const TEMPLATE_USE: &str = "use engine_fly_camera::FlyCameraBrick;\n";

const LIST_USES: &str = "use engine_core::Listing;\nuse engine_fly_camera::FlyCameraBrick;\n\n";

const LIST_DEPENDENCIES: &str = "engine_core.workspace = true\n\
                                 engine_fly_camera.workspace = true\n\
                                 engine_standard.workspace = true\n";

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

fn write_files(root: &Path, files: &[(std::path::PathBuf, String)]) -> Result<(), String> {
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

fn template(root: &Path, path: &str) -> Result<String, String> {
    fs::read_to_string(root.join(path))
        .map(|text| text.replace("\r\n", "\n"))
        .map_err(|error| format!("cannot read {path}: {error}"))
}

fn app(root: &Path, options: &[String]) -> Result<(), String> {
    let (project, rest) = project_option(options)?;
    let (Some(project), true) = (project, rest.is_empty()) else {
        return Err(USAGE.into());
    };
    let project = project.trim_end_matches('/').to_owned();
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
    for taken in [
        crate_dir.clone(),
        root.join("web").join(&page),
        root.join("web/src").join(&name),
    ] {
        if taken.exists() {
            return Err(format!("{} already exists", taken.display()));
        }
    }
    if clients()
        .iter()
        .any(|client| client.game.as_deref() == Some(name.as_str()) || client.wasm == wasm)
    {
        return Err(format!("a client named {name} already exists"));
    }
    let source = template(root, APP_TEMPLATE)?;
    for (marker, what) in [(TEMPLATE_LIST, "lists its bricks"), (TEMPLATE_USE, "names them")] {
        if !source.contains(marker) {
            return Err(format!(
                "{APP_TEMPLATE} no longer {what} as `{}`",
                marker.trim_end()
            ));
        }
    }
    let source = source
        .replace(TEMPLATE_USE, "")
        .replace(TEMPLATE_LIST, &format!("pub use {wasm}_bricks::bricks;\n"))
        .replace("Template", &pascal);
    let native = template(root, APP_NATIVE)?;
    let xtask = template(root, APP_XTASK)?;
    for (path, text) in [(APP_NATIVE, &native), (APP_XTASK, &xtask)] {
        if !text.contains(TEMPLATE_GAME) {
            return Err(format!(
                "{path} no longer reaches the app template as `{TEMPLATE_GAME}`"
            ));
        }
    }
    let game = format!("use {wasm} as app;");
    let files = [
        (crate_dir.join("Cargo.toml"), app_manifest(&name, &page)),
        (crate_dir.join("src/lib.rs"), source),
        (
            crate_dir.join("src/main.rs"),
            native
                .replace(TEMPLATE_GAME, &game)
                .replace("Template", &pascal),
        ),
        (
            crate_dir.join("bricks/Cargo.toml"),
            list_manifest(&name, LIST_DEPENDENCIES),
        ),
        (
            crate_dir.join("bricks/src/lib.rs"),
            format!("{LIST_USES}{TEMPLATE_LIST}"),
        ),
        (crate_dir.join("worker/Cargo.toml"), worker_manifest(&name)),
        (
            crate_dir.join("worker/src/lib.rs"),
            format!("engine_worker::export_worker!({wasm}_bricks::bricks);\n"),
        ),
        (crate_dir.join("xtask/Cargo.toml"), xtask_manifest(&name)),
        (
            crate_dir.join("xtask/src/main.rs"),
            xtask.replace(TEMPLATE_GAME, &game),
        ),
        (
            crate_dir.join(".cargo/config.toml"),
            format!("[alias]\nxtask = \"run -p {name}-xtask --\"\n"),
        ),
        (root.join("web").join(&page), page_html(&name, &pascal)),
        (
            root.join("web/src").join(&name).join("main.ts"),
            page_entry(&wasm, &pascal),
        ),
    ];
    write_files(root, &files)?;
    for member in ["", "/bricks", "/worker", "/xtask"] {
        join_workspace(root, &format!("{project}{member}"))?;
    }
    let bricks = format!("{name}-bricks");
    let worker = format!("{name}-worker");
    let tool = format!("{name}-xtask");
    join_dependencies(root, &name, &project)?;
    join_dependencies(root, &bricks, &format!("{project}/bricks"))?;
    for package in [name.as_str(), bricks.as_str(), worker.as_str(), tool.as_str()] {
        run(root, "cargo", &["check", "-p", package])?;
    }
    for package in [name.as_str(), worker.as_str()] {
        run(root, "cargo", &["check", "-p", package, "--target", WASM_TARGET])?;
    }
    run(root, "cargo", &["run", "-q", "-p", "kinds"])?;
    run(root, "cargo", &["xtask", "schema"])?;
    run(root, "cargo", &["run", "-p", &tool, "--", "schema"])?;
    println!(
        "{}",
        json!({
            "event": "app_created",
            "project": project,
            "page": format!("web/{page}"),
            "native": format!("{name}-native"),
            "list": bricks,
            "worker": worker,
            "xtask": tool,
        })
    );
    Ok(())
}

fn app_manifest(name: &str, page: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[lib]
crate-type = ["cdylib", "rlib"]

[[bin]]
name = "{name}-native"
path = "src/main.rs"

[package.metadata.engine]
kind = "app"
page = "{page}"
game = "{name}"
native = "{name}-native"

[features]
default = []
dev = ["engine/dev", "engine_native/dev"]
profiling = ["engine/profiling", "engine_native/profiling"]

[dependencies]
engine.workspace = true
{name}-bricks.workspace = true
serde_json.workspace = true

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
engine_native.workspace = true

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen.workspace = true
wasm-bindgen-futures.workspace = true
web-sys.workspace = true

[lints]
workspace = true
"#
    )
}

fn list_manifest(name: &str, dependencies: &str) -> String {
    format!(
        r#"[package]
name = "{name}-bricks"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[dependencies]
{dependencies}
[lints]
workspace = true

[package.metadata.engine]
kind = "list"
"#
    )
}

fn worker_manifest(name: &str) -> String {
    format!(
        r#"[package]
name = "{name}-worker"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[lib]
crate-type = ["cdylib", "rlib"]

[package.metadata.engine]
kind = "app"
worker = true

[dependencies]
engine_worker.workspace = true
{name}-bricks.workspace = true

[lints]
workspace = true
"#
    )
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
{name}.workspace = true

[lints]
workspace = true

[package.metadata.engine]
kind = "tool"
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
        r#"import {{ loadGame }} from "../engine/bindings";
import {{ startGame }} from "../engine/host";

startGame({{
  async create(canvas) {{
    const {{ default: init, {pascal}Game }} = await loadGame("{wasm}", () => import("../../pkg/{wasm}"));
    await init();
    return {pascal}Game.create(canvas);
  }},
}});
"#
    )
}

enum Shape {
    Slot { contract: Option<&'static str> },
    List,
    Draw,
}

impl Shape {
    const fn name(&self) -> &'static str {
        match self {
            Self::Slot { .. } => "slot",
            Self::List => "list",
            Self::Draw => "draw_point",
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

fn brick(root: &Path, options: &[String]) -> Result<(), String> {
    let [part, name] = options else {
        return Err(USAGE.into());
    };
    for (path, given) in [("PART", part), ("NAME", name)] {
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
    let package = format!("engine_{name}");
    let folder = format!("crates/{package}");
    let repository = crate::rules::repository(root)?;
    if root.join(&folder).exists() || repository.package(&package).is_some() {
        return Err(format!("{folder} already exists"));
    }
    let engine = crate::bricks::standard_engine().map_err(|error| error.to_string())?;
    let registry = engine
        .app
        .world()
        .get_resource::<Registry>()
        .ok_or("the standard engine holds no brick registry")?;
    for key in [name.clone(), format!("{name}_stats")] {
        let owner = registry
            .section(&key)
            .map(|entry| entry.brick)
            .or_else(|| registry.stats_section(&key).map(|entry| entry.brick));
        if let Some(owner) = owner {
            return Err(json!({
                "code": "name_twice",
                "path": format!("$.sections.{key}"),
                "got": owner,
            })
            .to_string());
        }
    }
    let pascal = pascal(name, '_');
    let claim = claim(root, repository, registry, part)?;
    if claim.is_none() {
        println!(
            "{}",
            json!({"event": "part_undeclared", "part": part, "brick": name})
        );
    }
    let mut files = vec![
        (
            root.join(&folder).join("Cargo.toml"),
            brick_manifest(&package, claim.as_ref()),
        ),
        (
            root.join(&folder).join("build.rs"),
            "fn main() {\n    engine_sources::emit();\n}\n".to_owned(),
        ),
        (
            root.join(&folder).join("src/lib.rs"),
            brick_source(&pascal, name, claim.as_ref()),
        ),
    ];
    if let Some(Claim {
        shape: Shape::Slot {
            contract: Some(contract),
        },
        place,
        ..
    }) = &claim
    {
        files.push((
            root.join(&folder)
                .join(format!("src/shaders/{name}/faces/{place}.wgsl")),
            answers(place, contract),
        ));
    }
    write_files(root, &files)?;
    join_workspace(root, &folder)?;
    join_dependencies(root, &package, &folder)?;
    run(root, "cargo", &["fmt", "-p", &package])?;
    run(root, "cargo", &["check", "-p", &package])?;
    run(
        root,
        "cargo",
        &["check", "-p", &package, "--target", WASM_TARGET],
    )?;
    run(root, "cargo", &["run", "-q", "-p", "kinds"])?;
    println!(
        "{}",
        json!({
            "event": "brick_created",
            "crate": package,
            "brick": format!("{package}::{pascal}Brick"),
            "part": part,
            "claims": claim.as_ref().map(|claim| claim.shape.name()),
            "filled_by": claim.as_ref().and_then(|claim| claim.filler),
        })
    );
    Ok(())
}

fn brick_manifest(package: &str, claim: Option<&Claim>) -> String {
    let mut dependencies = String::new();
    let draws = claim.is_some_and(|claim| matches!(claim.shape, Shape::Draw));
    if let Some(claim) = claim
        && !matches!(claim.package.as_str(), "engine_core" | "engine_render")
    {
        dependencies.push_str(&format!("{}.workspace = true\n", claim.package));
    }
    if draws || claim.is_some_and(|claim| claim.package == "engine_render") {
        dependencies.push_str("engine_render.workspace = true\n");
    }
    format!(
        r#"[package]
name = "{package}"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[dependencies]
engine_ecs.workspace = true
engine_core.workspace = true
{dependencies}
[build-dependencies]
engine_sources.workspace = true

[lints]
workspace = true

[package.metadata.engine]
kind = "brick"
"#
    )
}

fn brick_source(pascal: &str, name: &str, claim: Option<&Claim>) -> String {
    let mut core = vec!["Brick", "Bricks", "Schedule", "Source", "settings", "stats"];
    let mut render = String::new();
    let mut registered = String::new();
    let mut part = String::new();
    match claim {
        Some(Claim {
            shape: Shape::Slot { contract },
            place,
            path,
            ..
        }) => {
            let wgsl = match contract {
                Some(_) => format!("Some(shaders::{name}::faces::{place}::MODULE)"),
                None => "None".to_owned(),
            };
            registered = format!("\n            .fill::<{path}>({wgsl})");
        }
        Some(Claim {
            shape: Shape::List,
            path,
            ..
        }) => {
            registered = format!("\n            .join::<{path}>(Default::default(), None)");
        }
        Some(Claim {
            shape: Shape::Draw,
            path,
            ..
        }) => {
            core.extend(["DrawPoint", "Error"]);
            render = "use engine_render::{Draw, GpuPart, Install, Layer, Prepare, RenderBricks};\n"
                .to_owned();
            registered = format!(
                "\n            .draws::<{pascal}Gpu, {path}>(Layer::Opaque, {pascal}Gpu::draw)"
            );
            part = format!(
                r#"
struct {pascal}Gpu;

impl GpuPart for {pascal}Gpu {{
    type Reads = ();

    fn install(_: &mut Install<'_, Self>) -> Result<Self, Error> {{
        Ok(Self)
    }}

    fn prepare(&mut self, _: SystemParamItem<'_, '_, Self::Reads>, _: &mut Prepare<'_, Self>) {{}}
}}

impl {pascal}Gpu {{
    fn draw(&self, _: &<{path} as DrawPoint>::Args, _: &mut Draw<'_, Self, {path}>) -> u32 {{
        0
    }}
}}
"#
            );
        }
        None => {}
    }
    core.sort_unstable();
    let core = core.join(", ");
    format!(
        r#"include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use bevy_ecs::prelude::*;
use engine_core::{{{core}}};
{render}
settings! {{
    #[section("{name}")]
    pub struct {pascal} {{
        pub enabled: bool = true => any,
    }}
}}

stats! {{
    #[section("{name}_stats")]
    pub struct {pascal}Stats {{
        pub frames: u32,
    }}
}}

fn enabled(settings: Res<{pascal}>) -> bool {{
    settings.enabled
}}

fn count_frames(mut stats: ResMut<{pascal}Stats>) {{
    stats.frames = stats.frames.wrapping_add(1);
}}

pub struct {pascal}Brick;

impl Brick for {pascal}Brick {{
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {{
        bricks
            .when(enabled)
            .settings::<{pascal}>()
            .stats::<{pascal}Stats>(){registered}
            .systems(Schedule::Presentation, count_frames);
    }}
}}
{part}"#
    )
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
    if kind != "place" && !core {
        return Err(format!(
            "{part} is declared in {} (kind {kind}), which a brick cannot depend on",
            declared.package
        ));
    }
    let shape = match (decl.kind(), declared.layer.as_deref()) {
        (PlaceKind::Slot, Some("Layer")) => Shape::Draw,
        (PlaceKind::Slot, Some(layer)) => {
            return Err(format!(
                "the draw point {part} orders its members by {layer}; register its draws by hand"
            ));
        }
        (PlaceKind::Slot, None) => Shape::Slot {
            contract: decl.wgsl(),
        },
        (PlaceKind::List, _) => Shape::List,
        (PlaceKind::Component, _) => return Ok(None),
    };
    let filler = match shape {
        Shape::Slot { .. } => registry.filler_named(part),
        Shape::List | Shape::Draw => None,
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
    layer: Option<String>,
}

fn declared(
    root: &Path,
    repository: &Repository,
    part: &str,
) -> Result<Option<Declared>, String> {
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
                    layer: draw_layer(&text, &name),
                }));
            }
        }
    }
    Ok(None)
}

fn draw_layer(text: &str, name: &str) -> Option<String> {
    let marker = format!("impl DrawPoint for {name} ");
    let rest = &text[text.find(&marker)? + marker.len()..];
    let block = &rest[..rest.find('}').unwrap_or(rest.len())];
    let layer = block.split_once("type Layer = ")?.1;
    Some(layer[..layer.find(';')?].trim().to_owned())
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
