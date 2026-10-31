use std::{fs, path::Path};

use serde_json::json;

use crate::process::{WASM_TARGET, clients, project_option, run};

const USAGE: &str = "usage: cargo xtask new app --project DIR | new brick PART NAME";

const APP_TEMPLATE: &str = "crates/engine/examples/app_template.rs";

const APP_NATIVE: &str = "crates/engine_native/examples/app_native.rs";

const APP_XTASK: &str = "crates/engine_checks/examples/app_xtask.rs";

const BRICK_TEMPLATE: &str = "crates/engine_core/examples/brick_template.rs";

const TEMPLATE_GAME: &str = "#[path = \"../../engine/examples/app_template.rs\"]\nmod app;";

const TEMPLATE_IMPORTS: &str = "use engine_settings::{InvalidSetting, settings};\n";

const CORE_IMPORTS: &str = "use engine_core::{Brick, BrickStats, Bricks, Engine, EnvironmentSection, Error, Presentation};\n";

const SLOT_IMPORTS: &str = "use engine_core::{
    Brick, BrickStats, Bricks, Engine, EnvironmentSection, Error, Fill, Presentation,
};
";

const LIST_IMPORTS: &str = "use engine_core::{
    Brick, BrickStats, Bricks, Engine, EnvironmentSection, Error, Join, Presentation,
};
";

const GPU_IMPORTS: &str = "use engine_render::linker::ShaderModule;
use engine_render::parts::{Create, GpuPart, PartDecl, PartEntry};
use engine_render::render_bricks::AddRenderParts;
";

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
    let source = template(root, APP_TEMPLATE)?.replace("Template", &pascal);
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
    join_workspace(root, &project)?;
    join_workspace(root, &format!("{project}/xtask"))?;
    join_dependencies(root, &name, &project)?;
    run(root, "cargo", &["check", "-p", &name])?;
    run(root, "cargo", &["check", "-p", &format!("{name}-xtask")])?;
    run(root, "cargo", &["xtask", "schema"])?;
    run(
        root,
        "cargo",
        &["run", "-p", &format!("{name}-xtask"), "--", "schema"],
    )?;
    println!(
        "{}",
        json!({
            "event": "app_created",
            "project": project,
            "page": format!("web/{page}"),
            "native": format!("{name}-native"),
            "xtask": format!("{name}-xtask"),
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
page = "{page}"
game = "{name}"
native = "{name}-native"

[features]
default = []
dev = ["engine/dev", "engine_native/dev"]
profiling = ["engine/profiling", "engine_native/profiling"]

[dependencies]
engine.workspace = true

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
    let package = format!("engine_{name}");
    let folder = format!("crates/{package}");
    let repository = crate::rules::repository(root)?;
    if root.join(&folder).exists() || repository.package(&package).is_some() {
        return Err(format!("{folder} already exists"));
    }
    let pascal = pascal(name, '_');
    let claim = part_claim(root, repository, part)?;
    let mut source = template(root, BRICK_TEMPLATE)?
        .replace("Template", &pascal)
        .replace("\"template\"", &format!("\"{name}\""));
    let stats = format!(".stats::<{pascal}Stats>()");
    let mut dependencies = String::new();
    let mut metadata = String::new();
    match &claim {
        Some(claim) if claim.kind == "slot" => {
            let fill = format!("fill(Fill::<{}>::new())", claim.path);
            source = place_source(&source, SLOT_IMPORTS, &stats, &fill)?;
            if claim.gpu {
                source = gpu_source(&source, &pascal, name, &stats);
            }
        }
        Some(claim) => {
            let join = format!("join(Join::<{}>::new())", claim.path);
            source = place_source(&source, LIST_IMPORTS, &stats, &join)?;
        }
        None => println!(
            "{}",
            json!({"event": "part_undeclared", "part": part, "brick": name})
        ),
    }
    if let Some(claim) = &claim {
        if claim.package != "engine_core" {
            dependencies = format!("{}.workspace = true\n", claim.package);
        }
        if claim.gpu {
            dependencies.push_str("engine_render.workspace = true\n");
        }
        if let Some(replaced) = &claim.replaces {
            metadata = format!("\n[package.metadata.engine]\nreplaces = \"{replaced}\"\n");
        }
    }
    let manifest = format!(
        r#"[package]
name = "{package}"
version.workspace = true
edition.workspace = true
rust-version.workspace = true
publish.workspace = true

[features]
default = []
schema = ["dep:schemars", "engine_core/schema", "engine_settings/schema"]

[dependencies]
engine_core.workspace = true
engine_settings.workspace = true
{dependencies}bevy_ecs.workspace = true
serde.workspace = true
schemars = {{ workspace = true, optional = true }}

[lints]
workspace = true
{metadata}"#
    );
    let crate_dir = root.join(&folder);
    write_files(
        root,
        &[
            (crate_dir.join("Cargo.toml"), manifest),
            (crate_dir.join("src/lib.rs"), source),
        ],
    )?;
    join_workspace(root, &folder)?;
    join_dependencies(root, &package, &folder)?;
    run(root, "cargo", &["check", "-p", &package])?;
    run(
        root,
        "cargo",
        &["check", "-p", &package, "--features", "schema"],
    )?;
    run(
        root,
        "cargo",
        &["check", "-p", &package, "--target", WASM_TARGET],
    )?;
    println!(
        "{}",
        json!({
            "event": "brick_created",
            "crate": package,
            "brick": name,
            "part": part,
            "claims": claim.as_ref().map(|claim| claim.kind),
            "replaces": claim.as_ref().and_then(|claim| claim.replaces.clone()),
        })
    );
    Ok(())
}

fn place_source(
    source: &str,
    imports: &str,
    stats: &str,
    registration: &str,
) -> Result<String, String> {
    if !source.contains(CORE_IMPORTS) {
        return Err(format!(
            "{BRICK_TEMPLATE} no longer imports engine_core as `{}`",
            CORE_IMPORTS.trim_end()
        ));
    }
    Ok(source
        .replace(CORE_IMPORTS, imports)
        .replace(stats, &format!("{stats}\n            .{registration}")))
}

struct Claim {
    kind: &'static str,
    path: String,
    package: String,
    replaces: Option<String>,
    gpu: bool,
}

fn gpu_source(source: &str, pascal: &str, name: &str, stats: &str) -> String {
    let part = format!(
        r#"
pub const MODULES: &[ShaderModule] = &[];

struct {pascal}Part;

impl GpuPart for {pascal}Part {{}}

fn create(_: &Create<'_>) -> Box<dyn GpuPart> {{
    Box::new({pascal}Part)
}}

pub static PART: PartEntry = PartEntry {{
    decl: PartDecl {{
        name: "{name}",
        brick: "{name}",
        passes: &[],
        resources: &[],
        hooks: &[],
        contributes: &[],
        resolve: None,
        labels: &[],
        lazy: false,
        needs: &[],
        piece: None,
    }},
    create,
}};
"#
    );
    let registered = format!(
        "{stats}\n            .render_part(&PART)\n            .shaders(MODULES)\n            \
         .places(engine_faces::places::PLACES)"
    );
    source
        .replace(
            TEMPLATE_IMPORTS,
            &format!("{GPU_IMPORTS}{TEMPLATE_IMPORTS}"),
        )
        .replace(stats, &registered)
        + &part
}

fn part_claim(
    root: &Path,
    repository: &engine_checks::Repository,
    part: &str,
) -> Result<Option<Claim>, String> {
    let declaration = format!("const NAME: &'static str = \"{part}\";");
    for path in engine_checks::files::sources(root)? {
        if !path.ends_with(".rs") || !path.starts_with("crates/") {
            continue;
        }
        let text = fs::read_to_string(root.join(&path))
            .map_err(|error| format!("cannot read {path}: {error}"))?;
        for (kind, marker) in [("slot", "impl Slot for "), ("list", "impl List for ")] {
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
                let face = engine_checks::faces::face(&package.folder)?;
                let module = module_of(&path, package.path.as_deref().unwrap_or_default());
                let inner = if face.reaches(std::slice::from_ref(&name)) || module.is_empty() {
                    name.clone()
                } else {
                    format!("{module}::{name}")
                };
                let filler = if kind == "slot" {
                    let engine = crate::bricks::standard_engine();
                    engine
                        .app
                        .world()
                        .get_resource::<engine::bricks::BrickRegistry>()
                        .and_then(|registry| {
                            registry
                                .bricks()
                                .iter()
                                .find(|brick| {
                                    brick.fills.iter().any(|filled| filled.place == part)
                                })
                                .map(|brick| {
                                    let render = brick
                                        .extension::<engine_render::render_bricks::RenderBrick>();
                                    (brick.crate_name.to_owned(), render.is_some())
                                })
                        })
                } else {
                    None
                };
                return Ok(Some(Claim {
                    kind,
                    path: format!("{}::{inner}", package.crate_name()),
                    package: package.name.clone(),
                    gpu: filler.as_ref().is_some_and(|&(_, gpu)| gpu),
                    replaces: filler.map(|(replaced, _)| replaced),
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
