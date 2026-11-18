use std::{
    path::{Path, PathBuf},
    process::Command,
};

use sha2::{Digest, Sha256};

pub const WASM_TARGET: &str = "wasm32-unknown-unknown";

#[cfg(windows)]
pub const NPM: &str = "npm.cmd";
#[cfg(not(windows))]
pub const NPM: &str = "npm";

pub fn repository_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "cannot locate the workspace above tools/xtask".to_owned())
}

pub struct Client {
    pub package: String,
    pub folder: String,
    pub wasm: String,
    pub game: Option<String>,
    pub page: Option<String>,
    pub native: Option<String>,
}

pub fn clients() -> &'static [Client] {
    static CLIENTS: std::sync::OnceLock<Vec<Client>> = std::sync::OnceLock::new();
    CLIENTS.get_or_init(|| {
        read_clients().unwrap_or_else(|error| panic!("browser_clients_listed {error}"))
    })
}

fn engine_packages() -> Result<&'static [(serde_json::Value, String)], String> {
    static PACKAGES: std::sync::OnceLock<Result<Vec<(serde_json::Value, String)>, String>> =
        std::sync::OnceLock::new();
    PACKAGES
        .get_or_init(read_engine_packages)
        .as_deref()
        .map_err(Clone::clone)
}

fn read_engine_packages() -> Result<Vec<(serde_json::Value, String)>, String> {
    let root = repository_root()?;
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps", "--locked"])
        .current_dir(&root)
        .output()
        .map_err(|error| format!("cannot run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("cargo metadata is not JSON: {error}"))?;
    let mut packages = Vec::new();
    for package in metadata["packages"].as_array().into_iter().flatten() {
        if !package["metadata"]["engine"].is_object() {
            continue;
        }
        let name = package["name"].as_str().unwrap_or_default();
        let manifest = Path::new(package["manifest_path"].as_str().unwrap_or_default());
        let folder = manifest
            .parent()
            .and_then(|folder| folder.strip_prefix(&root).ok())
            .map(slashed)
            .ok_or_else(|| format!("{name}'s manifest lies outside the workspace"))?;
        packages.push((package.clone(), folder));
    }
    Ok(packages)
}

pub fn native_folders() -> Result<Vec<String>, String> {
    Ok(engine_packages()?
        .iter()
        .filter(|(package, _)| package["metadata"]["engine"]["native"].is_string())
        .map(|(_, folder)| folder.clone())
        .collect())
}

pub fn read_clients() -> Result<Vec<Client>, String> {
    let mut clients = Vec::new();
    for (package, folder) in engine_packages()? {
        let folder = folder.clone();
        let engine = &package["metadata"]["engine"];
        if engine["game"].is_null() && engine["page"].is_null() && engine["worker"] != true {
            continue;
        }
        let name = package["name"].as_str().unwrap_or_default();
        let wasm = package["targets"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|target| {
                target["crate_types"]
                    .as_array()
                    .is_some_and(|types| types.iter().any(|kind| kind == "cdylib"))
            })
            .and_then(|target| target["name"].as_str())
            .ok_or_else(|| {
                format!("{name} declares [package.metadata.engine] but builds no cdylib")
            })?;
        let field = |key: &str| engine[key].as_str().map(str::to_owned);
        let (game, page, native) = (field("game"), field("page"), field("native"));
        if game.is_some() != page.is_some() || (game.is_none() && engine["worker"] != true) {
            return Err(format!(
                "{name}'s [package.metadata.engine] declares a game's `page` and `game`, or `worker = true`"
            ));
        }
        clients.push(Client {
            package: name.to_owned(),
            folder,
            wasm: wasm.to_owned(),
            game,
            page,
            native,
        });
    }
    Ok(clients)
}

pub const DEV_FEATURES: &str = "engine/dev";

pub fn clients_json() -> serde_json::Value {
    clients()
        .iter()
        .map(|client| {
            serde_json::json!({
                "package": client.package,
                "folder": client.folder,
                "wasm": client.wasm,
                "game": client.game,
                "page": client.page,
                "native": client.native,
            })
        })
        .collect()
}

pub fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn slashed(path: &Path) -> String {
    path.components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

pub struct Project {
    pub package: String,
    pub native: String,
    pub scenes: Vec<String>,
}

impl Project {
    pub fn read(root: &Path, folder: &str) -> Result<Self, String> {
        let manifest = format!("{folder}/Cargo.toml");
        let text = std::fs::read_to_string(root.join(&manifest))
            .map_err(|error| format!("--project {folder}: cannot read {manifest}: {error}"))?;
        Self::parse(&manifest, text)
    }

    fn parse(manifest: &str, text: String) -> Result<Self, String> {
        let document = toml_edit::Document::parse(text)
            .map_err(|error| format!("{manifest} is not TOML: {error}"))?;
        let package = document.as_item().get("package");
        let name = package
            .and_then(|package| package.get("name"))
            .and_then(toml_edit::Item::as_str)
            .ok_or_else(|| format!("{manifest}: [package] has no name"))?;
        let engine = package
            .and_then(|package| package.get("metadata"))
            .and_then(|metadata| metadata.get("engine"));
        let native = engine
            .and_then(|engine| engine.get("native"))
            .and_then(toml_edit::Item::as_str)
            .ok_or_else(|| format!("{manifest}: [package.metadata.engine] declares no native"))?;
        let scenes = match engine.and_then(|engine| engine.get("scenes")) {
            None => Vec::new(),
            Some(listed) => listed
                .as_array()
                .and_then(|scenes| {
                    scenes
                        .iter()
                        .map(|scene| scene.as_str().map(str::to_owned))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| {
                    format!(
                        "{manifest}: [package.metadata.engine] scenes lists scene files by \
                         paths relative to the project's folder"
                    )
                })?,
        };
        Ok(Self {
            package: name.to_owned(),
            native: native.to_owned(),
            scenes,
        })
    }

    pub fn program(&self) -> String {
        format!("{}{}", self.native, std::env::consts::EXE_SUFFIX)
    }
}

pub fn project_option(arguments: &[String]) -> Result<(Option<String>, Vec<String>), String> {
    let mut project = None;
    let mut rest = Vec::new();
    let mut given = arguments.iter();
    while let Some(argument) = given.next() {
        if argument == "--project" {
            let folder = given
                .next()
                .ok_or("--project takes a game crate's folder")?;
            if project.replace(folder.replace('\\', "/")).is_some() {
                return Err("--project is given once".into());
            }
        } else {
            rest.push(argument.clone());
        }
    }
    Ok((project, rest))
}

pub fn game_names() -> String {
    clients()
        .iter()
        .filter_map(|client| client.game.as_deref())
        .collect::<Vec<_>>()
        .join("|")
}

fn wasm_bindgen_version(root: &Path) -> Result<String, String> {
    let lock = std::fs::read_to_string(root.join("Cargo.lock"))
        .map_err(|error| format!("cannot read Cargo.lock: {error}"))?;
    lock.split("[[package]]")
        .find(|package| {
            package
                .lines()
                .any(|line| line.trim() == "name = \"wasm-bindgen\"")
        })
        .and_then(|package| {
            package
                .lines()
                .find_map(|line| line.trim().strip_prefix("version = \""))
                .map(|version| version.trim_end_matches('"').to_owned())
        })
        .ok_or_else(|| "Cargo.lock pins no wasm-bindgen".to_owned())
}

pub fn verify_wasm_bindgen(root: &Path) -> Result<(), String> {
    let version = wasm_bindgen_version(root)?;
    let install = format!(
        "install the matching CLI with `cargo install wasm-bindgen-cli --version {version} --locked` and ensure it is on PATH"
    );
    let output = Command::new("wasm-bindgen")
        .arg("--version")
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run wasm-bindgen: {error}; {install}"))?;
    if !output.status.success() {
        return Err(format!(
            "wasm-bindgen --version failed ({}): {}; {install}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let printed = String::from_utf8_lossy(&output.stdout);
    let found = printed.trim();
    let expected = format!("wasm-bindgen {version}");
    if found != expected {
        return Err(format!("expected {expected}, found {found:?}; {install}"));
    }
    Ok(())
}

pub fn run(root: &Path, program: &str, arguments: &[&str]) -> Result<(), String> {
    run_with(root, program, arguments, &[])
}

pub fn run_with(
    root: &Path,
    program: &str,
    arguments: &[&str],
    environment: &[(&str, &str)],
) -> Result<(), String> {
    let record = serde_json::json!({
        "event": "running",
        "program": program,
        "arguments": arguments,
    });
    println!("{record}");
    let status = Command::new(program)
        .args(arguments)
        .envs(environment.iter().copied())
        .current_dir(root)
        .status()
        .map_err(|error| {
            format!("cannot start {program}: {error}; ensure it is installed and available on PATH")
        })?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "{program} {} failed ({status}); see the command output above",
            arguments.join(" ")
        ))
    }
}

pub fn run_to_stderr(root: &Path, program: &str, arguments: &[&str]) -> Result<(), String> {
    let record = serde_json::json!({
        "event": "running",
        "program": program,
        "arguments": arguments,
    });
    eprintln!("{record}");
    let mut child = Command::new(program)
        .args(arguments)
        .current_dir(root)
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot start {program}: {error}"))?;
    if let Some(mut output) = child.stdout.take() {
        std::io::copy(&mut output, &mut std::io::stderr())
            .map_err(|error| format!("cannot forward setup output: {error}"))?;
    }
    let status = child.wait().map_err(|error| error.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} failed ({status})"))
    }
}
