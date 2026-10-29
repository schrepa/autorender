use std::{
    path::{Path, PathBuf},
    process::Command,
};

pub const WASM_TARGET: &str = "wasm32-unknown-unknown";

pub fn tools_target(root: &Path) -> PathBuf {
    root.join("target").join("tools")
}

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
    pub schema: bool,
}

pub fn clients() -> &'static [Client] {
    static CLIENTS: std::sync::OnceLock<Vec<Client>> = std::sync::OnceLock::new();
    CLIENTS.get_or_init(|| {
        read_clients().unwrap_or_else(|error| panic!("cannot list the browser clients: {error}"))
    })
}

fn read_clients() -> Result<Vec<Client>, String> {
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
    let mut clients = Vec::new();
    for package in metadata["packages"].as_array().into_iter().flatten() {
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
        let schema = package["targets"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|target| {
                target["name"] == "schema"
                    && target["kind"]
                        .as_array()
                        .is_some_and(|kinds| kinds.iter().any(|kind| kind == "bin"))
            });
        let field = |key: &str| engine[key].as_str().map(str::to_owned);
        let (game, page, native) = (field("game"), field("page"), field("native"));
        if game.is_some() != page.is_some() || (game.is_none() && engine["worker"] != true) {
            return Err(format!(
                "{name}'s [package.metadata.engine] declares a game's `page` and `game`, or `worker = true`"
            ));
        }
        let manifest = Path::new(package["manifest_path"].as_str().unwrap_or_default());
        let folder = manifest
            .parent()
            .and_then(|folder| folder.strip_prefix(&root).ok())
            .map(slashed)
            .ok_or_else(|| format!("{name}'s manifest lies outside the workspace"))?;
        clients.push(Client {
            package: name.to_owned(),
            folder,
            wasm: wasm.to_owned(),
            game,
            page,
            native,
            schema,
        });
    }
    Ok(clients)
}

pub fn client_packages() -> impl Iterator<Item = &'static str> {
    clients()
        .iter()
        .flat_map(|client| ["-p", client.package.as_str()])
}

pub fn client_crates() -> Vec<&'static str> {
    clients()
        .iter()
        .map(|client| client.wasm.as_str())
        .collect()
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

pub fn slashed(path: &Path) -> String {
    path.components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

pub struct Project {
    pub package: String,
    pub native: String,
}

impl Project {
    pub fn read(root: &Path, folder: &str) -> Result<Self, String> {
        let manifest = format!("{folder}/Cargo.toml");
        let text = std::fs::read_to_string(root.join(&manifest))
            .map_err(|error| format!("--project {folder}: cannot read {manifest}: {error}"))?;
        Self::parse(&manifest, text)
    }

    pub fn at(root: &Path, sha: &str, folder: &str) -> Result<Self, String> {
        let manifest = format!("{folder}/Cargo.toml");
        let output = Command::new("git")
            .args(["show", &format!("{sha}:{manifest}")])
            .current_dir(root)
            .output()
            .map_err(|error| format!("cannot run git: {error}"))?;
        if !output.status.success() {
            return Err(format!("project {folder}: {manifest} is not at {sha}"));
        }
        Self::parse(
            &manifest,
            String::from_utf8_lossy(&output.stdout).into_owned(),
        )
    }

    fn parse(manifest: &str, text: String) -> Result<Self, String> {
        let document = toml_edit::Document::parse(text)
            .map_err(|error| format!("{manifest} is not TOML: {error}"))?;
        let package = document.as_item().get("package");
        let name = package
            .and_then(|package| package.get("name"))
            .and_then(toml_edit::Item::as_str)
            .ok_or_else(|| format!("{manifest}: [package] has no name"))?;
        let native = package
            .and_then(|package| package.get("metadata"))
            .and_then(|metadata| metadata.get("engine"))
            .and_then(|engine| engine.get("native"))
            .and_then(toml_edit::Item::as_str)
            .ok_or_else(|| format!("{manifest}: [package.metadata.engine] declares no native"))?;
        Ok(Self {
            package: name.to_owned(),
            native: native.to_owned(),
        })
    }

    pub fn program(&self) -> String {
        format!("{}{}", self.native, std::env::consts::EXE_SUFFIX)
    }

    pub fn client(&self) -> Option<&'static Client> {
        clients()
            .iter()
            .find(|client| client.package == self.package)
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

pub fn check_wasm_bindgen(root: &Path) -> Result<(), (&'static str, String)> {
    let version = wasm_bindgen_version(root).map_err(|error| ("bindgen_version", error))?;
    let install = format!(
        "install the matching CLI with `cargo install wasm-bindgen-cli --version {version} --locked` and ensure it is on PATH"
    );
    let output = Command::new("wasm-bindgen")
        .arg("--version")
        .current_dir(root)
        .output()
        .map_err(|error| {
            (
                "bindgen_unavailable",
                format!("cannot run wasm-bindgen: {error}; {install}"),
            )
        })?;
    if !output.status.success() {
        return Err((
            "bindgen_unavailable",
            format!(
                "wasm-bindgen --version failed ({}): {}; {install}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ));
    }
    let printed = String::from_utf8_lossy(&output.stdout);
    let expected = format!("wasm-bindgen {version}");
    if printed.trim() != expected {
        return Err((
            "bindgen_version",
            format!("expected {expected}, found {:?}; {install}", printed.trim()),
        ));
    }
    Ok(())
}

pub fn verify_wasm_bindgen(root: &Path) -> Result<(), String> {
    check_wasm_bindgen(root).map_err(|(_, message)| message)
}

pub fn engine_commit() -> (&'static str, String) {
    static COMMIT: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let commit = COMMIT.get_or_init(|| {
        let git = |arguments: &[&str]| {
            Command::new("git")
                .args(arguments)
                .output()
                .ok()
                .filter(|output| output.status.success())
        };
        let Some(head) = git(&["rev-parse", "--short=12", "HEAD"]) else {
            return "unknown".to_owned();
        };
        let head = String::from_utf8_lossy(&head.stdout).trim().to_owned();
        let clean = git(&["diff", "--quiet", "HEAD"]).is_some();
        if clean { head } else { format!("{head}-dirty") }
    });
    ("ENGINE_COMMIT", commit.clone())
}

pub fn run(root: &Path, program: &str, arguments: &[&str]) -> Result<(), String> {
    println!("Running {program} {}", arguments.join(" "));
    let status = Command::new(program)
        .args(arguments)
        .envs([engine_commit()])
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
    eprintln!("Running {program} {}", arguments.join(" "));
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
