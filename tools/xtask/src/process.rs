use std::{
    path::{Path, PathBuf},
    process::Command,
};

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

/// A browser client: its Cargo package, its Wasm crate's name, and for
/// games the name tools select it by and the page serving it.
pub struct Client {
    pub package: String,
    pub wasm: String,
    pub game: Option<String>,
    pub page: Option<String>,
    /// Whether the client has a `schema` binary printing its generated
    /// files.
    pub schema: bool,
}

/// The browser clients: the workspace members whose manifests declare
/// `[package.metadata.engine]` (a game's `page` and `game`, or `worker`),
/// in Cargo's order. `cargo xtask schema` writes them for the web host and
/// scripts.
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
        if engine.is_null() {
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
        let (game, page) = (field("game"), field("page"));
        if game.is_some() != page.is_some() || (game.is_none() && engine["worker"] != true) {
            return Err(format!(
                "{name}'s [package.metadata.engine] declares a game's `page` and `game`, or `worker = true`"
            ));
        }
        clients.push(Client {
            package: name.to_owned(),
            wasm: wasm.to_owned(),
            game,
            page,
            schema,
        });
    }
    Ok(clients)
}

/// `-p PACKAGE` for every client.
pub fn client_packages() -> impl Iterator<Item = &'static str> {
    clients()
        .iter()
        .flat_map(|client| ["-p", client.package.as_str()])
}

/// Every client's Wasm crate name.
pub fn client_crates() -> Vec<&'static str> {
    clients()
        .iter()
        .map(|client| client.wasm.as_str())
        .collect()
}

/// The engine features of development builds, as `--features` takes them:
/// clients declare none of their own.
pub const DEV_FEATURES: &str = "engine/dev";

/// The clients as the web host and scripts read them.
pub fn clients_json() -> serde_json::Value {
    clients()
        .iter()
        .map(|client| {
            serde_json::json!({
                "package": client.package,
                "wasm": client.wasm,
                "game": client.game,
                "page": client.page,
            })
        })
        .collect()
}

/// The games tools select by name, as `a|b|c`.
pub fn game_names() -> String {
    clients()
        .iter()
        .filter_map(|client| client.game.as_deref())
        .collect::<Vec<_>>()
        .join("|")
}

/// The wasm-bindgen version Cargo.lock pins, which the CLI must match.
pub fn wasm_bindgen_version(root: &Path) -> Result<String, String> {
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

/// Whether wasm-bindgen is the version the bindings need: otherwise an error
/// code (`bindgen_unavailable` or `bindgen_version`) and how to fix it.
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
    let version = String::from_utf8_lossy(&output.stdout);
    let expected = format!("wasm-bindgen {version}");
    if version.trim() != expected {
        return Err((
            "bindgen_version",
            format!("expected {expected}, found {:?}; {install}", version.trim()),
        ));
    }
    Ok(())
}

pub fn verify_wasm_bindgen(root: &Path) -> Result<(), String> {
    check_wasm_bindgen(root).map_err(|(_, message)| message)
}

pub fn run(root: &Path, program: &str, arguments: &[&str]) -> Result<(), String> {
    println!("Running {program} {}", arguments.join(" "));
    let status = Command::new(program)
        .args(arguments)
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

/// Setup diagnostics use stderr so an agent command can reserve stdout for JSON.
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
