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
#[derive(Clone, Copy)]
pub struct Client {
    pub package: &'static str,
    pub wasm: &'static str,
    pub game: Option<&'static str>,
    pub page: Option<&'static str>,
}

/// The browser clients: the one list of them, which `cargo xtask schema`
/// writes for the web host and scripts.
pub const CLIENTS: [Client; 4] = [
    Client {
        package: "orbit-demo",
        wasm: "orbit_demo",
        game: Some("orbit"),
        page: Some("index.html"),
    },
    Client {
        package: "playground",
        wasm: "playground",
        game: Some("playground"),
        page: Some("playground.html"),
    },
    Client {
        package: "stress-lab",
        wasm: "stress_lab",
        game: Some("stress"),
        page: Some("stress.html"),
    },
    // The job worker pages run beside the games.
    Client {
        package: "engine_worker",
        wasm: "engine_worker",
        game: None,
        page: None,
    },
];

/// `-p PACKAGE` for every client.
pub fn client_packages<'a>() -> impl Iterator<Item = &'a str> {
    CLIENTS
        .into_iter()
        .flat_map(|client| ["-p", client.package])
}

/// Every client's Wasm crate name.
pub fn client_crates() -> [&'static str; CLIENTS.len()] {
    CLIENTS.map(|client| client.wasm)
}

/// Every client's `dev` feature, as `--features` takes them.
pub fn client_dev_features() -> String {
    CLIENTS
        .map(|client| format!("{}/dev", client.package))
        .join(",")
}

/// The clients as the web host and scripts read them.
pub fn clients_json() -> serde_json::Value {
    CLIENTS
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
    CLIENTS
        .iter()
        .filter_map(|client| client.game)
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
