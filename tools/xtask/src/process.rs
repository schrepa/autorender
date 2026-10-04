use std::{
    path::{Path, PathBuf},
    process::Command,
};

pub const WASM_BINDGEN_VERSION: &str = "0.2.127";
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

/// The browser clients: each one's Cargo package and Wasm crate name.
pub const CLIENTS: [(&str, &str); 3] = [
    ("orbit-demo", "orbit_demo"),
    ("playground", "playground"),
    ("stress-lab", "stress_lab"),
];

/// `-p PACKAGE` for every client.
pub fn client_packages<'a>() -> impl Iterator<Item = &'a str> {
    CLIENTS.into_iter().flat_map(|(package, _)| ["-p", package])
}

/// Every client's Wasm crate name.
pub fn client_crates() -> [&'static str; 3] {
    CLIENTS.map(|(_, name)| name)
}

/// Every client's `dev` feature, as `--features` takes them.
pub fn client_dev_features() -> String {
    CLIENTS
        .map(|(package, _)| format!("{package}/dev"))
        .join(",")
}

/// Whether wasm-bindgen is the version the bindings need: otherwise an error
/// code (`bindgen_unavailable` or `bindgen_version`) and how to fix it.
pub fn check_wasm_bindgen(root: &Path) -> Result<(), (&'static str, String)> {
    let install = format!(
        "install the matching CLI with `cargo install wasm-bindgen-cli --version {WASM_BINDGEN_VERSION} --locked` and ensure it is on PATH"
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
    let expected = format!("wasm-bindgen {WASM_BINDGEN_VERSION}");
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
