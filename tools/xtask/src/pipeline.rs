use std::{fs, path::Path};

use crate::{
    authoring,
    process::{
        DEV_FEATURES, NPM, WASM_TARGET, client_crates, client_packages, run, verify_wasm_bindgen,
    },
};

/// The Cargo profile of builds nothing is timed on: the development server,
/// lint, agent sessions and asset tools (`Cargo.toml`).
pub(crate) const QUICK: &str = "quick";

pub fn build(root: &Path) -> Result<(), String> {
    authoring::validate_sources(root)?;
    build_wasm(root, false, "release")?;
    install_web_dependencies(root)?;
    run(root, NPM, &["run", "build", "--prefix", "web"])
}

/// What the development server serves, made: the authored sources checked,
/// the clients built with their development tools, the web dependencies
/// installed.
pub fn prepare_dev(root: &Path) -> Result<(), String> {
    authoring::validate_sources(root)?;
    build_wasm(root, true, QUICK)?;
    install_web_dependencies(root)
}

pub fn dev(root: &Path, port: u16) -> Result<(), String> {
    prepare_dev(root)?;
    println!("Serving the games at http://127.0.0.1:{port}; press Ctrl+C to stop.");
    println!(
        "For automatic Rust rebuilds and managed browser sessions, run `cargo xtask agent watch`."
    );
    run(
        root,
        NPM,
        &[
            "run",
            "dev",
            "--prefix",
            "web",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "--strictPort",
        ],
    )
}

/// The documentation of every item, private ones included, of the native
/// workspace and of the browser façade, which fails where the workspace's
/// rustdoc lints, all denied, find a link that does not resolve, public
/// docs linking a private item, or malformed markup.
pub fn docs(root: &Path) -> Result<(), String> {
    run(
        root,
        "cargo",
        &[
            "doc",
            "--workspace",
            "--all-features",
            "--no-deps",
            "--document-private-items",
            "--locked",
        ],
    )?;
    // Browser-only façade entry points do not exist in native rustdoc output.
    run(
        root,
        "cargo",
        &[
            "doc",
            "-p",
            "engine",
            "-p",
            "engine_render",
            "--features",
            "engine/dev",
            "--target",
            WASM_TARGET,
            "--no-deps",
            "--document-private-items",
            "--locked",
        ],
    )
}

/// The browser clients built with Cargo profile `profile`, their bindings in
/// `web/pkg-dev` with development tools or `web/pkg` without, beside a
/// `build.json` naming the profile and features.
pub(crate) fn build_wasm(
    root: &Path,
    development: bool,
    profile: &'static str,
) -> Result<(), String> {
    verify_wasm_bindgen(root)?;
    let mut arguments: Vec<&str> = vec!["build"];
    arguments.extend(client_packages());
    arguments.extend(["--target", WASM_TARGET, "--profile", profile, "--locked"]);
    if development {
        arguments.extend(["--features", DEV_FEATURES]);
    }
    run(root, "cargo", &arguments)?;
    let out_dir = if development {
        "web/pkg-dev"
    } else {
        "web/pkg"
    };
    for name in client_crates() {
        let artifact = format!("target/{WASM_TARGET}/{profile}/{name}.wasm");
        let mut arguments = vec![
            artifact.as_str(),
            "--target",
            "web",
            "--out-dir",
            out_dir,
            "--out-name",
            name,
        ];
        // Function names are a fifth of a module: development builds keep
        // them for stack traces and profiles.
        if !development {
            arguments.push("--remove-name-section");
        }
        run(root, "wasm-bindgen", &arguments)?;
    }
    let features: &[&str] = if development { &[DEV_FEATURES] } else { &[] };
    let identity = serde_json::json!({"profile": profile, "features": features});
    fs::write(
        root.join(out_dir).join("build.json"),
        serde_json::to_string_pretty(&identity).expect("identities serialize") + "\n",
    )
    .map_err(|error| format!("cannot write {out_dir}/build.json: {error}"))
}

pub(crate) fn install_web_dependencies(root: &Path) -> Result<(), String> {
    let web = root.join("web");
    let mut fingerprint = fs::read(web.join("package.json"))
        .map_err(|error| format!("cannot read web/package.json: {error}"))?;
    fingerprint.push(0);
    fingerprint.extend(
        fs::read(web.join("package-lock.json"))
            .map_err(|error| format!("cannot read web/package-lock.json: {error}"))?,
    );
    let stamp = web.join("node_modules/.engine-dependencies");
    if fs::read(&stamp).is_ok_and(|installed| installed == fingerprint)
        && web.join("node_modules/vite/package.json").is_file()
        && web.join("node_modules/typescript/package.json").is_file()
    {
        eprintln!("Web dependencies match the locked manifests; reusing installation.");
        return Ok(());
    }
    crate::process::run_to_stderr(root, NPM, &["ci", "--prefix", "web"]).map_err(|error|
        format!("{error}; stop this workspace's Vite server before reinstalling changed dependencies (Windows may lock native modules)"))?;
    fs::write(stamp, fingerprint)
        .map_err(|error| format!("cannot record installed dependency manifests: {error}"))
}
