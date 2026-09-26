use std::{fs, path::Path};

use crate::{
    authoring,
    process::{NPM, WASM_TARGET, run, verify_wasm_bindgen},
};

pub fn build(root: &Path) -> Result<(), String> {
    authoring::validate_sources(root)?;
    build_wasm(root, false)?;
    install_web_dependencies(root)?;
    run(root, NPM, &["run", "build", "--prefix", "web"])
}

pub fn dev(root: &Path, port: u16) -> Result<(), String> {
    authoring::validate_sources(root)?;
    build_wasm(root, true)?;
    install_web_dependencies(root)?;
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

pub fn lint(root: &Path) -> Result<(), String> {
    authoring::validate_sources(root)?;
    authoring::check_schema(root)?;
    run(root, "cargo", &["fmt", "--all", "--", "--check"])?;
    run(
        root,
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--locked",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    run(
        root,
        "cargo",
        &[
            "clippy",
            "-p",
            "orbit-demo",
            "-p",
            "playground",
            "-p",
            "stress-lab",
            "--target",
            WASM_TARGET,
            "--locked",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    run(
        root,
        "cargo",
        &[
            "clippy",
            "-p",
            "orbit-demo",
            "-p",
            "playground",
            "-p",
            "stress-lab",
            "--features",
            "orbit-demo/dev,playground/dev,stress-lab/dev",
            "--target",
            WASM_TARGET,
            "--locked",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    // The host imports generated declarations, including on a clean checkout.
    build_wasm(root, false)?;
    build_wasm(root, true)?;
    install_web_dependencies(root)?;
    run(root, NPM, &["run", "check", "--prefix", "web"])
}

pub fn docs(root: &Path) -> Result<(), String> {
    run(
        root,
        "cargo",
        &[
            "doc",
            "--workspace",
            "--all-features",
            "--no-deps",
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
            "--locked",
        ],
    )
}

pub fn test(root: &Path) -> Result<(), String> {
    run(
        root,
        "cargo",
        &["test", "--workspace", "--all-features", "--locked"],
    )?;
    run(
        root,
        "node",
        &[
            "--test",
            "web/scripts/input.test.mjs",
            "web/scripts/agent.test.mjs",
            "web/scripts/dev.test.mjs",
            "web/scripts/source.test.mjs",
            "web/scripts/iteration-watch.test.mjs",
            "web/scripts/iteration-controller.test.mjs",
            "web/scripts/iteration-bridge.test.mjs",
            "web/scripts/benchmark-report.test.mjs",
        ],
    )
}

pub(crate) fn build_wasm(root: &Path, development: bool) -> Result<(), String> {
    verify_wasm_bindgen(root)?;
    let mut arguments = vec![
        "build",
        "-p",
        "orbit-demo",
        "-p",
        "playground",
        "-p",
        "stress-lab",
        "--target",
        WASM_TARGET,
        "--release",
        "--locked",
    ];
    if development {
        arguments.extend(["--features", "orbit-demo/dev,playground/dev,stress-lab/dev"]);
    }
    run(root, "cargo", &arguments)?;
    for name in ["orbit_demo", "playground", "stress_lab"] {
        let artifact = format!("target/{WASM_TARGET}/release/{name}.wasm");
        run(
            root,
            "wasm-bindgen",
            &[
                &artifact,
                "--target",
                "web",
                "--out-dir",
                if development {
                    "web/pkg-dev"
                } else {
                    "web/pkg"
                },
                "--out-name",
                name,
            ],
        )?;
    }
    Ok(())
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
