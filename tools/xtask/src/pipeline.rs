use std::{fs, path::Path};

use crate::process::{
    Client, DEV_FEATURES, NPM, Project, WASM_TARGET, clients, game_names, run, run_with,
    verify_wasm_bindgen,
};

const PROFILING: &str = "profiling";

const PROFILING_FEATURES: &str = "dev,profiling";

const PACKAGES: &str = "web/pkg";

pub fn build(root: &Path, profiling: bool) -> Result<(), String> {
    build_clients(
        root,
        &clients().iter().collect::<Vec<_>>(),
        false,
        true,
        PACKAGES,
    )?;
    let mut arguments = vec!["run", "build", "--prefix", "web"];
    if profiling {
        build_profiling_wasm(root)?;
        arguments.extend(["--", "--mode", PROFILING]);
    }
    install_web_dependencies(root)?;
    run(root, NPM, &arguments)?;
    if profiling {
        let record = serde_json::json!({
            "event": "built",
            "folder": "web/dist-profile",
            "serve": "npm run preview --prefix web -- --outDir dist-profile --port 5175 --strictPort",
        });
        println!("{record}");
    }
    Ok(())
}

pub fn build_page(root: &Path, client: &Client) -> Result<(), String> {
    build_clients(root, &[client], false, true, PACKAGES)?;
    install_web_dependencies(root)?;
    run_with(
        root,
        NPM,
        &["run", "build", "--prefix", "web"],
        &[("ENGINE_GAME", client.game.as_str())],
    )
}

pub fn build_profiling_wasm(root: &Path) -> Result<(), String> {
    let mut arguments = vec!["build"];
    for game in clients() {
        arguments.extend(["-p", game.package.as_str()]);
    }
    arguments.extend(["--target", WASM_TARGET]);
    arguments.extend(["--profile", PROFILING, "--features", PROFILING_FEATURES]);
    arguments.push("--locked");
    run(root, "cargo", &arguments)?;
    for game in clients() {
        run(
            root,
            "wasm-bindgen",
            &[
                &format!("target/{WASM_TARGET}/{PROFILING}/{}.wasm", game.wasm),
                "--target",
                "web",
                "--out-dir",
                "web/pkg-profile",
                "--out-name",
                &game.wasm,
                "--keep-debug",
            ],
        )?;
    }
    Ok(())
}

pub fn build_dev(root: &Path, game: Option<&str>) -> Result<(), String> {
    let selected = match game {
        Some(game) => vec![of_game(game)?],
        None => clients().iter().collect(),
    };
    build_clients(root, &selected, true, false, "web/pkg-dev")
}

pub fn build_native(root: &Path, project: &Project) -> Result<String, String> {
    native(root, project, &["--features", DEV_FEATURES], "debug", &[])
}

pub fn build_profiling(root: &Path, project: &Project) -> Result<String, String> {
    let flags = format!(
        "{} -Cforce-frame-pointers=yes",
        std::env::var("RUSTFLAGS").unwrap_or_default()
    );
    let frame_pointers = [("RUSTFLAGS", flags.trim())];
    let environment: &[(&str, &str)] = if cfg!(target_os = "linux") {
        &frame_pointers
    } else {
        &[]
    };
    native(
        root,
        project,
        &["--profile", PROFILING, "--features", PROFILING_FEATURES],
        PROFILING,
        environment,
    )
}

fn native(
    root: &Path,
    project: &Project,
    build: &[&str],
    folder: &str,
    environment: &[(&str, &str)],
) -> Result<String, String> {
    let mut arguments = vec!["build", "-p", project.package.as_str()];
    arguments.extend(["--bin", project.native.as_str()]);
    arguments.extend(build);
    arguments.push("--locked");
    run_with(root, "cargo", &arguments, environment)?;
    let program = root.join("target").join(folder).join(project.program());
    Ok(program.to_string_lossy().into_owned())
}

fn of_game(game: &str) -> Result<&'static Client, String> {
    clients()
        .iter()
        .find(|client| client.game == game)
        .ok_or_else(|| {
            serde_json::json!({
                "code": "invalid_name",
                "path": "--game",
                "got": game,
                "expected": game_names(),
            })
            .to_string()
        })
}

pub fn build_wasm(root: &Path) -> Result<(), String> {
    build_clients(
        root,
        &clients().iter().collect::<Vec<_>>(),
        false,
        false,
        PACKAGES,
    )
}

fn build_clients(
    root: &Path,
    selected: &[&Client],
    development: bool,
    release: bool,
    out_dir: &str,
) -> Result<(), String> {
    verify_wasm_bindgen(root)?;
    let folder = if release { "release" } else { "debug" };
    let mut arguments: Vec<&str> = vec!["build"];
    for client in selected {
        arguments.extend(["-p", client.package.as_str()]);
    }
    arguments.extend(["--target", WASM_TARGET, "--locked"]);
    if release {
        arguments.push("--release");
    }
    if development {
        arguments.extend(["--features", DEV_FEATURES]);
    }
    run(root, "cargo", &arguments)?;
    for client in selected {
        let name = client.wasm.as_str();
        let artifact = format!("target/{WASM_TARGET}/{folder}/{name}.wasm");
        let mut arguments = vec![
            artifact.as_str(),
            "--target",
            "web",
            "--out-dir",
            out_dir,
            "--out-name",
            name,
        ];
        if !development {
            arguments.push("--remove-name-section");
        }
        run(root, "wasm-bindgen", &arguments)?;
    }
    Ok(())
}

pub fn install_web_dependencies(root: &Path) -> Result<(), String> {
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
        eprintln!(
            "{}",
            serde_json::json!({"event": "web_dependencies_reused"})
        );
        return Ok(());
    }
    crate::process::run_to_stderr(root, NPM, &["ci", "--prefix", "web"]).map_err(|error|
        format!("{error}; stop this workspace's Vite server before reinstalling changed dependencies (Windows may lock native modules)"))?;
    fs::write(stamp, fingerprint)
        .map_err(|error| format!("cannot record installed dependency manifests: {error}"))
}
