use std::{
    fs,
    path::{Path, PathBuf},
};

use engine_dev::authoring::{read_scene_source, validate_scene};

/// Validates the tracked fixture scenes the smokes load and any local scenes
/// in `web/public/scenes`, which a checkout need not have.
pub fn validate_sources(root: &Path) -> Result<(), String> {
    let mut paths = scene_paths(&root.join("web/scripts/fixtures"))?;
    if paths.is_empty() {
        return Err("no fixture scenes found in web/scripts/fixtures/*.scene.json".to_owned());
    }
    let local = root.join("web/public/scenes");
    if local.is_dir() {
        paths.extend(scene_paths(&local)?);
    }
    for path in &paths {
        let display = path.strip_prefix(root).unwrap_or(path).display();
        // The same bounded reader and validation as `cargo xtask scene validate`.
        let report = read_scene_source(path)
            .and_then(|source| validate_scene(&source))
            .map_err(|error| format!("{display}: {error}"))?;
        println!(
            "Validated {display}: scene={} revision={} authored_entities={} prefabs={} expanded_entities={}",
            report.id, report.revision, report.entities, report.prefabs, report.expanded_entities
        );
    }
    println!("Validated {} scene source(s).", paths.len());
    Ok(())
}

fn scene_paths(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = Vec::new();
    let entries = fs::read_dir(directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read scene directory entry: {error}"))?;
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".scene.json"))
            && path.is_file()
        {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// The files Rust generates for the web host, scripts and apps, by path
/// from the repository root: the development protocol's summary, the
/// browser clients, the frame stats' and game surface's types, and each
/// app's own, which its `schema` binary prints.
fn generated(root: &Path) -> Result<Vec<(String, String)>, String> {
    let clients = serde_json::to_string_pretty(&crate::process::clients_json())
        .expect("the client list serializes")
        + "
";
    let stats: serde_json::Value =
        serde_json::from_str(engine_dev::FRAME_STATS_SCHEMA).expect("generated schema is JSON");
    let mut files = vec![
        (
            "web/src/generated/protocol.json".to_owned(),
            engine_dev::PROTOCOL_SUMMARY.to_owned(),
        ),
        ("web/src/generated/clients.json".to_owned(), clients),
        (
            "web/src/generated/stats.ts".to_owned(),
            engine::tooling::interfaces(&stats, "FrameStats", "engine_core::FrameStats"),
        ),
        (
            "web/src/generated/runtime.ts".to_owned(),
            engine::runtime_typescript(),
        ),
        (
            "web/src/generated/coverage.json".to_owned(),
            crate::coverage::coverage(root)?,
        ),
    ];
    for client in crate::process::clients()
        .iter()
        .filter(|client| client.schema)
    {
        let output = std::process::Command::new("cargo")
            .args([
                "run",
                "-q",
                "--locked",
                "-p",
                &client.package,
                "--bin",
                "schema",
            ])
            .args(["--features", &format!("{}/schema", client.package)])
            .current_dir(root)
            .output()
            .map_err(|error| format!("cannot run {}'s schema binary: {error}", client.package))?;
        if !output.status.success() {
            return Err(format!(
                "{}'s schema binary failed: {}",
                client.package,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let printed: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&output.stdout).map_err(|error| {
                format!(
                    "{}'s schema binary printed no JSON object: {error}",
                    client.package
                )
            })?;
        for (path, contents) in printed {
            let contents = contents
                .as_str()
                .ok_or_else(|| format!("{}'s {path} is not a string", client.package))?;
            files.push((path, contents.to_owned()));
        }
    }
    Ok(files)
}

/// Writes the generated files the web host and scripts read.
pub fn write_generated(root: &Path) -> Result<(), String> {
    for (path, contents) in generated(root)? {
        if let Some(directory) = Path::new(&path).parent() {
            fs::create_dir_all(root.join(directory))
                .map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
        }
        fs::write(root.join(&path), contents)
            .map_err(|error| format!("cannot write {path}: {error}"))?;
        println!("Generated {path}.");
    }
    Ok(())
}

pub fn check_generated(root: &Path) -> Result<(), String> {
    for (path, contents) in generated(root)? {
        let saved = fs::read_to_string(root.join(&path))
            .map_err(|error| format!("cannot read {path}: {error}; run `cargo xtask schema`"))?;
        if saved.replace("\r\n", "\n") != contents {
            return Err(format!("{path} is out of date; run `cargo xtask schema`"));
        }
        println!("Generated file matches {path}.");
    }
    Ok(())
}
