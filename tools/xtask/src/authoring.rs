use std::{
    fs,
    path::{Path, PathBuf},
};

use engine_dev::authoring::{read_scene_source, validate_scene};

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

fn generated(root: &Path) -> Result<Vec<(String, String)>, String> {
    let clients = serde_json::to_string_pretty(&crate::process::clients_json())
        .expect("the client list serializes")
        + "
";
    let engine = crate::bricks::standard_engine();
    let world = engine.app.world();
    let development = engine_dev::schemas::generated(world)?;
    tooling_types(root)?;
    let stats = &development.frame_stats;
    let mut files: Vec<(String, String)> = development
        .files
        .iter()
        .map(|(path, contents)| ((*path).to_owned(), contents.clone()))
        .collect();
    files.extend([
        ("web/src/generated/clients.json".to_owned(), clients),
        (
            "web/src/generated/stats.ts".to_owned(),
            engine::tooling::interfaces(stats, "FrameStats", "engine_core::FrameStats"),
        ),
        (
            "web/src/generated/runtime.ts".to_owned(),
            engine::runtime_typescript(),
        ),
        (
            "web/src/generated/host.ts".to_owned(),
            engine::host::typescript(),
        ),
        (
            "web/src/generated/reply.ts".to_owned(),
            engine::tooling::message_interfaces(
                &engine_dev::schemas::reply(world)?,
                "Reply",
                "engine_dev::Reply",
                &[("ErrorCode", "./errors")],
            ),
        ),
        ("web/src/generated/errors.ts".to_owned(), error_codes()),
        (
            "web/src/generated/coverage.json".to_owned(),
            crate::coverage::coverage(root, &development)?,
        ),
        (
            crate::bricks::BRICKS_FILE.to_owned(),
            crate::bricks::bricks_json(root)?,
        ),
    ]);
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
            .arg("--target-dir")
            .arg(crate::process::tools_target(root))
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

fn error_codes() -> String {
    let codes: String = engine::ErrorCode::ALL
        .iter()
        .map(|(_, code)| format!("\n  | \"{code}\""))
        .collect();
    let name = engine::JS_ERROR_NAME;
    format!(
        "// Generated by `cargo xtask schema` from Rust's `engine::ErrorCode`; do not edit.\n\n\
         /** An error's code, as replies name it; protocol.json's `errors` say what each \
         means. */\nexport type ErrorCode ={codes};\n\n\
         /** An error the engine threw on the page: its code and where (a JSON path, or\n \
         * `?NAME` for a page parameter), its message saying both and what is wrong. */\n\
         export interface EngineError extends Error {{\n  code: ErrorCode;\n  path: string;\n}}\n\n\
         /** `thrown` as an error the engine threw, where it is one. */\n\
         export function engineError(thrown: unknown): EngineError | undefined {{\n  \
         return thrown instanceof Error && thrown.name === \"{name}\" ? thrown as EngineError : undefined;\n\
         }}\n"
    )
}

fn tooling_types(root: &Path) -> Result<(), String> {
    const MARK: &str = "kind: \"tooling\"";
    let folder = root.join("web/scripts");
    let mut problems = Vec::new();
    for entry in
        fs::read_dir(&folder).map_err(|error| format!("cannot read web/scripts: {error}"))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.extension().is_none_or(|extension| extension != "mjs") {
            continue;
        }
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        for (at, _) in text.match_indices(MARK) {
            let named = text[..at]
                .rsplit_once("name: \"")
                .and_then(|(_, rest)| rest.split_once('"'))
                .map(|(name, _)| name);
            let script = path.file_name().unwrap_or_default().to_string_lossy();
            match named {
                Some(name) if engine_dev::TOOLING_TYPES.contains(&name) => {}
                Some(name) => problems.push(format!(
                    "web/scripts/{script} adds {name} to describe's types, which \
                     engine_dev::TOOLING_TYPES does not reserve: list it there"
                )),
                None => problems.push(format!(
                    "web/scripts/{script} adds a tooling type to describe without a name"
                )),
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

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
    let mut stale = Vec::new();
    for (path, contents) in generated(root)? {
        match fs::read_to_string(root.join(&path)) {
            Ok(saved) if saved.replace("\r\n", "\n") == contents => {
                println!("Generated file matches {path}.");
            }
            Ok(_) => stale.push(format!("{path} is out of date")),
            Err(error) => stale.push(format!("cannot read {path}: {error}")),
        }
    }
    if stale.is_empty() {
        return Ok(());
    }
    Err(format!("{}; run `cargo xtask schema`", stale.join("; ")))
}
