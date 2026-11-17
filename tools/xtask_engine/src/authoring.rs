use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use engine_dev::authoring::{read_scene_source, validate_scene};

pub fn validate_sources(root: &Path) -> Result<(), String> {
    let paths = project_scenes(root)?;
    let kinds = crate::bricks::standard_kinds().map_err(|error| error.to_string())?;
    for path in &paths {
        let display = path.strip_prefix(root).unwrap_or(path).display();
        let report = read_scene_source(path)
            .and_then(|source| validate_scene(&source, &kinds))
            .map_err(|error| format!("{display}: {error}"))?;
        let record = serde_json::json!({
            "event": "scene_validated",
            "file": display.to_string(),
            "scene": report.id,
            "revision": report.revision,
            "authored_entities": report.entities,
            "prefabs": report.prefabs,
            "expanded_entities": report.expanded_entities,
        });
        println!("{record}");
    }
    println!(
        "{}",
        serde_json::json!({"event": "scenes_validated", "scenes": paths.len()})
    );
    Ok(())
}

fn project_scenes(root: &Path) -> Result<Vec<PathBuf>, String> {
    Ok(crate::rules::repository(root)?
        .own()
        .flat_map(|package| {
            package.engine["scenes"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(serde_json::Value::as_str)
                .map(move |scene| package.folder.join(scene))
        })
        .collect())
}

fn generated(root: &Path) -> Result<Vec<(String, String)>, String> {
    let clients = serde_json::to_string_pretty(&crate::process::clients_json())
        .expect("client_list_serializes")
        + "
";
    let repository = crate::rules::repository(root)?;
    let bricks = engine_checks::bricks::bricks_json(
        repository,
        crate::bricks::standard,
        &mut crate::bricks::standard_engine().map_err(|error| error.to_string())?,
    )?;
    tooling_types(root)?;
    let protocol = serde_json::to_string_pretty(&engine_dev::schemas::summary())
        .map_err(|error| format!("cannot write JSON: {error}"))?;
    let mut files = game_schemas(root)?;
    files.extend([
        ("web/src/generated/clients.json".to_owned(), clients),
        (
            "web/src/generated/protocol.json".to_owned(),
            protocol + "\n",
        ),
        (
            "web/src/generated/stats.ts".to_owned(),
            engine::tooling::declarations(&engine::stats::frame_stats_schema(), "FrameStats")
                .trim_start()
                .to_owned(),
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
                &engine_dev::schemas::reply()?,
                "Reply",
                &[("ErrorCode", "./errors")],
            ),
        ),
        ("web/src/generated/errors.ts".to_owned(), error_codes()),
        (engine_checks::bricks::FILE.to_owned(), bricks),
    ]);
    Ok(files)
}

fn game_schemas(root: &Path) -> Result<Vec<(String, String)>, String> {
    let mut files = Vec::new();
    for package in crate::rules::repository(root)?.projects() {
        let Some(folder) = package.path.as_deref() else {
            continue;
        };
        let project = crate::process::Project::read(root, folder)?;
        let program = crate::pipeline::build_native(root, &project)?;
        let output = Command::new(&program)
            .arg("--schemas")
            .current_dir(root)
            .output()
            .map_err(|error| format!("{program}: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "{program} --schemas: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let printed = String::from_utf8_lossy(&output.stdout);
        let listed: serde_json::Map<String, serde_json::Value> = printed
            .lines()
            .rev()
            .find_map(|line| serde_json::from_str(line).ok())
            .ok_or_else(|| format!("{program} --schemas printed no files"))?;
        for (name, contents) in listed {
            let contents = contents
                .as_str()
                .ok_or_else(|| format!("{program} --schemas printed {name} as no text"))?;
            files.push((format!("{folder}/generated/{name}"), contents.to_owned()));
        }
        files.push((
            format!("{folder}/generated/corpora.json"),
            corpora(root, &program)?,
        ));
    }
    Ok(files)
}

fn corpora(root: &Path, program: &str) -> Result<String, String> {
    let output = Command::new(program)
        .args(["--pure", engine::smokes::LIST])
        .current_dir(root)
        .output()
        .map_err(|error| format!("{program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} --pure list: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let listed = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|record| record["event"] == "corpora")
        .ok_or_else(|| format!("{program} --pure list printed no corpora"))?;
    Ok(serde_json::to_string_pretty(&listed["corpora"]).map_err(|error| error.to_string())? + "\n")
}

fn error_codes() -> String {
    let codes: String = engine::ErrorCode::NAMES
        .iter()
        .map(|code| format!("\n  | \"{code}\""))
        .collect();
    let name = engine::JS_ERROR_NAME;
    format!(
        "import type {{ Error as Failure }} from \"./reply\";\n\n\
         export type ErrorCode ={codes};\n\n\
         export type EngineError = Error & Failure;\n\n\
         export function engineError(thrown: unknown): EngineError | undefined {{\n  \
         return thrown instanceof Error && thrown.name === \"{name}\" ? thrown as EngineError : undefined;\n\
         }}\n\n\
         export function engineFailure(failure: Failure): EngineError {{\n  \
         return Object.assign(new Error(JSON.stringify(failure)), failure, {{ name: \"{name}\" }});\n\
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

pub fn generate(root: &Path, options: &[String]) -> Result<(), String> {
    match options {
        [] => engine_checks::generated::write(root, &generated(root)?),
        [check] if check == "--check" => engine_checks::generated::check(root, &generated(root)?),
        _ => Err("usage: generate [--check]".to_owned()),
    }
}
