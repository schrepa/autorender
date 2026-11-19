use std::{
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
    let mut files = game_schemas(root)?;
    files.extend([
        ("web/src/generated/clients.json".to_owned(), clients),
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
                &engine_dev::Reply::schema(),
                "Reply",
                &[("ErrorCode", "./errors")],
            ),
        ),
        ("web/src/generated/errors.ts".to_owned(), error_codes()),
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
    }
    Ok(files)
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

pub fn generate(root: &Path, options: &[String]) -> Result<(), String> {
    match options {
        [] => engine_checks::generated::write(root, &generated(root)?),
        [check] if check == "--check" => engine_checks::generated::check(root, &generated(root)?),
        _ => Err("usage: generate [--check]".to_owned()),
    }
}
