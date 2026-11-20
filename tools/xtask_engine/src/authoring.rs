use std::{path::Path, process::Command};

fn generated(root: &Path) -> Result<Vec<(String, String)>, String> {
    let clients = serde_json::to_string_pretty(&crate::process::clients_json())
        .expect("client_list_serializes")
        + "
";
    let mut files = game_files(root)?;
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

fn game_files(root: &Path) -> Result<Vec<(String, String)>, String> {
    let mut files = Vec::new();
    for package in crate::rules::repository(root)?.projects() {
        let Some(folder) = package.path.as_deref() else {
            continue;
        };
        let project = crate::process::Project::read(root, folder)?;
        let program = crate::pipeline::build_native(root, &project)?;
        for mode in ["--schemas", "--shaders"] {
            let output = Command::new(&program)
                .arg(mode)
                .current_dir(root)
                .output()
                .map_err(|error| format!("{program}: {error}"))?;
            if !output.status.success() {
                return Err(format!(
                    "{program} {mode}: {}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
            let printed = String::from_utf8_lossy(&output.stdout);
            let listed: serde_json::Map<String, serde_json::Value> = printed
                .lines()
                .rev()
                .find_map(|line| serde_json::from_str(line).ok())
                .ok_or_else(|| format!("{program} {mode} printed no files"))?;
            for (name, contents) in listed {
                let contents = contents
                    .as_str()
                    .ok_or_else(|| format!("{program} {mode} printed {name} as no text"))?;
                files.push((format!("{folder}/generated/{name}"), contents.to_owned()));
            }
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
