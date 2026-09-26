use std::{path::Path, process::Command};

use engine_dev::{DevError, authoring};
use serde_json::{Value, json};

/// Keep stdout as one JSON result per command; subprocess diagnostics use stderr.
pub fn agent(root: &Path, arguments: &[String]) -> Result<(), String> {
    if arguments
        .first()
        .is_some_and(|argument| argument == "watch")
    {
        crate::pipeline::install_web_dependencies(root)?;
    }
    let status = Command::new("node")
        .arg("web/scripts/agent.mjs")
        .args(arguments)
        .env(
            "ENGINE_XTASK_BIN",
            std::env::current_exe().map_err(|error| error.to_string())?,
        )
        .current_dir(root)
        .status()
        .map_err(|error| format!("cannot start agent bridge: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err("agent command failed; see its JSON result".into())
    }
}

pub fn scene(arguments: &[String]) -> Result<(), String> {
    let result = scene_command(arguments);
    let succeeded = result.is_ok();
    let envelope = match result {
        Ok(result) => json!({"version":1,"ok":true,"result":result}),
        Err(error) => json!({"version":1,"ok":false,"error":error}),
    };
    println!("{envelope}");
    if succeeded {
        Ok(())
    } else {
        Err("scene command failed; see its JSON result".into())
    }
}

fn scene_command(arguments: &[String]) -> Result<Value, DevError> {
    match arguments {
        [command, file, path] if command == "validate" && file == "--file" => {
            let source = authoring::read_scene_source(Path::new(path))?;
            Ok(json!(authoring::validate_scene(&source)?))
        }
        [
            command,
            file,
            path,
            candidate,
            candidate_path,
            expected,
            revision,
        ] if command == "save"
            && file == "--file"
            && candidate == "--candidate"
            && expected == "--expected-revision" =>
        {
            let revision = revision.parse::<u64>().map_err(|_| {
                DevError::new(
                    "invalid_revision",
                    "--expected-revision",
                    "expected an unsigned 64-bit decimal integer",
                )
            })?;
            let source = authoring::read_scene_source(Path::new(candidate_path))?;
            Ok(json!(authoring::save_scene(
                Path::new(path),
                &source,
                revision
            )?))
        }
        [command] if command == "schema" => {
            Ok(json!({"schema":engine_scene::SceneDocument::schema_json()}))
        }
        _ => Err(DevError::new(
            "invalid_arguments",
            "$",
            "usage: cargo xtask scene validate --file PATH | save --file TARGET --candidate PATH --expected-revision N | schema",
        )),
    }
}
