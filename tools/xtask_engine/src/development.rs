use std::path::Path;
use std::process::Command;

use engine_core::{Error, ErrorCode, Expected};
use engine_dev::authoring;
use serde_json::{Value, json};

pub fn scene(arguments: &[String]) -> Result<(), String> {
    let result = scene_command(arguments);
    let succeeded = result.is_ok();
    let envelope = match result {
        Ok(result) => json!({"ok":true,"result":result}),
        Err(error) => json!({"ok":false,"error":error}),
    };
    println!("{envelope}");
    if succeeded {
        Ok(())
    } else {
        Err("scene command failed; see its JSON result".into())
    }
}

fn scene_command(arguments: &[String]) -> Result<Value, Error> {
    match arguments {
        [command, file, path] if command == "validate" && file == "--file" => {
            Ok(json!(authoring::validate_scene(
                Path::new(path),
                &crate::bricks::standard_kinds()?
            )?))
        }
        [command, file, path] if command == "format" && file == "--file" => format_scene(path),
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
                Error::new(ErrorCode::InvalidRevision, "$.expected-revision")
                    .expected(Expected::Integer)
                    .got(revision)
            })?;
            let source = authoring::read_scene_source(Path::new(candidate_path))?;
            Ok(json!(authoring::save_scene(
                Path::new(path),
                &source,
                revision,
                &crate::bricks::standard_kinds()?,
                alive
            )?))
        }
        _ => Err(Error::new(ErrorCode::InvalidArguments, "$")
            .got(arguments)
            .variants(["validate", "format", "save"])),
    }
}

fn format_scene(path: &str) -> Result<Value, Error> {
    let engine = crate::bricks::standard_engine()?;
    let schema: Value = serde_json::from_str(&crate::bricks::scene_schema(&engine)?)
        .map_err(|_| Error::new(ErrorCode::InvalidJson, "$.schema"))?;
    authoring::validate_scene(Path::new(path), &engine_scene::SceneKinds::of(engine.app.world()))?;
    let source = authoring::read_scene_source(Path::new(path))?;
    let document = engine_scene::SceneDocument::from_json(&source)?;
    let canonical = document.to_canonical_json(&schema)?;
    let changed = canonical != source;
    if changed {
        std::fs::write(path, &canonical).map_err(|_| {
            Error::new(ErrorCode::SourceIo, path)
                .expected(Expected::Writable)
                .got(path)
        })?;
    }
    Ok(json!({
        "scene": document.id,
        "changed": changed,
        "bytes": canonical.len(),
    }))
}

fn alive(pid: u64) -> bool {
    let listed = if cfg!(windows) {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
            .output()
    } else {
        Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "pid="])
            .output()
    };
    match listed {
        Ok(output) if cfg!(windows) => {
            !output.status.success()
                || String::from_utf8_lossy(&output.stdout).contains(&format!("\"{pid}\""))
        }
        Ok(output) => {
            output.status.success()
                && String::from_utf8_lossy(&output.stdout).trim() == pid.to_string()
        }
        Err(_) => true,
    }
}
