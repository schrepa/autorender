use std::{path::Path, process::Command};

use engine_core::{Error, ErrorCode, Expected};
use engine_dev::{CheckRequest, PROTOCOL_VERSION, authoring};
use serde_json::{Value, json};

use crate::Failure;

const MAX_LOCATIONS: usize = 64;

pub fn agent(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    match arguments.first().map(String::as_str) {
        Some("help") => return Ok(agent_help(root, &arguments[1..])?),
        Some("check") => return check(root, &arguments[1..]),
        Some("iterate") => crate::pipeline::install_web_dependencies(root)?,
        Some(_) | None => {}
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

pub fn verbs() -> Result<Value, String> {
    let engine = crate::bricks::standard_engine().map_err(|error| error.to_string())?;
    let verbs = engine_dev::schemas::verbs(engine.app.world())?;
    Ok(Value::Array(
        verbs
            .into_iter()
            .map(|(verb, request)| json!({ "verb": verb, "request": request }))
            .collect(),
    ))
}

fn agent_help(root: &Path, arguments: &[String]) -> Result<(), String> {
    if let Some(argument) = arguments.first() {
        return Err(format!(
            "agent help takes no arguments; received {argument:?}"
        ));
    }
    let output = Command::new("node")
        .args(["web/scripts/agent.mjs", "help"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot start the agent client: {error}"))?;
    let mut help: Value = String::from_utf8_lossy(&output.stdout)
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str(line).ok())
        .unwrap_or_else(|| json!({ "version": PROTOCOL_VERSION, "ok": true }));
    help["verbs"] = verbs()?;
    println!("{help}");
    Ok(())
}

fn check_request(arguments: &[String]) -> Result<CheckRequest, Error> {
    let source = match arguments {
        [] => "{}".to_owned(),
        [option, value] if option == "--json" => value.clone(),
        [option, value] if option == "--file" => {
            std::fs::read_to_string(value).map_err(|error| {
                Error::new(ErrorCode::SourceIo, "$.file").got(format!("{:?}", error.kind()))
            })?
        }
        _ => {
            return Err(Error::new(ErrorCode::InvalidArguments, "$")
                .got(arguments)
                .variants(["--json", "--file"]));
        }
    };
    engine_scene::parse_json(&source)
}

fn check(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let request = match check_request(arguments) {
        Ok(request) => request,
        Err(error) => {
            println!(
                "{}",
                json!({ "version": PROTOCOL_VERSION, "ok": false, "error": error })
            );
            return Err("agent check refused its request; see its JSON result".into());
        }
    };
    let mut verify = Command::new(std::env::current_exe().map_err(|error| error.to_string())?);
    verify.arg("verify").arg("--json").current_dir(root);
    if let Some(level) = request.level {
        verify.args(["--level", level.name()]);
    }
    if let Some(base) = &request.base {
        verify.args(["--base", base]);
    }
    let output = verify
        .output()
        .map_err(|error| format!("cannot run cargo xtask verify: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let ran = stdout
        .lines()
        .rev()
        .find_map(|line| {
            serde_json::from_str::<Value>(line)
                .ok()
                .filter(|value| value.get("verify").is_some())
        })
        .map(|value| value["verify"].clone());
    let exit = output.status.code();
    let steps: Vec<Value> = ran
        .as_ref()
        .and_then(|ran| ran["steps"].as_array())
        .into_iter()
        .flatten()
        .map(|step| {
            let mut step = step.clone();
            if step["ok"] != true {
                let failure = match &step["failure"] {
                    Value::String(failure) => failure.clone(),
                    failure => failure.to_string(),
                };
                step["locations"] = json!(locations(&format!("{failure}\n{stderr}")));
            }
            step
        })
        .collect();
    let passed = output.status.success();
    let reply = match &ran {
        Some(ran) => json!({
            "version": PROTOCOL_VERSION,
            "ok": true,
            "result": {
                "passed": passed,
                "exit": exit,
                "level": ran["level"],
                "base": ran["base"],
                "failed": steps.iter().filter(|step| step["ok"] != true).count(),
                "steps": steps,
            },
        }),
        None => json!({
            "version": PROTOCOL_VERSION,
            "ok": false,
            "error": Error::new(ErrorCode::InvalidArguments, "$.verify")
                .expected(Expected::Present)
                .got(json!({ "exit": exit, "locations": locations(&stderr) })),
        }),
    };
    println!("{reply}");
    match (ran.is_some(), passed) {
        (true, true) => Ok(()),
        (true, false) => Err(Failure::found(
            "agent check found faults; see its JSON result",
        )),
        (false, _) => Err("agent check could not run verify; see its JSON result".into()),
    }
}

fn logged_text(text: &str) -> String {
    fn strings(value: &Value, out: &mut String) {
        match value {
            Value::String(text) => {
                out.push_str(text);
                out.push('\n');
            }
            Value::Array(items) => items.iter().for_each(|item| strings(item, out)),
            Value::Object(fields) => fields.values().for_each(|field| strings(field, out)),
            Value::Null | Value::Bool(_) | Value::Number(_) => {}
        }
    }
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        match serde_json::from_str::<Value>(line) {
            Ok(record) => strings(&record, &mut out),
            Err(_) => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}

fn locations(text: &str) -> Vec<String> {
    const EXTENSIONS: [&str; 6] = [".rs:", ".mjs:", ".ts:", ".wgsl:", ".json:", ".toml:"];
    let mut found: Vec<String> = Vec::new();
    let separators =
        |character: char| character.is_whitespace() || "()[]{}'\"`,;<>".contains(character);
    let text = logged_text(text);
    for token in text.split(separators) {
        for extension in EXTENSIONS {
            let Some(at) = token.find(extension) else {
                continue;
            };
            let after = at + extension.len();
            let line: String = token[after..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            if line.is_empty() {
                continue;
            }
            let path = token[..after - 1].replace('\\', "/");
            let location = format!("{path}:{line}");
            if !found.contains(&location) && found.len() < MAX_LOCATIONS {
                found.push(location);
            }
        }
    }
    found
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

fn scene_command(arguments: &[String]) -> Result<Value, Error> {
    match arguments {
        [command, file, path] if command == "validate" && file == "--file" => {
            let source = authoring::read_scene_source(Path::new(path))?;
            Ok(json!(authoring::validate_scene(
                &source,
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
                &crate::bricks::standard_kinds()?
            )?))
        }
        [command] if command == "schema" => {
            let engine = crate::bricks::standard_engine()?;
            Ok(json!({"schema": crate::bricks::scene_schema(&engine)?}))
        }
        _ => Err(Error::new(ErrorCode::InvalidArguments, "$")
            .got(arguments)
            .variants(["validate", "format", "save", "schema"])),
    }
}

fn format_scene(path: &str) -> Result<Value, Error> {
    let engine = crate::bricks::standard_engine()?;
    let schema: Value = serde_json::from_str(&crate::bricks::scene_schema(&engine)?)
        .map_err(|_| Error::new(ErrorCode::InvalidJson, "$.schema"))?;
    let source = authoring::read_scene_source(Path::new(path))?;
    let document = engine_scene::SceneDocument::from_json(&source)?;
    document.validate(&engine_scene::SceneKinds::of(engine.app.world()))?;
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
