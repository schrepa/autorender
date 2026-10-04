//! A captured, generation-scoped build for the development supervisor.
//!
//! This module never publishes bindings. The supervisor selects a generation only
//! after a successful build and browser startup, retaining the previous session
//! when compilation fails. All child output is drained with bounded storage so
//! stdout remains one machine-readable result, even for very large diagnostics.

use std::{
    collections::BTreeMap,
    fs,
    io::{self, BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    thread,
    time::Instant,
};

use serde_json::{Value, json};

use crate::process::{
    CLIENTS, WASM_TARGET, check_wasm_bindgen, client_crates, client_dev_features, client_packages,
};

const MAX_LINE_BYTES: usize = 256 * 1024;
const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_DIAGNOSTICS: usize = 100;
const MAX_DIAGNOSTIC_BYTES: usize = 256 * 1024;
const MAX_SPANS: usize = 8;
const MAX_CHILDREN: usize = 8;

/// Compile all development clients and generate bindings into a new directory.
/// `output` must be an unused direct child of `root/web/pkg-iterations`.
/// The returned value is bounded and is the only output the caller should print.
pub fn build(root: &Path, output: &Path) -> Value {
    let started = Instant::now();
    let mut report = json!({
        "ok": false,
        "build": {
            "target": WASM_TARGET,
            "profile": "release",
            "development": true,
            "output": output.to_string_lossy(),
            "packages": [],
            "stage": "prepare"
        },
        "diagnostics": [],
        "diagnostics_truncated": 0,
        "diagnostic_fields_truncated": 0,
        "oversized_messages": 0,
        "output_bytes_truncated": 0
    });
    if let Err((code, path, message)) = execute(root, output, &mut report) {
        report["error"] = json!({"code": code, "path": path, "message": message});
    } else {
        report["ok"] = json!(true);
        report["build"]["stage"] = json!("complete");
    }
    report["build"]["elapsed_ms"] = json!(started.elapsed().as_millis() as u64);
    report
}

type BuildError = (&'static str, String, String);

fn execute(root: &Path, output: &Path, report: &mut Value) -> Result<(), BuildError> {
    let (root, output) = prepare_output(root, output)
        .map_err(|message| ("invalid_output_path", "output".into(), message))?;
    report["build"]["output"] = json!(output);

    report["build"]["stage"] = json!("toolchain");
    check_wasm_bindgen(&root).map_err(|(code, message)| (code, "wasm-bindgen".into(), message))?;

    report["build"]["stage"] = json!("cargo");
    let target_dir = root.join("target/iteration-wasm");
    let features = client_dev_features();
    let compiled = capture(
        Command::new("cargo")
            .current_dir(&root)
            .arg("build")
            .args(client_packages())
            .args([
                "--features",
                &features,
                "--target",
                WASM_TARGET,
                "--release",
                "--locked",
                "--message-format=json",
                "--target-dir",
            ])
            .arg(&target_dir),
        true,
    )
    .map_err(|message| ("cargo_unavailable", "cargo".into(), message))?;
    admit_compilation(report, &compiled)?;

    // Validate all compiler artifacts before admitting any generated bindings.
    // The dedicated target directory keeps ordinary dev/lint builds from
    // replacing a compiled artifact while wasm-bindgen reads it.
    let mut artifacts = Vec::with_capacity(CLIENTS.len());
    let artifact_root = target_dir.join(WASM_TARGET).join("release");
    for package in client_crates() {
        let expected = artifact_root.join(format!("{package}.wasm"));
        let artifact = compiled.stdout.artifacts.get(package).ok_or_else(|| {
            (
                "missing_artifact",
                package.into(),
                format!("Cargo did not report {package}.wasm"),
            )
        })?;
        let actual = artifact.canonicalize().map_err(|error| {
            (
                "missing_artifact",
                package.into(),
                format!("cannot read compiled artifact: {error}"),
            )
        })?;
        if !actual.is_file() || expected.canonicalize().ok().as_ref() != Some(&actual) {
            return Err((
                "invalid_artifact",
                package.into(),
                "Cargo artifact is outside the dedicated build output".into(),
            ));
        }
        artifacts.push((package, actual));
    }

    // A failed compilation cannot create or mutate the generation directory.
    // create_dir is exclusive: even a racing second build cannot reuse it.
    fs::create_dir(&output).map_err(|error| {
        (
            "output_io",
            "output".into(),
            format!("cannot create fresh generation: {error}"),
        )
    })?;
    report["build"]["stage"] = json!("bindings");
    let mut packages = Vec::with_capacity(CLIENTS.len());
    for (package, artifact) in artifacts {
        let generated = capture(
            Command::new("wasm-bindgen")
                .current_dir(&root)
                .arg(artifact)
                .args(["--target", "web", "--out-dir"])
                .arg(&output)
                .args(["--out-name", package]),
            false,
        )
        .map_err(|message| ("bindgen_unavailable", package.into(), message))?;
        let discarded = report["output_bytes_truncated"]
            .as_u64()
            .unwrap_or_default()
            + generated.stdout.text_truncated
            + generated.stderr_truncated;
        report["output_bytes_truncated"] = json!(discarded);
        if !generated.status.success() {
            return Err((
                "bindgen_failed",
                package.into(),
                failure_message("Binding generation failed", &generated),
            ));
        }
        let javascript = output.join(format!("{package}.js"));
        let wasm = output.join(format!("{package}_bg.wasm"));
        let declarations = output.join(format!("{package}.d.ts"));
        if [&javascript, &wasm, &declarations]
            .iter()
            .any(|path| !path.is_file())
        {
            return Err((
                "missing_bindings",
                package.into(),
                "wasm-bindgen succeeded without producing all required files".into(),
            ));
        }
        packages.push(json!({"name": package, "javascript": javascript, "wasm": wasm, "declarations": declarations}));
    }
    report["build"]["packages"] = json!(packages);
    Ok(())
}

fn prepare_output(root: &Path, output: &Path) -> Result<(PathBuf, PathBuf), String> {
    let root = root
        .canonicalize()
        .map_err(|error| format!("cannot resolve workspace: {error}"))?;
    let web = root.join("web");
    if web.canonicalize().ok().as_ref() != Some(&web) {
        return Err("web must be a real directory directly within the workspace".into());
    }
    let requested = if output.is_absolute() {
        output.to_path_buf()
    } else {
        root.join(output)
    };
    let generation = requested
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| {
            !name.is_empty()
                && name.len() <= 100
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
        })
        .ok_or_else(|| {
            "generation must contain 1–100 ASCII letters, digits, underscores or hyphens".to_owned()
        })?;
    let parent = web.join("pkg-iterations");
    // Validate the requested parent before creating anything. Resolving the web
    // directory also handles Windows' canonical extended-length path prefix.
    let requested_parent = requested
        .parent()
        .ok_or_else(|| "missing output parent".to_owned())?;
    let requested_web = requested_parent
        .parent()
        .and_then(|path| path.canonicalize().ok());
    if requested_parent.file_name() != Some(std::ffi::OsStr::new("pkg-iterations"))
        || requested_web.as_ref() != Some(&web)
    {
        return Err("output must be a direct child of web/pkg-iterations".into());
    }
    match fs::create_dir(&parent) {
        Ok(()) => (),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(format!("cannot create generation parent: {error}")),
    }
    if parent.canonicalize().ok().as_ref() != Some(&parent) {
        return Err("generation parent must remain inside the workspace".into());
    }
    let output = parent.join(generation);
    if fs::symlink_metadata(&output).is_ok() {
        return Err("generation directory already exists; choose a fresh generation".into());
    }
    Ok((root, output))
}

struct Captured {
    status: ExitStatus,
    stdout: ParsedOutput,
    stderr: String,
    stderr_truncated: u64,
}

fn admit_compilation(report: &mut Value, compiled: &Captured) -> Result<(), BuildError> {
    report["build"]["cargo_exit_code"] = json!(compiled.status.code());
    report["diagnostics"] = json!(compiled.stdout.diagnostics);
    report["diagnostics_truncated"] = json!(compiled.stdout.diagnostics_truncated);
    report["diagnostic_fields_truncated"] = json!(compiled.stdout.fields_truncated);
    report["oversized_messages"] = json!(compiled.stdout.oversized_messages);
    report["output_bytes_truncated"] =
        json!(compiled.stdout.text_truncated + compiled.stderr_truncated);
    if compiled.status.success() {
        Ok(())
    } else {
        Err((
            "cargo_failed",
            "cargo".into(),
            failure_message("Rust compilation failed", compiled),
        ))
    }
}

fn capture(command: &mut Command, cargo: bool) -> Result<Captured, String> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot start build tool: {error}"))?;
    let stdout = child.stdout.take().ok_or("missing child stdout")?;
    let stderr = child.stderr.take().ok_or("missing child stderr")?;
    thread::scope(|scope| {
        let stdout = scope.spawn(|| parse_stream(stdout, cargo));
        let stderr = scope.spawn(|| capture_text(stderr));
        let status = child
            .wait()
            .map_err(|error| format!("cannot wait for build tool: {error}"))?;
        let stdout = stdout
            .join()
            .map_err(|_| "build stdout reader failed")?
            .map_err(|error| format!("cannot read build stdout: {error}"))?;
        let (stderr, stderr_truncated) =
            stderr
                .join()
                .map_err(|_| "build stderr reader failed")?
                .map_err(|error| format!("cannot read build stderr: {error}"))?;
        Ok(Captured {
            status,
            stdout,
            stderr,
            stderr_truncated,
        })
    })
}

fn failure_message(prefix: &str, captured: &Captured) -> String {
    let detail = if captured.stderr.trim().is_empty() {
        &captured.stdout.text
    } else {
        &captured.stderr
    };
    let mut ignored = 0;
    format!(
        "{prefix} ({}): {}",
        captured.status,
        bounded(detail.trim(), 4096, &mut ignored)
    )
}

fn capture_text(mut input: impl Read) -> io::Result<(String, u64)> {
    let mut bytes = Vec::with_capacity(MAX_TEXT_BYTES);
    let mut buffer = [0_u8; 8192];
    let mut discarded = 0;
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        let admitted = count.min(MAX_TEXT_BYTES - bytes.len());
        bytes.extend_from_slice(&buffer[..admitted]);
        discarded += (count - admitted) as u64;
    }
    Ok((String::from_utf8_lossy(&bytes).into_owned(), discarded))
}

#[derive(Default)]
struct ParsedOutput {
    diagnostics: Vec<Value>,
    diagnostic_bytes: usize,
    diagnostics_truncated: u64,
    fields_truncated: u64,
    oversized_messages: u64,
    artifacts: BTreeMap<String, PathBuf>,
    text: String,
    text_truncated: u64,
}

fn parse_stream(input: impl Read, cargo: bool) -> io::Result<ParsedOutput> {
    if !cargo {
        let (text, text_truncated) = capture_text(input)?;
        return Ok(ParsedOutput {
            text,
            text_truncated,
            ..Default::default()
        });
    }
    let mut input = BufReader::new(input);
    let mut output = ParsedOutput::default();
    let mut line = Vec::new();
    let mut oversized = false;
    loop {
        let buffer = input.fill_buf()?;
        if buffer.is_empty() {
            if !line.is_empty() || oversized {
                output.line(&line, oversized);
            }
            break;
        }
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let count = newline.map_or(buffer.len(), |index| index + 1);
        let admitted = count.min(MAX_LINE_BYTES - line.len());
        line.extend_from_slice(&buffer[..admitted]);
        oversized |= admitted != count;
        input.consume(count);
        if newline.is_some() {
            output.line(&line, oversized);
            line.clear();
            oversized = false;
        }
    }
    Ok(output)
}

impl ParsedOutput {
    fn line(&mut self, line: &[u8], oversized: bool) {
        if oversized {
            self.oversized_messages += 1;
            return;
        }
        let Ok(value) = serde_json::from_slice::<Value>(line) else {
            let text = String::from_utf8_lossy(line);
            let mut ignored = 0;
            let admitted = bounded(&text, MAX_TEXT_BYTES - self.text.len(), &mut ignored);
            self.text_truncated += (text.len() - admitted.len()) as u64;
            self.text.push_str(&admitted);
            return;
        };
        match value["reason"].as_str() {
            Some("compiler-message") => self.diagnostic(&value),
            Some("compiler-artifact") => {
                if let Some(name) = value["target"]["name"]
                    .as_str()
                    .filter(|name| client_crates().contains(name))
                    && let Some(files) = value["filenames"].as_array()
                    && let Some(wasm) = files
                        .iter()
                        .filter_map(Value::as_str)
                        .find(|path| path.ends_with(".wasm"))
                {
                    self.artifacts.insert(name.to_owned(), PathBuf::from(wasm));
                }
            }
            _ => (),
        }
    }

    fn diagnostic(&mut self, value: &Value) {
        if self.diagnostics.len() == MAX_DIAGNOSTICS {
            self.diagnostics_truncated += 1;
            return;
        }
        let message = &value["message"];
        let mut spans = Vec::new();
        if let Some(source) = message["spans"].as_array() {
            self.fields_truncated += source.len().saturating_sub(MAX_SPANS) as u64;
            for span in source.iter().take(MAX_SPANS) {
                spans.push(json!({
                    "file": self.string(&span["file_name"], 1024),
                    "line_start": span["line_start"].as_u64(),
                    "line_end": span["line_end"].as_u64(),
                    "column_start": span["column_start"].as_u64(),
                    "column_end": span["column_end"].as_u64(),
                    "is_primary": span["is_primary"].as_bool().unwrap_or(false),
                    "label": self.string(&span["label"], 512)
                }));
            }
        }
        let mut help = Vec::new();
        if let Some(children) = message["children"].as_array() {
            self.fields_truncated += children.len().saturating_sub(MAX_CHILDREN) as u64;
            for child in children.iter().take(MAX_CHILDREN) {
                help.push(json!({"level": self.string(&child["level"], 32), "message": self.string(&child["message"], 2048)}));
            }
        }
        let diagnostic = json!({
            "package": self.string(&value["package_id"], 512),
            "target": self.string(&value["target"]["name"], 128),
            "code": self.string(&message["code"]["code"], 64),
            "level": self.string(&message["level"], 32),
            "message": self.string(&message["message"], 4096),
            "spans": spans,
            "help": help
        });
        let size = diagnostic.to_string().len() + 1;
        if self.diagnostic_bytes + size > MAX_DIAGNOSTIC_BYTES {
            self.diagnostics_truncated += 1;
        } else {
            self.diagnostic_bytes += size;
            self.diagnostics.push(diagnostic);
        }
    }

    fn string(&mut self, value: &Value, limit: usize) -> Value {
        value.as_str().map_or(Value::Null, |text| {
            json!(bounded(text, limit, &mut self.fields_truncated))
        })
    }
}

fn bounded(text: &str, limit: usize, truncated: &mut u64) -> String {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    if end < text.len() {
        *truncated += 1;
    }
    text[..end].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn diagnostic() -> Value {
        json!({
            "reason": "compiler-message", "package_id": "path+file:///game#playground@0.1.0",
            "target": {"name": "playground", "kind": ["cdylib"]},
            "message": {
                "code": {"code": "E0308", "explanation": "long explanation omitted"},
                "level": "error", "message": "mismatched types",
                "spans": [{"file_name": "examples/playground/src/lib.rs", "line_start": 12,
                    "line_end": 12, "column_start": 4, "column_end": 8, "is_primary": true,
                    "label": "expected `f32`, found `&str`", "text": [{"text": "source excluded"}]}],
                "children": [{"level": "help", "message": "use a floating point value", "spans": []}],
                "rendered": "terminal formatting excluded"
            }
        })
    }

    #[test]
    fn cargo_messages_keep_actionable_locations_and_ignore_terminal_payloads() {
        let stream = format!(
            "{}\n{}\n{}\n",
            diagnostic(),
            json!({
                "reason": "compiler-artifact", "target": {"name": "playground"},
                "filenames": ["target/iteration-wasm/wasm32-unknown-unknown/release/playground.wasm"]
            }),
            json!({"reason": "build-finished", "success": false})
        );
        let parsed = parse_stream(stream.as_bytes(), true).unwrap();
        assert_eq!(parsed.diagnostics.len(), 1);
        let message = &parsed.diagnostics[0];
        assert_eq!(message["code"], "E0308");
        assert_eq!(message["spans"][0]["line_start"], 12);
        assert_eq!(message["spans"][0]["column_start"], 4);
        assert_eq!(message["help"][0]["message"], "use a floating point value");
        assert!(message.get("rendered").is_none());
        assert_eq!(parsed.artifacts.len(), 1);
        assert!(parsed.text.is_empty());
    }

    #[test]
    fn oversized_lines_are_drained_before_following_diagnostics() {
        let mut source = vec![b'x'; MAX_LINE_BYTES * 3];
        source.push(b'\n');
        source.extend(serde_json::to_vec(&diagnostic()).unwrap());
        let parsed = parse_stream(&source[..], true).unwrap();
        assert_eq!(parsed.oversized_messages, 1);
        assert_eq!(parsed.diagnostics.len(), 1);
        assert!(parsed.text.is_empty());
    }

    #[test]
    fn diagnostic_count_text_spans_and_help_are_bounded() {
        let mut message = diagnostic();
        message["message"]["message"] = json!("界".repeat(3000));
        message["message"]["spans"] = json!(vec![message["message"]["spans"][0].clone(); 20]);
        message["message"]["children"] = json!(vec![message["message"]["children"][0].clone(); 20]);
        let parsed = parse_stream(message.to_string().as_bytes(), true).unwrap();
        assert_eq!(parsed.diagnostics.len(), 1);
        assert_eq!(parsed.fields_truncated, 25);
        assert!(parsed.diagnostics[0]["message"].as_str().unwrap().len() <= 4096);
        assert_eq!(
            parsed.diagnostics[0]["spans"].as_array().unwrap().len(),
            MAX_SPANS
        );
        assert_eq!(
            parsed.diagnostics[0]["help"].as_array().unwrap().len(),
            MAX_CHILDREN
        );
        let line = format!("{}\n", diagnostic());
        let parsed = parse_stream(line.repeat(103).as_bytes(), true).unwrap();
        assert_eq!(parsed.diagnostics.len(), MAX_DIAGNOSTICS);
        assert_eq!(parsed.diagnostics_truncated, 3);
    }

    #[test]
    fn total_serialized_diagnostics_fit_the_bridge_budget() {
        let mut message = diagnostic();
        message["message"]["message"] = json!("m".repeat(4096));
        message["message"]["children"] = json!(vec![
            json!({"level": "help", "message": "h".repeat(2048)});
            8
        ]);
        let source = format!("{message}\n").repeat(100);
        let parsed = parse_stream(source.as_bytes(), true).unwrap();
        assert!(!parsed.diagnostics.is_empty());
        assert!(parsed.diagnostics_truncated > 0);
        assert!(serde_json::to_vec(&parsed.diagnostics).unwrap().len() <= MAX_DIAGNOSTIC_BYTES + 2);
    }

    #[test]
    fn captured_tool_output_is_bounded_and_drained() {
        let source = vec![b'a'; MAX_TEXT_BYTES * 3];
        let (text, discarded) = capture_text(&source[..]).unwrap();
        assert_eq!(text.len(), MAX_TEXT_BYTES);
        assert_eq!(discarded, (MAX_TEXT_BYTES * 2) as u64);
    }

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "engine-iteration-build-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            fs::create_dir(path.join("web")).unwrap();
            Self(path.canonicalize().unwrap())
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            // Only remove this uniquely created fixture, whose canonical parent
            // and prefix are verified before recursive cleanup.
            let temporary = std::env::temp_dir().canonicalize().unwrap();
            assert_eq!(self.0.parent(), Some(temporary.as_path()));
            assert!(
                self.0
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("engine-iteration-build-")
            );
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn failed_compilation_preserves_published_bindings_and_requires_fresh_output() {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt;

        let root = Fixture::new();
        fs::create_dir(root.0.join("web/pkg-dev")).unwrap();
        let live = root.0.join("web/pkg-dev/playground.js");
        fs::write(&live, "published generation").unwrap();
        let generation = root.0.join("web/pkg-iterations/next");
        let (_, prepared) = prepare_output(&root.0, &generation).unwrap();
        let compiled = Captured {
            status: ExitStatus::from_raw(256),
            stdout: parse_stream(diagnostic().to_string().as_bytes(), true).unwrap(),
            stderr: "error: could not compile playground due to one previous error".into(),
            stderr_truncated: 0,
        };
        let mut report = json!({"build": {}});
        let failure = admit_compilation(&mut report, &compiled).unwrap_err();
        assert_eq!(failure.0, "cargo_failed");
        assert_eq!(report["diagnostics"][0]["code"], "E0308");
        // The failed compilation cannot proceed to generating bindings.
        assert!(!prepared.exists());
        assert_eq!(fs::read_to_string(&live).unwrap(), "published generation");
        fs::create_dir(&prepared).unwrap();
        assert!(prepare_output(&root.0, &generation).is_err());
        assert!(prepare_output(&root.0, &root.0.join("web/pkg-dev")).is_err());
        assert!(prepare_output(&root.0, &root.0.join("web/pkg-iterations/../escape")).is_err());
        assert!(prepare_output(&root.0, &root.0.join("elsewhere/pkg-iterations/next")).is_err());
        assert_eq!(fs::read_to_string(&live).unwrap(), "published generation");
    }
}
