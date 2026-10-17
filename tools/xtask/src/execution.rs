//! Versioned local operation requests and caller-owned, portable results.
//! There are no assignment, service, upload or acceptance concepts here.

use crate::files;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

/// Parsed CLI and immutable request. Pair count/deadline may increase on
/// explicit continuation; all experimental inputs must stay identical.
pub struct Request {
    pub spec: Value,
    pub out: PathBuf,
    pub previous: Option<Value>,
    pub digest: String,
    pub compatibility: String,
    pub run_id: String,
    pub folder: PathBuf,
    pub started: Instant,
    stages: RefCell<Vec<Value>>,
    _report_lease: fs::File,
}

impl Request {
    /// Read a request, validate its version/operation, and create a new attempt.
    /// Output always belongs to the caller, outside disposable cache storage.
    pub fn read(arguments: &[String], operation: &str) -> Result<Self, String> {
        let mut request = None;
        let mut out = None;
        let mut previous = None;
        let mut args = arguments.iter();
        while let Some(key) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("{key} requires a path"))?;
            match key.as_str() {
                "--request" => request = Some(PathBuf::from(value)),
                "--out" => out = Some(PathBuf::from(value)),
                "--extend" => previous = Some(PathBuf::from(value)),
                _ => {
                    return Err(format!(
                        "unknown option {key}; use {operation} --request REQUEST.json --out REPORT.json [--extend REPORT.json]"
                    ));
                }
            }
        }
        let spec = files::json_file(&request.ok_or("--request is required")?)?;
        crate::request_contract::validate(&spec, operation)?;
        let out = out.ok_or("--out is required")?;
        let out = if out.is_absolute() {
            out
        } else {
            std::env::current_dir()
                .map_err(|e| e.to_string())?
                .join(out)
        };
        let report_lease = report_lease(&out)?;
        let previous_path = previous;
        let mut previous = previous_path.as_deref().map(read_result).transpose()?;
        let digest = files::digest(spec.to_string().as_bytes());
        let mut method = spec.clone();
        for key in ["pairs", "deadline", "result_max_bytes"] {
            method
                .as_object_mut()
                .ok_or("request must be an object")?
                .remove(key);
        }
        let compatibility = files::digest(method.to_string().as_bytes());
        if let Some(previous) = &previous
            && (previous["version"] != 1 || previous["compatibility"] != compatibility)
        {
            return Err("continuation inputs differ (source, method, setup, environment or requested checks)".into());
        }
        let run_id = files::unique();
        let name = out
            .file_stem()
            .and_then(|name| name.to_str())
            .ok_or("invalid report file name")?;
        let folder = out
            .parent()
            .ok_or("report needs a parent directory")?
            .join(format!("{name}.files"))
            .join(&run_id);
        fs::create_dir_all(&folder).map_err(|e| e.to_string())?;
        if let (Some(path), Some(report)) = (previous_path, previous.as_mut()) {
            let old = files::json_file(&path)?;
            let old_root = path
                .parent()
                .unwrap_or(Path::new("."))
                .join(files::relative(
                    old["artifacts"]["root"]
                        .as_str()
                        .ok_or("missing artifact root")?,
                )?);
            let kept = folder.join("continued");
            files::copy_tree(&old_root, &kept)?;
            relocate(report, &files::canonical(&old_root)?, &kept);
            relocate_documents(
                &kept,
                path.parent().unwrap_or(Path::new(".")),
                &files::canonical(&old_root)?,
            )?;
        }
        files::atomic_json(&folder.join("request.json"), &spec)?;
        let this = Self {
            spec,
            out,
            previous,
            digest,
            compatibility,
            run_id,
            folder,
            started: Instant::now(),
            stages: RefCell::new(Vec::new()),
            _report_lease: report_lease,
        };
        // Keep the original report untouched until admission (actual tool,
        // build and setup compatibility) has succeeded. Incompatible requests
        // must not overwrite the evidence they were asked to extend.
        files::atomic_json(
            &this.folder.join("status.json"),
            &json!({"stage": "preparing", "incomplete": true}),
        )?;
        Ok(this)
    }

    /// Atomic, portable checkpoint. Previous complete observations survive any
    /// later build, profile, analyzer or check failure. Never marks a partial
    /// pair as complete; callers journal those attempts separately.
    pub fn checkpoint(&self, stage: &str, observations: &Value) -> Result<(), String> {
        let mut report = observations.clone();
        report["version"] = json!(1);
        report["operation"] = self.spec["operation"].clone();
        report["run_id"] = json!(self.run_id);
        report["request_sha256"] = json!(self.digest);
        report["compatibility"] = json!(self.compatibility);
        let elapsed = self.started.elapsed().as_millis();
        let mut stages = self.stages.borrow_mut();
        let previous = stages
            .last()
            .and_then(|s| s["elapsed_ms"].as_u64())
            .unwrap_or(0);
        stages.push(json!({"stage": stage, "elapsed_ms": elapsed, "since_previous_ms": elapsed.saturating_sub(u128::from(previous))}));
        report["execution"] = json!({"stage": stage, "elapsed_ms": elapsed, "checkpoints": *stages, "continued_from": self.previous.as_ref().map(|p| &p["run_id"])});
        let base = self.out.parent().ok_or("report needs a parent")?;
        portable(&mut report, base, &self.folder)?;
        let artifacts = files::manifest(&self.folder)?;
        let prefix = self
            .folder
            .strip_prefix(base)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .replace('\\', "/");
        report["artifacts"] = json!({"root": prefix, "files": artifacts});
        if let Some(limit) = self.spec["result_max_bytes"].as_u64() {
            let bytes = files::bytes(&self.folder)?;
            if bytes > limit {
                return Err(format!(
                    "result quota exceeded: {bytes} > {limit}; prior checkpoints and raw output retained in {}",
                    self.folder.display()
                ));
            }
        }
        files::atomic_json(&self.out, &report)
    }

    /// Immutable attempt evidence, including failed or unpaired observations.
    pub fn attempt(&self, label: &str, value: &Value) -> Result<(), String> {
        let folder = self.folder.join("attempts");
        files::atomic_json(
            &folder.join(format!("{}-{label}.json", files::unique())),
            value,
        )
    }
}

fn report_lease(out: &Path) -> Result<fs::File, String> {
    let parent = out.parent().ok_or("report needs a parent directory")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let parent = files::canonical(parent)?;
    for ancestor in parent.ancestors() {
        let owner = ancestor.join("owner.json");
        if owner.is_file() && files::json_file(&owner).is_ok_and(|v| v["kind"] == "engine-cache-v1")
        {
            return Err("result directory must be outside disposable engine cache storage".into());
        }
    }
    let mut name = out.as_os_str().to_owned();
    name.push(".lock");
    let lease = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(Path::new(&name))
        .map_err(|e| e.to_string())?;
    lease
        .try_lock()
        .map_err(|e| format!("report already in use ({}): {e}", out.display()))?;
    Ok(lease)
}

/// Verify the retained bundle before resolving relative paths for execution.
pub fn read_result(path: &Path) -> Result<Value, String> {
    let path = files::canonical(path)?;
    let mut value = files::json_file(&path)?;
    let base = path.parent().unwrap_or(Path::new("."));
    let root = files::relative(
        value["artifacts"]["root"]
            .as_str()
            .ok_or("result has no artifact root")?,
    )?;
    let root = base.join(root);
    let actual = files::manifest(&root)?;
    let expected = value["artifacts"]["files"]
        .as_array()
        .ok_or("result has no artifact manifest")?;
    let actual = actual.as_array().ok_or("invalid artifact manifest")?;
    // A killed run can leave newer, incomplete attempts after its last atomic
    // checkpoint. Verify every committed byte, retain the extra evidence, and
    // never promote those extra files into completed observations.
    if expected.iter().any(|file| !actual.contains(file)) {
        return Err("continuation artifacts are incomplete or modified".into());
    }
    let uncommitted: Vec<_> = actual
        .iter()
        .filter(|file| !expected.contains(file))
        .cloned()
        .collect();
    value["recovery"] = json!({"uncommitted_files": uncommitted});
    resolve(&mut value, base);
    Ok(value)
}

fn relocate_documents(kept: &Path, old_base: &Path, old_root: &Path) -> Result<(), String> {
    let old_base = files::canonical(old_base)?;
    for file in files::list(kept)? {
        if file
            .extension()
            .is_some_and(|extension| extension == "json")
            && let Ok(mut document) = files::json_file(&kept.join(&file))
            && document["entries"].is_array()
        {
            resolve(&mut document, &old_base);
            relocate(&mut document, old_root, kept);
            files::atomic_json(&kept.join(file), &document)?;
        }
    }
    Ok(())
}

fn resolve(value: &mut Value, base: &Path) {
    match value {
        Value::String(name) if base.join(&*name).is_file() && !Path::new(name).is_absolute() => {
            *name = base.join(&*name).to_string_lossy().into_owned()
        }
        Value::Array(values) => values.iter_mut().for_each(|value| resolve(value, base)),
        Value::Object(values) => values.values_mut().for_each(|value| resolve(value, base)),
        _ => {}
    }
}

/// Copy external files into this attempt, then make references relative to the
/// report. JSON artifacts are rewritten transitively (shader reports/profiles).
fn portable(value: &mut Value, base: &Path, folder: &Path) -> Result<(), String> {
    match value {
        Value::String(name) if Path::new(name).is_absolute() && Path::new(name).is_file() => {
            let source = PathBuf::from(&*name);
            let target = if source.starts_with(folder) {
                source.clone()
            } else {
                let into = folder.join("imported").join(files::hash(&source)?);
                fs::create_dir_all(&into).map_err(|e| e.to_string())?;
                let target = into.join(source.file_name().ok_or("artifact has no name")?);
                fs::copy(&source, &target).map_err(|e| e.to_string())?;
                target
            };
            if target.extension().is_some_and(|s| s == "json")
                && let Ok(mut document) = files::json_file(&target)
            {
                // Only rewrite engine artifact documents, not request/input
                // metadata or arbitrary source files named by diagnostics.
                if document["entries"].is_array() {
                    portable(&mut document, base, folder)?;
                    files::atomic_json(&target, &document)?;
                }
            }
            *name = target
                .strip_prefix(base)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .replace('\\', "/");
        }
        Value::Array(values) => {
            for value in values {
                portable(value, base, folder)?;
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                portable(value, base, folder)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn relocate(value: &mut Value, from: &Path, into: &Path) {
    match value {
        Value::String(name) => {
            if let Ok(relative) = Path::new(name).strip_prefix(from) {
                *name = into.join(relative).to_string_lossy().into_owned();
            }
        }
        Value::Array(values) => values.iter_mut().for_each(|v| relocate(v, from, into)),
        Value::Object(values) => values.values_mut().for_each(|v| relocate(v, from, into)),
        _ => {}
    }
}

/// An interrupted preparation has no observations tied to a build yet. Once
/// runtime admission occurred, continuation requires every actual identity.
pub fn admit_builds(previous: &Value, current: &Value) -> Result<(), String> {
    let unstarted = previous["build_identities"].is_null()
        && previous["execution"]["stage"] == "preparation"
        && ["workloads", "checks"]
            .iter()
            .all(|key| previous[key].as_array().is_some_and(Vec::is_empty));
    if unstarted || previous["build_identities"] == *current {
        Ok(())
    } else {
        Err("continuation requires identical binaries, symbols, runtime libraries, generators and toolchains for every arm".into())
    }
}

/// Identify the actual executable and the selected source tree. Working edits
/// are permitted and recorded; the executable hash identifies the tool in use.
pub fn tool_identity(root: &Path) -> Result<Value, String> {
    let git = |args: &[&str]| -> Result<String, String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().into())
    };
    Ok(
        json!({"executable_sha256": files::hash(&std::env::current_exe().map_err(|e| e.to_string())?)?, "revision": git(&["rev-parse", "HEAD"])?, "working_tree_changed": !git(&["status", "--porcelain"])?.is_empty()}),
    )
}

/// JSON Schema for the versioned local operation request.
pub fn schema(operation: &str) -> Value {
    crate::request_contract::schema(operation)
}
