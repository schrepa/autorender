//! Cargo is always consulted, including for a warm build. Its JSON artifacts
//! replace assumptions about target directories, profiles and executable names.

use crate::{
    cache::{Entry, Store},
    files, jobs,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

/// Keep mutable Cargo output leased until callers copy or finish using it.
pub struct Build {
    pub artifacts: Vec<Value>,
    pub elapsed_ms: u128,
    pub wait_ms: u128,
    _context: Option<Entry>,
}

impl Build {
    /// The executable Cargo produced for this named target.
    pub fn executable(&self, name: &str) -> Result<PathBuf, String> {
        self.artifacts
            .iter()
            .find(|a| a["target"]["name"] == name)
            .and_then(|a| a["executable"].as_str())
            .map(PathBuf::from)
            .ok_or_else(|| format!("Cargo did not report executable {name}"))
    }
    /// The artifact of a target with a requested extension, including Wasm.
    pub fn artifact(&self, name: &str, extension: &str) -> Result<PathBuf, String> {
        self.artifacts
            .iter()
            .filter(|a| a["target"]["name"] == name)
            .flat_map(|a| a["filenames"].as_array().into_iter().flatten())
            .filter_map(Value::as_str)
            .map(PathBuf::from)
            .find(|p| p.extension().is_some_and(|e| e == extension))
            .ok_or_else(|| format!("Cargo did not report {name}.{extension}"))
    }
    /// Fresh means Cargo validated the current requested inputs without work.
    pub fn fresh(&self) -> bool {
        !self.artifacts.is_empty() && self.artifacts.iter().all(|a| a["fresh"] == true)
    }
}

/// A source-independent context per caller class and compiler. Different
/// cdylib feature domains never share mutable outputs. Workspace changes
/// clear local packages through Cargo while retaining registry dependencies.
pub fn cargo(root: &Path, arguments: &[&str], class: &str) -> Result<Build, String> {
    cargo_with(root, arguments, class, &[])
}

/// Caller-specified build-script inputs are passed to Cargo, not interpreted
/// by a parallel source fingerprinting implementation.
pub fn cargo_with(
    root: &Path,
    arguments: &[&str],
    class: &str,
    environment: &[(&str, &str)],
) -> Result<Build, String> {
    execute(root, arguments, class, environment, false)
}

/// Build one explicitly selected native executable. MSVC's reproducible link
/// removes wall-clock timestamps and random debug identifiers without changing
/// optimization or debug-information levels. Only this final target receives
/// the flag; dependency compilation remains shared and Cargo validates it.
pub fn native_binary(
    root: &Path,
    arguments: &[&str],
    class: &str,
    environment: &[(&str, &str)],
) -> Result<Build, String> {
    execute(
        root,
        arguments,
        class,
        environment,
        cfg!(target_env = "msvc"),
    )
}

fn execute(
    root: &Path,
    arguments: &[&str],
    class: &str,
    environment: &[(&str, &str)],
    reproducible_link: bool,
) -> Result<Build, String> {
    let store = Store::open(root)?;
    let identity = compiler(root)?;
    let context = store.entry("cargo", &format!("{class}:{identity}"))?;
    context.require_capacity(0)?;
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .map(|p| if p.is_absolute() { p } else { root.join(p) })
        .map(|p| {
            p.join("engine-contexts")
                .join(&files::digest(format!("{class}:{identity}").as_bytes())[..16])
        })
        .unwrap_or_else(|| context.path().join("target"));
    crate::cargo_context::select(root, arguments, &context, &target)?;
    let logs = context.path().join("last-build");
    if logs.exists() {
        files::list(&logs)?;
        fs::remove_dir_all(&logs).map_err(|e| e.to_string())?;
    }
    let mut command = Command::new("cargo");
    command
        .current_dir(root)
        .arg(if reproducible_link { "rustc" } else { "build" })
        .args(arguments)
        .arg("--message-format=json-render-diagnostics")
        .arg("--target-dir")
        .arg(&target);
    command
        .env("ENGINE_TOOL_WORKSPACE", root)
        .env("ENGINE_COMMIT", source_label(root))
        .env("CARGO_BUILD_BUILD_DIR", &target)
        .envs(environment.iter().copied());
    crate::cargo_context::pin_build_directory(&mut command, &target)?;
    if reproducible_link {
        command.args(["--", "-C", "link-arg=/Brepro"]);
    }
    let output = jobs::run(&mut command, &logs, &jobs::Limits::default())?;
    if !output.status.success() {
        let diagnostics = fs::read_to_string(&output.stdout)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter_map(|value| value["message"]["rendered"].as_str().map(str::to_owned))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "Cargo failed ({}); raw output: {}\n{}\n{diagnostics}",
            output.status,
            logs.display(),
            fs::read_to_string(&output.stderr).unwrap_or_default()
        ));
    }
    let artifacts =
        parse_artifacts(&fs::read_to_string(&output.stdout).map_err(|e| e.to_string())?)?;
    let wait_ms = context.wait_ms;
    Ok(Build {
        artifacts,
        elapsed_ms: output.elapsed_ms,
        wait_ms,
        _context: Some(context),
    })
}

/// Publish a complete executable bundle keyed by actual bytes, including
/// symbols and selected runtime libraries. Cargo freshness remains separate.
pub fn bundle(
    root: &Path,
    build: &Build,
    name: &str,
    libraries: &[PathBuf],
) -> Result<Entry, String> {
    let executable = build.executable(name)?;
    let mut paths = vec![executable.clone()];
    for artifact in build
        .artifacts
        .iter()
        .filter(|a| a["target"]["name"] == name)
    {
        for file in artifact["filenames"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            let path = PathBuf::from(file);
            if path.extension().is_some_and(|e| e == "pdb") {
                paths.push(path);
            }
        }
    }
    // Some Cargo versions report only the executable, while MSVC emits a PDB
    // beside it. It is a companion of this reported artifact, not a target-dir
    // assumption. Its hash participates in the immutable bundle identity.
    let pdb = executable.with_file_name(format!("{}.pdb", name.replace('-', "_")));
    if pdb.is_file() && !paths.contains(&pdb) {
        paths.push(pdb);
    }
    paths.extend_from_slice(libraries);
    let identities: Vec<_> = paths
        .iter()
        .map(|p| Ok(json!({"name": p.file_name(), "sha256": files::hash(p)?})))
        .collect::<Result<_, String>>()?;
    // Existing native revisions keep their bounded job cache beside the
    // executable. Preserve that runtime behavior without allowing it to alter
    // the identity or validity of the prepared executable, symbols and DLLs.
    let mutable = ["engine-job-cache"];
    let key = json!({"version": 2, "files": identities, "mutable_directories":mutable}).to_string();
    let store = Store::open(root)?;
    if let Some(entry) = store.read("native-bundle", &key)? {
        return Ok(entry);
    }
    let entry = store.entry("native-bundle", &key)?;
    if !entry.valid()? {
        let staging = entry.staging()?;
        for path in paths {
            fs::copy(
                &path,
                staging.join(path.file_name().ok_or("artifact lacks a filename")?),
            )
            .map_err(|e| e.to_string())?;
        }
        entry.publish_mutable(&mutable)?;
    }
    entry.shared()
}

/// Continuation requires identical executable/library bytes and matching
/// linker symbol IDs. The separate raw manifest still validates every PDB byte
/// on cache admission and records the exact files used for each preparation.
pub fn bundle_identity(bundle: &Entry, name: &str) -> Result<Value, String> {
    let manifest = bundle.manifest()?;
    let executable = bundle
        .data()
        .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    let mut identities = Vec::new();
    for entry in manifest["files"]
        .as_array()
        .ok_or("bundle lacks a file manifest")?
    {
        let path = entry["path"].as_str().ok_or("bundle file lacks a path")?;
        let relative = files::relative(path)?;
        if relative
            .extension()
            .is_some_and(|extension| extension == "pdb")
        {
            let id = crate::symbols::matched_id(&executable, &bundle.data().join(relative))?;
            identities.push(json!({"path":path, "symbols":id}));
        } else {
            identities.push(entry.clone());
        }
    }
    Ok(json!({"version":1, "files":identities}))
}

/// Runtime build provenance, evaluated in the workspace actually being built.
/// Cargo tracks this environment input; explicit caller inputs may override it.
pub fn source_label(root: &Path) -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .ok()
            .filter(|out| out.status.success())
    };
    let Some(head) = git(&["rev-parse", "--short=12", "HEAD"]) else {
        return "unknown".into();
    };
    let head = String::from_utf8_lossy(&head.stdout).trim().to_owned();
    let clean = git(&["status", "--porcelain"]).is_some_and(|out| out.stdout.is_empty());
    if clean { head } else { format!("{head}-dirty") }
}

/// Actual compiler metadata, evaluated in the selected source workspace.
pub fn compiler(root: &Path) -> Result<String, String> {
    let result = Command::new("rustc")
        .arg("-vV")
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    if !result.status.success() {
        return Err(String::from_utf8_lossy(&result.stderr).into_owned());
    }
    Ok(files::digest(&result.stdout))
}

/// Interpret Cargo's supported external-tools interface, never its internals.
pub fn parse_artifacts(stdout: &str) -> Result<Vec<Value>, String> {
    let mut artifacts = Vec::new();
    let mut finished = false;
    for line in stdout.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value["reason"] == "compiler-artifact" {
            artifacts.push(value);
        } else if value["reason"] == "build-finished" {
            finished = value["success"] == true;
        }
    }
    if !finished {
        return Err("Cargo omitted a successful build-finished event".into());
    }
    for artifact in &artifacts {
        for filename in artifact["filenames"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if !Path::new(filename).is_file() {
                return Err(format!("Cargo artifact is missing: {filename}"));
            }
        }
    }
    Ok(artifacts)
}

/// Copy a validated helper to an immutable generation before executing it.
/// Windows can then rebuild its mutable Cargo output while this copy runs.
pub fn helper(root: &Path) -> Result<Entry, String> {
    let manifest =
        fs::read_to_string(root.join("tools/xtask/Cargo.toml")).map_err(|e| e.to_string())?;
    let modern = manifest.contains("name = \"engine-xtask\"");
    let name = if modern { "engine-xtask" } else { "xtask" };
    let mut args = vec![
        "-p",
        "xtask",
        "--bin",
        name,
        "--profile",
        "quick",
        "--locked",
    ];
    if modern {
        args.extend(["--features", "engine"]);
    }
    let build = native_binary(root, &args, "engine-helper", &[])?;
    let exe = build.executable(name)?;
    let digest = files::hash(&exe)?;
    let store = Store::open(root)?;
    if let Some(entry) = store.read("helper", &digest)? {
        return Ok(entry);
    }
    let entry = store.entry("helper", &digest)?;
    if !entry.valid()? {
        let staging = entry.staging()?;
        fs::copy(
            exe,
            staging.join(format!("engine-xtask{}", std::env::consts::EXE_SUFFIX)),
        )
        .map_err(|e| e.to_string())?;
        files::atomic_json(
            &staging.join("build.json"),
            &json!({"version": 1, "sha256": digest, "cargo_ms": build.elapsed_ms, "lock_ms": build.wait_ms, "cargo_fresh": build.fresh()}),
        )?;
        entry.publish()?;
    }
    entry.shared()
}

/// Prepare arbitrary selected binaries and, with --hold, pin all bundles until
/// the caller closes stdin. Build contexts are released before the lease wait.
pub fn prepare_binaries(root: &Path, arguments: &[String]) -> Result<(), String> {
    use std::io::Write;
    let hold = arguments.first().is_some_and(|s| s == "--hold");
    let arguments = if hold { &arguments[1..] } else { arguments };
    if arguments.first().is_none_or(|s| s != "--") {
        return Err("usage: prepare-binaries [--hold] -- CARGO_BUILD_ARGUMENTS".into());
    }
    let mut args: Vec<_> = arguments[1..].iter().map(String::as_str).collect();
    if !args.contains(&"--locked") {
        args.push("--locked");
    }
    let build = cargo(root, &args, "native")?;
    let mut binaries = serde_json::Map::new();
    let mut leases = Vec::new();
    for artifact in &build.artifacts {
        if artifact["executable"].is_string() {
            let name = artifact["target"]["name"]
                .as_str()
                .ok_or("Cargo binary has no name")?;
            if binaries.contains_key(name) {
                return Err(format!(
                    "multiple binaries named {name}; narrow the Cargo selection"
                ));
            }
            let bundle = bundle(root, &build, name, &[])?;
            binaries.insert(
                name.into(),
                json!(
                    bundle.data().join(
                        build
                            .executable(name)?
                            .file_name()
                            .ok_or("binary has no filename")?
                    )
                ),
            );
            leases.push(bundle);
        }
    }
    if binaries.is_empty() {
        return Err("Cargo produced no selected binaries".into());
    }
    let result = json!({"version":1, "binaries":binaries, "preparation":{"cargo_fresh":build.fresh(), "cargo_ms":build.elapsed_ms, "lock_ms":build.wait_ms}});
    drop(build);
    println!("{result}");
    std::io::stdout().flush().map_err(|e| e.to_string())?;
    if hold {
        std::io::copy(&mut std::io::stdin(), &mut std::io::sink()).map_err(|e| e.to_string())?;
    }
    drop(leases);
    Ok(())
}
