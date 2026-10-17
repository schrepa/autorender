//! Real filesystem/process checks, callable without engine compilation or GPU.

use crate::{cache::Store, files, jobs};
use serde_json::json;
use std::{fs, path::Path, process::Command, time::Duration};

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

/// Exercise corruption, incomplete publication, active eviction and cancellation.
pub fn run(root: &Path) -> Result<(), String> {
    let output = root.join("target/tooling-selfcheck").join(files::unique());
    fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let store = Store::at(&output, 1024 * 1024)?;
    let entry = store.entry("fixture", "one")?;
    require(!entry.valid()?, "empty cache admitted an output")?;
    let scratch = entry.staging()?;
    fs::write(scratch.join("result"), b"verified output").map_err(|e| e.to_string())?;
    require(!entry.valid()?, "unpublished output admitted")?;
    entry.publish()?;
    require(entry.valid()?, "valid output not admitted")?;
    fs::write(entry.data().join("result"), b"corrupt output").map_err(|e| e.to_string())?;
    require(!entry.valid()?, "corruption admitted")?;
    fs::write(entry.data().join("result"), b"verified output").map_err(|e| e.to_string())?;
    require(entry.valid()?, "restored output not admitted")?;
    let exported = output.join("exported");
    files::copy_tree(&entry.data(), &exported)?;
    let export_manifest = files::manifest(&exported)?;
    let mut tiny = Store::at(&output, 1)?;
    let pinned = tiny.collect(0)?;
    require(
        pinned["within_budget"] == false && entry.valid()?,
        "collection removed active data",
    )?;
    drop(entry);
    let collected = tiny.configure(4096)?;
    require(
        collected["within_budget"] == true,
        "idle collection did not meet budget",
    )?;
    tiny.configure(1)?;
    require(
        files::verify(&exported, &export_manifest)?,
        "eviction broke exported result",
    )?;
    cancellation(root, &output)?;
    concurrent(root, &output)?;
    storage_lifecycle(root, &output)?;
    crate::tooling_workflows::builds(&output)?;
    crate::tooling_workflows::continuation(&output)?;
    crate::analyzer_check::run(root, &output)?;
    files::atomic_json(
        &output.join("result.json"),
        &json!({"version": 1, "passed": ["empty", "incomplete", "corrupt", "active lease", "idle eviction", "portable export", "deadline", "concurrent publication", "terminated lease recovery", "bounded storage churn", "Cargo edit and feature freshness", "relocated cdylib workspaces", "A-B-A immutable builds", "interrupted continuation", "portable nested artifacts", "continuation compatibility", "analyzer cold warm disabled equivalence", "failed analyzer rejection"]}),
    )?;
    println!(
        "tooling selfcheck passed; {}",
        output.join("result.json").display()
    );
    Ok(())
}

fn cancellation(root: &Path, output: &Path) -> Result<(), String> {
    let mut command = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    command.current_dir(root).args(["tooling-child", "wait"]);
    let result = jobs::run(
        &mut command,
        &output.join("cancelled"),
        &jobs::Limits {
            timeout: Duration::from_millis(250),
            cancel: None,
            ..Default::default()
        },
    );
    require(
        result.is_err_and(|error| error.contains("deadline exceeded")),
        "deadline did not stop the owned process",
    )
}

fn concurrent(root: &Path, output: &Path) -> Result<(), String> {
    let mut children = Vec::new();
    for _ in 0..4 {
        let child = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .current_dir(root)
            .args(["tooling-child", "publish"])
            .arg(output.join("concurrent"))
            .spawn()
            .map_err(|e| e.to_string())?;
        children.push(child);
    }
    for mut child in children {
        require(
            child.wait().map_err(|e| e.to_string())?.success(),
            "concurrent publisher failed",
        )?;
    }
    let store = Store::at(&output.join("concurrent"), 1024 * 1024)?;
    let entry = store.entry("concurrent", "same-input")?;
    require(entry.valid()?, "concurrent publication incomplete")?;
    require(
        fs::read_to_string(output.join("concurrent/computations")).map_err(|e| e.to_string())?
            == "one\n",
        "a missing key was computed more than once",
    )
}

fn storage_lifecycle(root: &Path, output: &Path) -> Result<(), String> {
    let dead = output.join("terminated");
    let mut command = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    command
        .current_dir(root)
        .args(["tooling-child", "lease"])
        .arg(&dead);
    let stopped = jobs::run(
        &mut command,
        &output.join("terminated-log"),
        &jobs::Limits {
            timeout: Duration::from_secs(1),
            cancel: None,
            ..Default::default()
        },
    );
    require(
        stopped.is_err_and(|error| error.contains("deadline exceeded")),
        "lease holder was not terminated",
    )?;
    require(
        dead.join("ready").is_file(),
        "lease holder never acquired the lock",
    )?;
    let store = Store::at(&dead, 65536)?;
    let recovered = store.entry("fixture", "terminated")?;
    require(!recovered.valid()?, "killed staging output was admitted")?;
    drop(recovered);
    let mut peak = 0;
    for iteration in 0..48 {
        let entry = store.entry("fixture", &format!("generation-{iteration}"))?;
        let scratch = entry.staging()?;
        fs::write(scratch.join("result"), vec![7; 8192]).map_err(|e| e.to_string())?;
        entry.publish()?;
        drop(entry);
        let report = store.collect(0)?;
        require(
            report["within_budget"] == true,
            "storage grew beyond idle budget",
        )?;
        peak = peak.max(
            report["logical_bytes"]
                .as_u64()
                .ok_or("missing accounted bytes")?,
        );
    }
    files::atomic_json(
        &output.join("storage.json"),
        &json!({"generations":48, "idle_peak_bytes":peak, "max_bytes":65536}),
    )
}

/// Selfcheck subprocess protocol; intentionally absent from normal help.
pub fn child(args: &[String]) -> Result<u8, String> {
    match args.first().map(String::as_str) {
        Some("analyzer") => crate::analyzer_check::child(args),
        Some("wait") => {
            std::thread::sleep(Duration::from_secs(60));
            Ok(0)
        }
        Some("lease") => {
            let root = Path::new(args.get(1).ok_or("missing store path")?);
            let store = Store::at(root, 65536)?;
            let entry = store.entry("fixture", "terminated")?;
            fs::write(entry.staging()?.join("partial"), b"unfinished")
                .map_err(|e| e.to_string())?;
            fs::write(root.join("ready"), b"held").map_err(|e| e.to_string())?;
            std::thread::sleep(Duration::from_secs(60));
            Ok(0)
        }
        Some("publish") => {
            let root = Path::new(args.get(1).ok_or("missing store path")?);
            let store = Store::at(root, 1024 * 1024)?;
            let entry = store.entry("concurrent", "same-input")?;
            if !entry.valid()? {
                use std::io::Write;
                let mut count = fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(root.join("computations"))
                    .map_err(|e| e.to_string())?;
                count.write_all(b"one\n").map_err(|e| e.to_string())?;
                let staging = entry.staging()?;
                fs::write(staging.join("output"), b"one computation").map_err(|e| e.to_string())?;
                entry.publish()?;
            }
            Ok(0)
        }
        _ => Err("unknown tooling selfcheck child".into()),
    }
}
