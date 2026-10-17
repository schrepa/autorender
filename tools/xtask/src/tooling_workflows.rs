//! Integration scenarios for source switching and durable continuation.

use crate::{
    builds,
    execution::{Request, read_result},
    files,
};
use serde_json::json;
use std::{fs, path::Path, process::Command};

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}

/// Exercise real Cargo dependency tracking and artifacts, without engine/GPU
/// compilation. The included source deliberately remains untracked in Git.
pub fn builds(output: &Path) -> Result<(), String> {
    let root = output.join("cargo-fixture");
    let other = output.join("relocated-cargo-fixture");
    fixture(&root, 1)?;
    fixture(&other, 2)?;
    require(
        Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(&root)
            .status()
            .map_err(|e| e.to_string())?
            .success(),
        "fixture git init failed",
    )?;
    let build = |extra: &[&str]| {
        let mut arguments = vec!["--bin", "tooling-fixture"];
        arguments.extend_from_slice(extra);
        builds::native_binary(&root, &arguments, "fixture", &[])
    };
    let first = build(&[])?;
    let kept = builds::bundle(&root, &first, "tooling-fixture", &[])?;
    drop(first);
    let check = |expected: &str, extra: &[&str]| -> Result<(), String> {
        let built = build(extra)?;
        if cfg!(target_env = "msvc") && expected == "2" {
            let executable = kept.data().join("tooling-fixture.exe");
            let pdb = built
                .executable("tooling-fixture")?
                .with_file_name("tooling_fixture.pdb");
            require(
                crate::symbols::matched_id(&executable, &pdb).is_err(),
                "wrong symbols admitted",
            )?;
        }
        let output = Command::new(built.executable("tooling-fixture")?)
            .output()
            .map_err(|e| e.to_string())?;
        require(
            output.status.success() && String::from_utf8_lossy(&output.stdout).trim() == expected,
            "Cargo produced the wrong source/feature output",
        )
    };
    let warm = build(&[])?;
    require(warm.fresh(), "unchanged Cargo request rebuilt")?;
    let identity = kept.manifest()?;
    let compatible = builds::bundle_identity(&kept, "tooling-fixture")?;
    fs::create_dir_all(kept.data().join("engine-job-cache")).map_err(|e| e.to_string())?;
    fs::write(kept.data().join("engine-job-cache/runtime"), "warm jobs")
        .map_err(|e| e.to_string())?;
    let reused = builds::bundle(&root, &warm, "tooling-fixture", &[])?;
    require(
        reused.path() == kept.path() && reused.valid()? && reused.manifest()? == identity,
        "runtime job state invalidated or replaced a leased preparation",
    )?;
    fs::write(kept.data().join("unexpected"), "not runtime state").map_err(|e| e.to_string())?;
    require(!kept.valid()?, "undeclared bundle mutation admitted")?;
    fs::remove_file(kept.data().join("unexpected")).map_err(|e| e.to_string())?;
    drop(reused);
    drop(warm);
    fs::write(root.join("src/value.rs"), "const VALUE: u8 = 2;\n").map_err(|e| e.to_string())?;
    check("2", &[])?;
    check("3", &["--features", "alternate"])?;
    fs::write(root.join("src/value.rs"), "const VALUE: u8 = 1;\n").map_err(|e| e.to_string())?;
    check("1", &[])?;
    // Both source folders predate the builds. Relative cdylib dep-info alone
    // can incorrectly admit the first workspace's outputs for the second.
    check(
        "2",
        &[
            "--manifest-path",
            &other.join("Cargo.toml").display().to_string(),
        ],
    )?;
    check("1", &[])?;
    check(
        "2",
        &[
            "--manifest-path",
            &other.join("Cargo.toml").display().to_string(),
            "--profile",
            "quick",
        ],
    )?;
    check("1", &["--profile", "quick"])?;
    check("1", &[])?;
    let returned = build(&[])?;
    require(returned.fresh(), "returning workspace did not warm")?;
    if cfg!(target_env = "msvc") {
        let again = builds::bundle(&root, &returned, "tooling-fixture", &[])?;
        require(
            builds::bundle_identity(&again, "tooling-fixture")? == compatible,
            "rebuild changed reproducible executable or symbols",
        )?;
        malformed_symbols(output, &kept)?;
    }
    let original = Command::new(
        kept.data()
            .join(format!("tooling-fixture{}", std::env::consts::EXE_SUFFIX)),
    )
    .output()
    .map_err(|e| e.to_string())?;
    require(
        original.status.success() && String::from_utf8_lossy(&original.stdout).trim() == "1",
        "later builds replaced a leased executable",
    )?;
    Ok(())
}

fn malformed_symbols(output: &Path, bundle: &crate::cache::Entry) -> Result<(), String> {
    let executable = bundle.data().join("tooling-fixture.exe");
    let source = fs::read(bundle.data().join("tooling_fixture.pdb")).map_err(|e| e.to_string())?;
    let path = output.join("malformed.pdb");
    for length in [0, 31, 55, 4096] {
        fs::write(&path, &source[..length]).map_err(|e| e.to_string())?;
        require(
            crate::symbols::matched_id(&executable, &path).is_err(),
            "truncated PDB admitted",
        )?;
    }
    for field in [32, 40, 44, 52] {
        let mut bytes = source.clone();
        bytes[field..field + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        fs::write(&path, bytes).map_err(|e| e.to_string())?;
        require(
            crate::symbols::matched_id(&executable, &path).is_err(),
            "invalid PDB geometry admitted",
        )?;
    }
    let path = output.join("truncated.exe");
    fs::write(&path, b"MZ").map_err(|e| e.to_string())?;
    require(
        crate::symbols::matched_id(&path, &bundle.data().join("tooling_fixture.pdb")).is_err(),
        "truncated PE admitted",
    )
}

fn fixture(root: &Path, value: u8) -> Result<(), String> {
    fs::create_dir_all(root.join("src")).map_err(|e| e.to_string())?;
    fs::write(root.join("Cargo.toml"), "[package]\nname = \"tooling-fixture\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[workspace]\n[features]\nalternate = []\n[lib]\ncrate-type = [\"cdylib\", \"rlib\"]\n[profile.quick]\ninherits = \"dev\"\n").map_err(|e| e.to_string())?;
    fs::write(root.join("src/lib.rs"), "include!(\"value.rs\");\npub fn value() -> u8 { if cfg!(feature = \"alternate\") { 3 } else { VALUE } }\n").map_err(|e| e.to_string())?;
    fs::write(
        root.join("src/main.rs"),
        "fn main() { println!(\"{}\", tooling_fixture::value()); }\n",
    )
    .map_err(|e| e.to_string())?;
    fs::write(
        root.join("src/value.rs"),
        format!("const VALUE: u8 = {value};\n"),
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

/// Exercise request admission and portable artifacts through the same public
/// result API used by measurements, including a nested shader report.
pub fn continuation(output: &Path) -> Result<(), String> {
    preparation_admission()?;
    let request_path = output.join("request.json");
    let report_path = output.join("report.json");
    let spec = json!({"version":1, "operation":"measure", "setup":{"runtime":"native"}, "environment":{"driver":"one"}, "pairs":2, "base_sha":"0000000000000000000000000000000000000000", "sha":"0000000000000000000000000000000000000000", "workloads":[{"name":"fixture", "feature":"none", "arms":{"base":{"sha":"0000000000000000000000000000000000000000", "options":{}}, "candidate":{"sha":"0000000000000000000000000000000000000000", "options":{}}}, "workload":{}}]});
    for (field, invalid) in [
        ("deadline", json!("2026-02-30T00:00:00Z")),
        ("pairs", json!(-1)),
        ("task", json!(1)),
    ] {
        let mut bad = spec.clone();
        bad[field] = invalid;
        require(
            crate::request_contract::validate(&bad, "measure").is_err(),
            "invalid request was admitted",
        )?;
    }
    files::atomic_json(&request_path, &spec)?;
    let args = vec![
        "--request".into(),
        request_path.display().to_string(),
        "--out".into(),
        report_path.display().to_string(),
    ];
    let mut cached_output = args.clone();
    cached_output[3] = output
        .join("engine-cache-v1/forbidden/report.json")
        .display()
        .to_string();
    require(
        Request::read(&cached_output, "measure").is_err_and(|e| e.contains("outside disposable")),
        "caller results were admitted inside disposable storage",
    )?;
    let execution = Request::read(&args, "measure")?;
    require(
        Request::read(&args, "measure").is_err_and(|error| error.contains("report already in use")),
        "two writers acquired the same result path",
    )?;
    let isa = execution.folder.join("shader.isa");
    fs::write(&isa, "v_mov_b32 v0, 0\n").map_err(|e| e.to_string())?;
    let shader = execution.folder.join("shader.json");
    files::atomic_json(&shader, &json!({"entries":[{"isa":isa}]}))?;
    execution.checkpoint("runtime", &json!({"workloads":[{"runs":[{"arm":"base"},{"arm":"candidate"}]}], "shader":shader, "incomplete":true}))?;
    // A killed process can leave a new partial run after this checkpoint.
    fs::write(
        execution.folder.join("interrupted.raw"),
        "partial unpaired output",
    )
    .map_err(|e| e.to_string())?;
    let previous = read_result(&report_path)?;
    require(
        previous["recovery"]["uncommitted_files"]
            .as_array()
            .is_some_and(|files| files.len() == 1),
        "interrupted evidence was lost or promoted",
    )?;
    require(
        previous["workloads"][0]["runs"]
            .as_array()
            .is_some_and(|v| v.len() == 2),
        "checkpoint lost a completed pair",
    )?;
    let mut extend = args.clone();
    extend.extend(["--extend".into(), report_path.display().to_string()]);
    let previous_folder = execution.folder.clone();
    drop(execution);
    let next = Request::read(&extend, "measure")?;
    next.checkpoint(
        "complete",
        next.previous.as_ref().ok_or("missing continuation")?,
    )?;
    drop(next);
    // The new bundle must survive removal of the old attempt and relocation
    // of the entire caller-owned result directory.
    fs::remove_dir_all(&previous_folder).map_err(|e| e.to_string())?;
    let resumed = read_result(&report_path)?;
    let nested = files::json_file(Path::new(
        resumed["shader"].as_str().ok_or("missing shader")?,
    ))?;
    let reference = nested["entries"][0]["isa"]
        .as_str()
        .ok_or("missing nested artifact")?;
    require(
        output.join(reference).is_file(),
        "continued shader report references an old cache or attempt",
    )?;
    let before = fs::read(&report_path).map_err(|e| e.to_string())?;
    let mut changed = spec;
    changed["environment"]["driver"] = json!("two");
    files::atomic_json(&request_path, &changed)?;
    require(
        Request::read(&extend, "measure").is_err(),
        "incompatible driver admitted for continuation",
    )?;
    require(
        fs::read(&report_path).map_err(|e| e.to_string())? == before,
        "failed admission overwrote the original report",
    )?;
    fs::write(output.join(reference), "corrupt").map_err(|e| e.to_string())?;
    require(
        read_result(&report_path).is_err(),
        "corrupt continuation artifact admitted",
    )
}

fn preparation_admission() -> Result<(), String> {
    let identity = json!({"binary":"one", "symbols":"one"});
    let mut previous = json!({"execution":{"stage":"preparation"}, "workloads":[], "checks":[]});
    require(
        crate::execution::admit_builds(&previous, &identity).is_ok(),
        "empty interrupted preparation cannot resume",
    )?;
    previous["execution"]["stage"] = json!("runtime");
    require(
        crate::execution::admit_builds(&previous, &identity).is_err(),
        "runtime admitted missing build identities",
    )?;
    previous["build_identities"] = identity.clone();
    require(
        crate::execution::admit_builds(&previous, &identity).is_ok(),
        "identical build identities rejected",
    )?;
    previous["build_identities"]["symbols"] = json!("two");
    require(
        crate::execution::admit_builds(&previous, &identity).is_err(),
        "changed symbols admitted for continuation",
    )
}
