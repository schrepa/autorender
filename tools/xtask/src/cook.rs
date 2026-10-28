use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::heavy::HeavyRun;
use crate::pipeline::QUICK;
use crate::process::run;

const PRESETS: &[&str] = &[
    "visual",
    "city",
    "metro",
    "dense",
    "highlands",
    "mixed",
    "motion",
    "queries",
    "sensors",
];

const COOKED: &str = "web/public/cooked";

const CHECK_FRAMES: &str = "600";

pub fn cook(root: &Path, options: &[String]) -> Result<(), String> {
    let check = options.iter().any(|option| option == "--check");
    let named: Vec<&str> = options
        .iter()
        .map(String::as_str)
        .filter(|option| *option != "--check")
        .collect();
    if let Some(option) = named.iter().find(|option| option.starts_with("--")) {
        return Err(format!(
            "unknown option {option}; usage: cook [--check] [PRESET...]"
        ));
    }
    let presets: Vec<&str> = if named.is_empty() {
        PRESETS.to_vec()
    } else {
        named
    };
    run(
        root,
        "cargo",
        &["build", "-p", "stress-lab", "--profile", QUICK, "--locked"],
    )?;
    let program = root
        .join("target")
        .join(QUICK)
        .join(format!("stress-lab-native{}", std::env::consts::EXE_SUFFIX));
    let program = program.to_string_lossy().into_owned();
    let folder = root.join(COOKED);
    fs::create_dir_all(&folder)
        .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
    for preset in &presets {
        let _lock = HeavyRun::hold(&format!("cook {preset}"))?;
        if check {
            check_pack(root, &program, preset)?;
        } else {
            let pack = format!("{COOKED}/{preset}.pack");
            run(root, &program, &["--preset", preset, "--cook", &pack])?;
        }
    }
    write_index(&folder)
}

fn check_pack(root: &Path, program: &str, preset: &str) -> Result<(), String> {
    let output = std::process::Command::new(program)
        .args([
            "--preset",
            preset,
            "--offscreen",
            CHECK_FRAMES,
            "--size",
            "640x360",
            "--jobcache",
            "verify",
        ])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{preset} did not run on its pack: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let report: Value = text
        .find('{')
        .and_then(|start| serde_json::from_str(&text[start..]).ok())
        .ok_or_else(|| format!("{preset} printed no report"))?;
    let stats = &report["stats"];
    let jobs = &stats["jobs"];
    let (hits, mismatches) = (
        jobs["cache_hits"].as_u64().unwrap_or(0),
        jobs["cache_mismatches"].as_u64().unwrap_or(u64::MAX),
    );
    let (listed, unlisted) = (
        stats["pipelines_listed"].as_u64().unwrap_or(0),
        stats["pipelines_unlisted"].as_u64().unwrap_or(u64::MAX),
    );
    println!(
        "{preset}: {hits} records read, {mismatches} differing from a fresh run; \
         {listed} pipelines created up front, {unlisted} after"
    );
    if hits == 0 {
        return Err(format!("{preset} read no record from its pack"));
    }
    if mismatches > 0 {
        return Err(format!(
            "{preset}: {mismatches} records differ from a fresh run of their jobs"
        ));
    }
    if listed == 0 {
        return Err(format!(
            "{preset} created no pipeline from its pack's pipeline list"
        ));
    }
    if unlisted > 0 {
        return Err(format!(
            "{preset} created {unlisted} pipelines its pipeline list lacks"
        ));
    }
    Ok(())
}

fn write_index(folder: &Path) -> Result<(), String> {
    let mut packs: Vec<String> = fs::read_dir(folder)
        .map_err(|error| format!("cannot list {}: {error}", folder.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".pack"))
        .collect();
    packs.sort();
    let index = folder.join("index.json");
    fs::write(
        &index,
        serde_json::to_string(&packs).expect("names serialize") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", index.display()))
}
