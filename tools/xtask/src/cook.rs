//! `cargo xtask cook`: the Stress Lab's presets cooked into packs, every
//! record each one's content needs (`--cook`), in `web/public/cooked/`,
//! which native hosts read whole and the page reads by range through the
//! list `index.json` gives; and `--check`, the cook's parity: each preset
//! runs on its pack with every record made again and compared
//! (`--jobcache verify`), and must read records from it with none
//! differing.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::heavy::HeavyRun;
use crate::pipeline::QUICK;
use crate::process::run;

/// The presets cooked when none is named: every one, as each scene's
/// meshes have levels of detail and impostors the cook makes.
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

/// Where packs are written and read, from the repository's root.
const COOKED: &str = "web/public/cooked";

/// Frames a check draws on its pack.
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

/// Runs `preset` on its pack, every record made again and compared: it
/// must read records from the pack and find none differing.
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
    let jobs = &report["stats"]["jobs"];
    let (hits, mismatches) = (
        jobs["cache_hits"].as_u64().unwrap_or(0),
        jobs["cache_mismatches"].as_u64().unwrap_or(u64::MAX),
    );
    println!("{preset}: {hits} records read, {mismatches} differing from a fresh run");
    if hits == 0 {
        return Err(format!("{preset} read no record from its pack"));
    }
    if mismatches > 0 {
        return Err(format!(
            "{preset}: {mismatches} records differ from a fresh run of their jobs"
        ));
    }
    Ok(())
}

/// Lists every pack in `folder` in its `index.json`, which the page reads.
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
