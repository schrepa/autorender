//! `cargo xtask selfcheck [measure] [checks]`: the measurement tools run
//! against themselves, each an A/A comparison whose two sides must come out
//! equal, on the working tree as a commit (made without touching any ref,
//! the same content giving the same commit, so builds are reused; the
//! clean HEAD as it is).
//!
//! - measure: a measure task whose two arms are that commit, its tools
//!   that commit's (this program), on Field (`field-90k`) for
//!   [`PAIRS`] clean pairs: it must complete, every run settled; every run
//!   counts the same work; the arms' median frame times ([`FRAME_TIME`])
//!   lie within [`TIME_RATIO`] of each other, judged only where other
//!   programs used at most [`BUSY_CPU_PCT`] of the processors (else it
//!   could not check).
//! - checks: one check ([`CHECK`]) on that commit with its own scripts and
//!   with the scripts of the same commit checked out again: both must hold
//!   at their first attempt; expected to fail instead, it must not hold
//!   after every attempt.
//!
//! Its files are in target/selfcheck/; it exits 3 when an A/A comparison
//! differed, 1 when a tool could not run.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};

use crate::Failure;
use crate::measure::{git, read_json};

/// Clean pairs the measure self-check asks for.
const PAIRS: u64 = 2;
/// The most the arms' median frame times may differ by, as a ratio.
const TIME_RATIO: f64 = 1.25;
/// The frame time compared: the tool's own clock's, from the host's marks.
const FRAME_TIME: &str = "wall:frame_ms_mean";
/// The most of the processors other programs may use during a run whose
/// frame time is compared, in percent: busier, times say nothing (at up to
/// 46% the arms came out 1.011 apart, at 48-56% 0.705 apart).
const BUSY_CPU_PCT: f64 = 50.0;
/// The check the checks self-check runs: quick, and needing no GPU.
const CHECK: &str = "agent-lifecycle-smoke";
/// What the self-checks are, in the order they run.
pub(crate) const KINDS: [&str; 3] = ["measure", "checks", "tooling"];

pub fn selfcheck(root: &Path, arguments: &[String]) -> Result<(), Failure> {
    let kinds: Vec<&str> = if arguments.is_empty() {
        KINDS.to_vec()
    } else {
        arguments
            .iter()
            .map(|kind| {
                KINDS
                    .into_iter()
                    .find(|known| known == kind)
                    .ok_or_else(|| {
                        format!(
                            "usage: cargo xtask selfcheck [measure] [checks]; no self-check {kind}"
                        )
                    })
            })
            .collect::<Result<_, _>>()?
    };
    let folder = root.join("target").join("selfcheck");
    fs::create_dir_all(&folder).map_err(|error| error.to_string())?;
    let sha = working_commit(root, &folder)?;
    println!("Self-checking the tools at {sha}.");
    let mut differed = Vec::new();
    for kind in kinds {
        let found = match kind {
            "tooling" => {
                xtask::tooling_check::run(root)?;
                Vec::new()
            }
            "measure" => measure_aa(root, &sha, &folder.join("measure"))?,
            _ => checks_aa(root, &sha, &folder.join("checks"))?,
        };
        if found.is_empty() {
            println!("selfcheck {kind}: the two sides came out equal.");
        }
        differed.extend(
            found
                .into_iter()
                .map(|problem| format!("{kind}: {problem}")),
        );
    }
    if differed.is_empty() {
        return Ok(());
    }
    Err(Failure::found(format!(
        "the self-checks found the tools' A/A sides differ:\n  {}",
        differed.join("\n  ")
    )))
}

/// The working tree as a commit: HEAD when nothing changed, else a commit
/// of every file git would add over HEAD, made through a scratch index in
/// `folder` and given HEAD's author, committer and dates, so the same
/// content makes the same commit. No ref moves.
fn working_commit(root: &Path, folder: &Path) -> Result<String, String> {
    let head = git(root, &["rev-parse", "HEAD"])?;
    let index = folder.join("index");
    let with_index = |arguments: &[&str]| -> Result<String, String> {
        let output = Command::new("git")
            .args(arguments)
            .env("GIT_INDEX_FILE", &index)
            .current_dir(root)
            .output()
            .map_err(|error| format!("cannot run git: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "git {} failed: {}",
                arguments.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    with_index(&["read-tree", "HEAD"])?;
    with_index(&["add", "--all"])?;
    let tree = with_index(&["write-tree"])?;
    if tree == git(root, &["rev-parse", "HEAD^{tree}"])? {
        return Ok(head);
    }
    let identity = git(
        root,
        &["log", "-1", "--format=%an%n%ae%n%aI%n%cn%n%ce%n%cI", "HEAD"],
    )?;
    let fields: Vec<&str> = identity.lines().collect();
    let [
        author,
        author_email,
        author_date,
        committer,
        committer_email,
        committer_date,
    ] = fields[..]
    else {
        return Err(format!("HEAD's identity reads oddly: {identity}"));
    };
    let output = Command::new("git")
        .args([
            "commit-tree",
            &tree,
            "-p",
            &head,
            "-m",
            "cargo xtask selfcheck: the working tree",
        ])
        .envs([
            ("GIT_AUTHOR_NAME", author),
            ("GIT_AUTHOR_EMAIL", author_email),
            ("GIT_AUTHOR_DATE", author_date),
            ("GIT_COMMITTER_NAME", committer),
            ("GIT_COMMITTER_EMAIL", committer_email),
            ("GIT_COMMITTER_DATE", committer_date),
        ])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git commit-tree failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// The measure task of an A/A comparison of `sha` and its machine profile:
/// this machine's discrete GPU (or its only one) through the platform's
/// API, as the local setup measurement runs.
fn measure_task(sha: &str) -> (Value, Value) {
    let api = if cfg!(windows) {
        "d3d12"
    } else if cfg!(target_os = "macos") {
        "metal"
    } else {
        "vulkan"
    };
    let setup = json!({"id": "selfcheck", "runtime": "native", "api": api, "gpu": "selfcheck"});
    let profile = json!({
        "machine": {"id": "selfcheck", "gpus": [{"model": "selfcheck", "type": "discrete"}]},
        "setups": [setup],
    });
    let arm = json!({"sha": sha, "options": {}});
    let task = json!({
            "version": 1, "operation": "measure", "environment": profile,
            "tools_sha": sha,
            "base_sha": sha,
            "sha": sha,
            "setup": setup,
            "pairs": PAIRS,
            "want": {"profiles": false, "shader_report": false},
            "checks": [],
            "workloads": [{
                "name": "field-90k",
                "feature": "none",
                "arms": {"base": arm, "candidate": arm},
                "workload": {
                    "scene": "field-90k",
                    "settings": "default",
                    "runtimes": {"native": {"args": [
                        "--scene", "bench/scenes/field-90k.json",
                        "--settings", "bench/settings/default.json",
                    ]}},
                    "settle": {
                        "stats": {"loading": false, "pipelines_compiling": false},
                        "quiet_frames": 3,
                        "min_ms": 2000,
                        "max_ms": 120_000,
                    },
                    "sample": {"warmup_frames": 60, "frames": 300, "paced": false},
                    "capture": {"timestamp_ms": 10_000, "overlays": false},
                },
            }],
    });
    (task, profile)
}

/// Writes `value` as JSON into `path`.
fn write(path: &Path, value: &Value) -> Result<PathBuf, String> {
    fs::write(
        path,
        serde_json::to_string_pretty(value).expect("JSON serializes") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(path.to_path_buf())
}

/// Runs the A/A measure task of `sha` into `folder`: what differed.
fn measure_aa(root: &Path, sha: &str, folder: &Path) -> Result<Vec<String>, String> {
    fs::create_dir_all(folder).map_err(|error| error.to_string())?;
    let (task, _profile) = measure_task(sha);
    let task = write(&folder.join("task.json"), &task)?;
    let out = folder.join("report.json");
    let path = |path: &Path| path.display().to_string();
    let arguments = ["--request", &path(&task), "--out", &path(&out)].map(str::to_owned);
    crate::measure::measure_with(root, &arguments, Some(sha))?;
    let report = read_json(&out)?;
    let mut differed = Vec::new();
    let mut unjudged = None;
    if report["incomplete"] == true {
        differed.push("the task did not complete".to_owned());
    }
    for workload in report["workloads"].as_array().into_iter().flatten() {
        let name = workload["name"].as_str().unwrap_or_default();
        if workload["settle"] != "ok" {
            differed.push(format!("{name}: {}", workload["settle"]));
        }
        let runs = workload["runs"].as_array().map_or(&[][..], Vec::as_slice);
        if runs.len() as u64 != 2 * PAIRS {
            differed.push(format!("{name}: {} runs, not {}", runs.len(), 2 * PAIRS));
        }
        for unavailable in workload["unavailable"].as_array().into_iter().flatten() {
            if unavailable["metric"] == "*" {
                differed.push(format!("{name}: a run failed: {}", unavailable["reason"]));
            }
        }
        // The same commit draws the same work every run.
        if let Some(first) = runs.first() {
            for run in runs.iter().skip(1) {
                let changed: Vec<String> = first["work"]
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter(|(counter, value)| run["work"][counter.as_str()] != **value)
                    .map(|(counter, value)| {
                        format!(
                            "{counter} {value} against {}",
                            run["work"][counter.as_str()]
                        )
                    })
                    .collect();
                if !changed.is_empty()
                    || run["work"].as_object().map(serde_json::Map::len)
                        != first["work"].as_object().map(serde_json::Map::len)
                {
                    differed.push(format!(
                        "{name}: the {} run's work differs from the first run's: {}",
                        run["arm"].as_str().unwrap_or_default(),
                        changed.join(", ")
                    ));
                }
            }
        }
        let median = |arm: &str| {
            let mut times: Vec<f64> = runs
                .iter()
                .filter(|run| run["arm"] == arm)
                .filter_map(|run| run["metrics"][FRAME_TIME].as_f64())
                .collect();
            times.sort_by(f64::total_cmp);
            let middle = times.len() / 2;
            match times.len() {
                0 => None,
                count if count % 2 == 1 => Some(times[middle]),
                _ => Some(f64::midpoint(times[middle - 1], times[middle])),
            }
        };
        let busiest = runs
            .iter()
            .filter_map(|run| run["conditions"]["foreign_cpu_load_pct"].as_f64())
            .fold(0.0, f64::max);
        match (median("base"), median("candidate")) {
            (Some(base), Some(candidate)) if base > 0.0 && candidate > 0.0 => {
                let ratio = candidate / base;
                println!(
                    "selfcheck measure: {name} {FRAME_TIME} base {base:.3} ms, candidate {candidate:.3} ms (ratio {ratio:.3})."
                );
                if busiest > BUSY_CPU_PCT {
                    unjudged = Some(format!(
                        "{name}: other programs used up to {busiest:.0}% of the processors while it measured (more than {BUSY_CPU_PCT}%), so its frame times say nothing: run it on a quiet machine"
                    ));
                } else if ratio > TIME_RATIO || ratio < 1.0 / TIME_RATIO {
                    differed.push(format!(
                        "{name}: the arms' median frame times differ by {ratio:.3} ({base:.3} against {candidate:.3} ms)"
                    ));
                }
            }
            _ => differed.push(format!("{name}: the runs report no {FRAME_TIME}")),
        }
    }
    match unjudged {
        Some(why) if differed.is_empty() => Err(why),
        _ => Ok(differed),
    }
}

/// Runs the A/A checks of `sha` into `folder`: what differed.
fn checks_aa(root: &Path, sha: &str, folder: &Path) -> Result<Vec<String>, String> {
    let checks = [
        json!({"name": CHECK, "sha": sha, "expect": "pass"}),
        json!({"name": CHECK, "sha": sha, "scripts_sha": sha, "expect": "pass"}),
        json!({"name": CHECK, "sha": sha, "expect": "fail"}),
    ];
    let (entries, _) = crate::checks::run(root, &checks, folder, None)?;
    write(&folder.join("report.json"), &json!({"checks": entries}))?;
    let exits = |entry: &Value| -> Vec<i64> {
        entry["attempts"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|attempt| attempt["exit_code"].as_i64())
            .collect()
    };
    let mut differed = Vec::new();
    for (entry, scripts) in entries
        .iter()
        .take(2)
        .zip(["its own", "the same commit's checked out again"])
    {
        if exits(entry) != [0] {
            differed.push(format!(
                "{CHECK} with {scripts} scripts exited {:?}, not 0 at its first attempt",
                exits(entry)
            ));
        }
    }
    if let Some(expecting_failure) = entries.get(2) {
        let attempts = exits(expecting_failure);
        if attempts.len() != 3 || attempts.iter().any(|code| *code != 0) {
            differed.push(format!(
                "{CHECK} expected to fail ran {attempts:?}: it passes, so it must fail every attempt"
            ));
        }
    }
    Ok(differed)
}
