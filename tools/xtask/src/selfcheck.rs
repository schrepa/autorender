use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};

use crate::Failure;
use crate::measure::{git, read_json};

const PAIRS: u64 = 2;
const TIME_RATIO: f64 = 1.25;
const FRAME_TIME: &str = "wall:frame_ms_mean";
const BUSY_CPU_PCT: f64 = 50.0;
const CHECK: &str = "agent-lifecycle-smoke";
pub(crate) const KINDS: [&str; 2] = ["measure", "checks"];

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

fn first_scene(root: &Path) -> Result<String, String> {
    let folder = root.join("bench").join("scenes");
    let mut scenes: Vec<String> = fs::read_dir(&folder)
        .map_err(|error| format!("cannot read {}: {error}", folder.display()))?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.strip_suffix(".json").map(str::to_owned)
        })
        .collect();
    scenes.sort();
    scenes
        .into_iter()
        .next()
        .ok_or_else(|| "bench/scenes holds no scene".to_owned())
}

fn measure_task(sha: &str, scene: &str) -> (Value, Value) {
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
        "id": 0,
        "spec": {
            "tools_sha": sha,
            "base_sha": sha,
            "sha": sha,
            "setup": "selfcheck",
            "pairs": PAIRS,
            "want": {"profiles": false, "shader_report": false},
            "checks": [],
            "workloads": [{
                "name": scene,
                "feature": "none",
                "arms": {"base": arm, "candidate": arm},
                "workload": {
                    "scene": scene,
                    "settings": "default",
                    "runtimes": {"native": {"args": [
                        "--scene", format!("bench/scenes/{scene}.json"),
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
        },
    });
    (task, profile)
}

fn write(path: &Path, value: &Value) -> Result<PathBuf, String> {
    fs::write(
        path,
        serde_json::to_string_pretty(value).expect("JSON serializes") + "\n",
    )
    .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(path.to_path_buf())
}

fn measure_aa(root: &Path, sha: &str, folder: &Path) -> Result<Vec<String>, String> {
    fs::create_dir_all(folder).map_err(|error| error.to_string())?;
    let (task, profile) = measure_task(sha, &first_scene(root)?);
    let task = write(&folder.join("task.json"), &task)?;
    let profile = write(&folder.join("profile.json"), &profile)?;
    let out = folder.join("report.json");
    let path = |path: &Path| path.display().to_string();
    let arguments = [
        "--task",
        &path(&task),
        "--profile",
        &path(&profile),
        "--out",
        &path(&out),
    ]
    .map(str::to_owned);
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
