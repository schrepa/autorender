use std::path::Path;
use std::process::Command;

use serde_json::{Map, Value, json};

const GATE: &str = "bench/gate.json";
const ROW_SHARE: f64 = 1.3;
const ROW_SLACK_MS: f64 = 0.03;
const ROW_SLACK_ALLOCATIONS: f64 = 2.0;
const SPEED_MOVE: f64 = 1.2;
const DRAW_ALLOCATIONS: f64 = 0.5;

pub fn gate(root: &Path, options: &[String]) -> Result<(), String> {
    let path = root.join(GATE);
    let text = std::fs::read_to_string(&path).map_err(|error| format!("{GATE}: {error}"))?;
    let mut gate: Value =
        serde_json::from_str(&text).map_err(|error| format!("{GATE}: {error}"))?;
    match options.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        ["flight"] => flight(&report(root, &gate, "flight")?, &gate["flight_limits"]),
        ["still"] => still(&report(root, &gate, "still")?, &gate["still_rows"]),
        ["still", "--write"] => {
            let report = report(root, &gate, "still")?;
            moved(&report).map_or(Ok(()), Err)?;
            gate["still_rows"] = rows(&report).into();
            let text = serde_json::to_string_pretty(&gate).map_err(|error| error.to_string())?;
            std::fs::write(&path, text + "\n").map_err(|error| format!("{GATE}: {error}"))
        }
        _ => Err("usage: cargo xtask gate flight | still [--write]".into()),
    }
}

fn report(root: &Path, gate: &Value, run: &str) -> Result<Value, String> {
    let package = gate["package"].as_str().ok_or("gate.json: package")?;
    let bin = gate["bin"].as_str().ok_or("gate.json: bin")?;
    crate::process::run(
        root,
        "cargo",
        &[
            "build",
            "-p",
            package,
            "--bin",
            bin,
            "--profile",
            "quick",
            "--features",
            "profiling",
        ],
    )?;
    let arguments: Vec<&str> = gate[run]
        .as_array()
        .ok_or_else(|| format!("gate.json: {run}"))?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let program = root.join("target/quick").join(bin);
    let output = Command::new(&program)
        .args(&arguments)
        .current_dir(root)
        .output()
        .map_err(|error| format!("{}: {error}", program.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{bin} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| format!("{bin} report: {error}"))
}

fn moved(report: &Value) -> Option<String> {
    let [fast, slow] = ["fast_ms", "slow_ms"].map(|side| report["speed"][side].as_f64());
    let (fast, slow) = fast.zip(slow)?;
    (slow > fast * SPEED_MOVE).then(|| {
        format!("the CPU's speed moved during the run ({fast:.3} to {slow:.3} ms): not compared")
    })
}

fn tallied(report: &Value, name: &str) -> f64 {
    report["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|row| row["name"] == name)
        .and_then(|row| row["runs"].as_f64())
        .unwrap_or(0.0)
}

fn flight(report: &Value, limits: &Value) -> Result<(), String> {
    let measured = &report["flight"];
    let holes = tallied(report, "water.uncovered_frames");
    let rows = [
        (
            "over_33_per_minute",
            measured["over_33_per_minute"].as_f64(),
        ),
        ("worst_ms", measured["worst_ms"].as_f64()),
        (
            "resolution_steps_per_minute",
            measured["resolution_steps_per_minute"].as_f64(),
        ),
        ("water_uncovered_frames", Some(holes)),
    ];
    let mut failed = Vec::new();
    for (name, value) in rows {
        let (value, limit) = (value.unwrap_or(f64::INFINITY), limits[name].as_f64());
        println!(
            "{}",
            json!({ "check": name, "value": value, "limit": limit })
        );
        if limit.is_some_and(|limit| value > limit) {
            failed.push(name);
        }
    }
    println!(
        "{}",
        json!({
            "median_ms": measured["median_ms"],
            "frames": measured["frames"],
            "group_rebuilds": tallied(report, "render.group_rebuilds"),
            "buffer_recreations": tallied(report, "render.buffer_recreations"),
            "texture_recreations": tallied(report, "render.texture_recreations"),
            "draw_compiles": tallied(report, "render.draw_compiles"),
            "wet_tile_waits": tallied(report, "water.wet_tile_waits"),
            "speed": report["speed"],
            "waits": report["waits"],
        })
    );
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!("flight over its limits: {}", failed.join(", ")))
    }
}

fn rows(report: &Value) -> Map<String, Value> {
    let frames = report["rows_frames"].as_f64().unwrap_or(1.0).max(1.0);
    report["rows"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|row| row["ms"].as_f64().is_some_and(|ms| ms > 0.0))
        .filter_map(|row| {
            let ms = row["ms"].as_f64()? / frames;
            let allocations = row["allocations"].as_f64()? / frames;
            Some((
                row["name"].as_str()?.to_owned(),
                json!({ "ms": ms, "allocations": allocations }),
            ))
        })
        .collect()
}

fn tallied_draws(report: &Value) -> f64 {
    let frames = report["rows_frames"].as_f64().unwrap_or(1.0).max(1.0);
    tallied(report, "render.draw") / frames
}

fn still(report: &Value, budget: &Value) -> Result<(), String> {
    if let Some(moved) = moved(report) {
        return Err(moved);
    }
    let mut failed = Vec::new();
    let rows = rows(report);
    for (name, row) in &rows {
        let ms = row["ms"].as_f64().unwrap_or(0.0);
        let allocations = row["allocations"].as_f64().unwrap_or(0.0);
        let held = &budget[name];
        let (held_ms, held_allocations) = (held["ms"].as_f64(), held["allocations"].as_f64());
        println!(
            "{}",
            json!({ "row": name, "ms": ms, "allocations": allocations, "budget": held })
        );
        if name == "render.draw" && allocations > DRAW_ALLOCATIONS * tallied_draws(report) {
            failed.push(format!("{name}: {allocations:.1} allocations per frame"));
        }
        if held_ms.is_some_and(|held| ms > held * ROW_SHARE + ROW_SLACK_MS) {
            failed.push(format!("{name}: {ms:.3} ms"));
        }
        if held_allocations.is_some_and(|held| allocations > held + ROW_SLACK_ALLOCATIONS) {
            failed.push(format!("{name}: {allocations:.1} allocations"));
        }
        if held.is_null() && (ms > ROW_SLACK_MS || allocations > ROW_SLACK_ALLOCATIONS) {
            failed.push(format!("{name}: new row"));
        }
    }
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!("still budget: {}", failed.join("; ")))
    }
}
