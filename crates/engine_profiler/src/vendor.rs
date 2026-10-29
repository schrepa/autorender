use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use engine::{Error, ErrorCode, Expected};
use serde_json::{Value, json};

use crate::capabilities::installed;

const PIXTOOL: &[&str] = &[r"C:\Program Files\Microsoft PIX\*/pixtool.exe"];
const NGFX: &[&str] = &[
    r"C:\Program Files\NVIDIA Corporation\Nsight Graphics */host/windows-desktop-nomad-x64/ngfx.exe",
];

const PIX_COUNTERS: &[&str] = &[
    "GPU Core Clocks",
    "XVE Active",
    "XVE Stall",
    "XVE Thread Occupancy",
    "Samplers Busy",
    "Samplers Bottleneck",
    "Early Depth Test Fails",
    "Rasterized Pixels",
    "GPU Memory Byte Read",
    "GPU Memory Byte Write",
    "IA Vertices",
    "VS Invocations",
    "PS Invocations",
    "CS Invocations",
    "Clipper Primitives In",
    "Clipper Primitives Out",
];

pub fn pix_counters(capture: &Path, folder: &Path, counters: &[String]) -> Result<Value, Error> {
    let pixtool = installed("pixtool", PIXTOOL).ok_or_else(|| {
        Error::new(ErrorCode::PixUnavailable, "$.pixtool").expected(Expected::Present)
    })?;
    let ran = |error: std::io::Error| {
        Error::new(ErrorCode::PixUnavailable, "$.pixtool").got(error.to_string())
    };
    settled(capture);
    let listed = Command::new(&pixtool)
        .args(["--output=quiet", "open-capture"])
        .arg(capture)
        .arg("list-counters")
        .output()
        .map_err(ran)?;
    let available: Vec<String> = String::from_utf8_lossy(&listed.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect();
    if available.is_empty() {
        return Err(Error::new(ErrorCode::PixUnavailable, "$.counters")
            .expected(Expected::NonEmpty)
            .got(tail(&listed.stdout, &listed.stderr)));
    }
    let wanted: Vec<String> = if counters.is_empty() {
        PIX_COUNTERS
            .iter()
            .filter(|name| available.iter().any(|listed| listed == *name))
            .map(|name| (*name).to_owned())
            .collect()
    } else {
        counters.to_vec()
    };
    std::fs::create_dir_all(folder).map_err(|_| unwritable("$.folder", folder))?;
    let events = folder.join("pix-events.csv");
    let mut command = Command::new(&pixtool);
    command
        .args(["--output=quiet", "open-capture"])
        .arg(capture)
        .arg("save-event-list")
        .arg(&events);
    for name in &wanted {
        command.arg(format!("--counters={}", name.replace([' ', '(', ')'], "*")));
    }
    let saved = command.output().map_err(ran)?;
    let table = std::fs::read_to_string(&events).map_err(|_| {
        Error::new(ErrorCode::PixUnavailable, "$.events")
            .expected(Expected::File)
            .got(tail(&saved.stdout, &saved.stderr))
    })?;
    let mut summary = summarize_events(&table, &wanted);
    summary["tool"] = "pix".into();
    summary["capture"] = capture.display().to_string().into();
    summary["events"] = events.display().to_string().into();
    summary["counters_available"] = available.len().into();
    summary["open"] = "Open the capture in PIX (Windows Developer Mode replays it): every \
                       command, resource and counter of the frame; `pixtool open-capture FILE \
                       list-counters` lists the counters `--vendor-counters` can name"
        .into();
    write(&folder.join("pix.summary.json"), &summary)?;
    Ok(summary)
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn summarize_events(table: &str, wanted: &[String]) -> Value {
    let mut lines = table.lines();
    let header = lines.next().map(split_csv).unwrap_or_default();
    let columns: Vec<(String, usize)> = wanted
        .iter()
        .filter_map(|name| {
            header
                .iter()
                .position(|column| column == name)
                .map(|index| (name.clone(), index))
        })
        .collect();
    let named = |name: &str| columns.iter().position(|(column, _)| column == name);
    let time = named("GPU Core Clocks");
    let weight = time;
    let shares: Vec<bool> = columns
        .iter()
        .map(|(name, _)| {
            [
                "Active",
                "Stall",
                "Occupancy",
                "Busy",
                "Bottleneck",
                "Utilization",
            ]
            .iter()
            .any(|word| name.contains(word))
        })
        .collect();
    let mut events: BTreeMap<i64, (i64, String, Option<Vec<f64>>)> = BTreeMap::new();
    for line in lines {
        let fields = split_csv(line);
        let (Some(id), Some(parent), Some(name)) = (
            fields.first().and_then(|field| field.parse().ok()),
            fields.get(1).and_then(|field| field.parse().ok()),
            fields.get(2),
        ) else {
            continue;
        };
        let values: Option<Vec<f64>> = columns
            .iter()
            .map(|(_, index)| fields.get(*index).and_then(|field| field.parse().ok()))
            .collect();
        let name = name
            .trim_start_matches("<deprecated - use pix3.h instead> ")
            .to_owned();
        events.insert(id, (parent, name, values));
    }
    let path = |mut id: i64| {
        let mut names = Vec::new();
        while let Some((parent, name, _)) = events.get(&id) {
            names.push(name.clone());
            id = *parent;
        }
        names.reverse();
        names
    };
    let combine = |rows: &[&Vec<f64>]| -> Vec<f64> {
        let weights: Vec<f64> = rows
            .iter()
            .map(|values| weight.map_or(1.0, |weight| values[weight]))
            .collect();
        let total: f64 = weights.iter().sum();
        (0..columns.len())
            .map(|column| {
                let sum: f64 = rows.iter().map(|values| values[column]).sum();
                match (shares[column], total > 0.0) {
                    (false, _) => sum,
                    (true, true) => {
                        rows.iter()
                            .zip(&weights)
                            .map(|(values, weight)| values[column] * weight)
                            .sum::<f64>()
                            / total
                    }
                    (true, false) => sum / rows.len().max(1) as f64,
                }
            })
            .collect()
    };
    let parents: std::collections::BTreeSet<i64> =
        events.values().map(|(parent, _, _)| *parent).collect();
    let mut nodes: BTreeMap<String, Vec<&Vec<f64>>> = BTreeMap::new();
    let mut costliest: Vec<(Vec<String>, &Vec<f64>)> = Vec::new();
    for (&id, (_, _, values)) in &events {
        let Some(values) = values.as_ref().filter(|_| !parents.contains(&id)) else {
            continue;
        };
        let path = path(id);
        let node = if path.len() > 1 {
            path[0].clone()
        } else {
            "(outside the graph's nodes)".into()
        };
        nodes.entry(node).or_default().push(values);
        costliest.push((path, values));
    }
    let named_values = |values: &[f64]| -> serde_json::Map<String, Value> {
        columns
            .iter()
            .zip(values)
            .map(|((name, _), value)| (name.clone(), json!(value)))
            .collect()
    };
    let by_time = |a: &[f64], b: &[f64]| match time {
        Some(time) => b[time].total_cmp(&a[time]),
        None => std::cmp::Ordering::Equal,
    };
    let mut passes: Vec<(String, usize, Vec<f64>)> = nodes
        .into_iter()
        .map(|(node, rows)| (node, rows.len(), combine(&rows)))
        .collect();
    passes.sort_by(|a, b| by_time(&a.2, &b.2));
    costliest.sort_by(|a, b| by_time(a.1, b.1));
    json!({
        "counters": columns.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        "passes": passes.iter().map(|(node, events, values)| json!({
            "pass": node,
            "events": events,
            "total": named_values(values),
        })).collect::<Vec<_>>(),
        "costliest_events": costliest.iter().take(25).map(|(path, values)| json!({
            "event": path.join(" > "),
            "counters": named_values(values),
        })).collect::<Vec<_>>(),
    })
}

fn settled(path: &Path) {
    let size = || std::fs::metadata(path).map_or(0, |metadata| metadata.len());
    let started = std::time::Instant::now();
    let (mut last, mut since) = (size(), std::time::Instant::now());
    while started.elapsed() < std::time::Duration::from_secs(60) {
        std::thread::sleep(std::time::Duration::from_millis(250));
        let now = size();
        if now != last {
            (last, since) = (now, std::time::Instant::now());
        } else if last > 0 && since.elapsed() >= std::time::Duration::from_secs(1) {
            return;
        }
    }
}

fn split_csv(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut characters = line.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                field.push('"');
                characters.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut field).trim().to_owned()),
            _ => field.push(character),
        }
    }
    fields.push(field.trim().to_owned());
    fields
}

pub fn nsight_trace(
    arguments: &[String],
    start_ms: u64,
    frames: u32,
    folder: &Path,
) -> Result<Value, Error> {
    let ngfx = installed("ngfx", NGFX).ok_or_else(|| {
        Error::new(ErrorCode::ProfilingUnavailable, "$.ngfx").expected(Expected::Present)
    })?;
    let program = std::env::current_exe()
        .map_err(|error| Error::new(ErrorCode::SourceIo, "$.program").got(error.to_string()))?;
    let directory = std::env::current_dir()
        .map_err(|error| Error::new(ErrorCode::SourceIo, "$.directory").got(error.to_string()))?;
    std::fs::create_dir_all(folder).map_err(|_| unwritable("$.folder", folder))?;
    let before = files(folder);
    let quoted: Vec<String> = arguments
        .iter()
        .map(|argument| {
            if argument.contains(' ') {
                format!("\"{argument}\"")
            } else {
                argument.clone()
            }
        })
        .collect();
    let output = Command::new(&ngfx)
        .args(["--activity", "GPU Trace Profiler", "--exe"])
        .arg(&program)
        .arg("--dir")
        .arg(&directory)
        .arg(format!("--args={}", quoted.join(" ")))
        .arg("--output-dir")
        .arg(folder)
        .args([
            "--start-after-ms",
            &start_ms.to_string(),
            "--limit-to-frames",
            &frames.to_string(),
            "--auto-export",
            "--real-time-shader-profiler",
        ])
        .output()
        .map_err(|error| {
            Error::new(ErrorCode::ProfilingUnavailable, "$.ngfx").got(error.to_string())
        })?;
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if said.contains("Performance Counters unavailable") {
        return Err(
            Error::new(ErrorCode::ProfilingUnavailable, "$.ngfx.counters")
                .expected(Expected::Supported),
        );
    }
    let written: Vec<String> = files(folder)
        .into_iter()
        .filter(|file| !before.contains(file))
        .map(|file| file.display().to_string())
        .collect();
    if written.is_empty() {
        return Err(Error::new(ErrorCode::ProfilingUnavailable, "$.trace")
            .expected(Expected::File)
            .got(tail(&output.stdout, &output.stderr)));
    }
    let summary = json!({
        "tool": "nsight",
        "files": written,
        "viewer": "ngfx-ui",
    });
    write(&folder.join("nsight.summary.json"), &summary)?;
    Ok(summary)
}

fn files(folder: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut folders = vec![folder.to_path_buf()];
    while let Some(folder) = folders.pop() {
        for entry in std::fs::read_dir(&folder).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                folders.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found
}

fn tail(stdout: &[u8], stderr: &[u8]) -> String {
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(stdout),
        String::from_utf8_lossy(stderr)
    );
    let lines: Vec<&str> = said
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    lines[lines.len().saturating_sub(4)..].join(" ")
}

fn write(path: &Path, value: &Value) -> Result<(), Error> {
    let text = serde_json::to_string_pretty(value).expect("json_serializes");
    std::fs::write(path, text).map_err(|_| unwritable("$.summary", path))
}

fn unwritable(at: &str, path: &Path) -> Error {
    Error::new(ErrorCode::SourceIo, at)
        .expected(Expected::Writable)
        .got(path.display().to_string())
}
