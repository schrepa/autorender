//! Absolute budgets (`bench/budgets.json`): for each bench scene and device
//! class, a soft and a hard limit on each budgeted metric (a GPU pass
//! group's milliseconds, a memory pool's bytes, a count of draws,
//! instances or triangles), the value each came from and where that was
//! measured, and waivers as data. `cargo xtask bench budgets` judges a
//! measure report's runs against them, or writes them from one; lint's
//! bench stage holds the file to the scenes, the device classes and the
//! metric catalog.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::{Map, Value, json};

use crate::measure::read_json;

/// The device classes budgets name: machine profiles' GPU types.
const DEVICES: &[&str] = &["integrated", "discrete"];

/// The GPU's timings budgets are written for: the frame's, and each pass
/// group's (`gpu:stage_*`), but for the spans between and around them.
const FRAME_GPU: &str = "gpu:frame_gpu";
const STAGES: &str = "gpu:stage_";
const NO_PASSES: &[&str] = &["gpu:stage_gaps", "gpu:stage_lead", "gpu:stage_tail"];
/// Pass groups taking less than this many milliseconds get no budget: their
/// timings are mostly noise.
const SMALLEST_STAGE_MS: f64 = 0.05;
/// The memory pools and counts budgets are written for.
const AMOUNTS: &[&str] = &[
    "resident_mesh_bytes",
    "resident_texture_bytes",
    "mem:peak_requested_bytes",
    "draw_calls",
    "instances",
    "visible_instances",
    "triangles",
    "shadows.draws",
    "transparent_draws",
];
/// The soft and hard limits over a measured timing: times vary from run to
/// run and with the machine's load, so their limits sit a quarter and a
/// half above it, and at least 0.1 and 0.2 ms.
const TIME_HEADROOM: (f64, f64) = (1.25, 1.5);
const TIME_HEADROOM_MS: (f64, f64) = (0.1, 0.2);
/// The soft and hard limits over a measured amount: a scene's settled
/// view draws and holds about the same each run, so a tenth and a quarter.
const AMOUNT_HEADROOM: (f64, f64) = (1.1, 1.25);

const USAGE: &str = "usage: cargo xtask bench budgets --report REPORT.json --device integrated|discrete [--arm base|candidate] [--today YYYY-MM-DD] [--write]";

/// Reads `bench/budgets.json`.
fn read(root: &Path) -> Result<Value, String> {
    read_json(&root.join("bench/budgets.json"))
}

/// A date `YYYY-MM-DD`, as waivers end.
fn is_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() == 10
        && bytes.iter().enumerate().all(|(index, byte)| match index {
            4 | 7 => *byte == b'-',
            _ => byte.is_ascii_digit(),
        })
}

/// Problems with `bench/budgets.json`: scenes `bench/scenes` lacks, device
/// classes but integrated and discrete, budgets without their source,
/// metrics the catalog (`covered`, whose entries cover names or prefixes)
/// lacks, limits that are not numbers with the measured value at most soft
/// at most hard, and waivers naming no budget or lacking a reason or an
/// end.
pub fn problems(root: &Path, covered: &dyn Fn(&str) -> bool) -> Result<Vec<String>, String> {
    let budgets = read(root)?;
    let mut problems = Vec::new();
    if budgets["note"].as_str().is_none_or(str::is_empty) {
        problems.push("bench/budgets.json: a note says what the budgets are".into());
    }
    let Some(scenes) = budgets["scenes"].as_object() else {
        return Ok(vec![
            "bench/budgets.json: scenes is an object of scenes".into(),
        ]);
    };
    for (scene, devices) in scenes {
        if !root
            .join("bench/scenes")
            .join(format!("{scene}.json"))
            .is_file()
        {
            problems.push(format!(
                "bench/budgets.json: scene {scene} is no bench/scenes file"
            ));
        }
        for (device, budget) in devices.as_object().into_iter().flatten() {
            let at = format!("bench/budgets.json: {scene} {device}");
            if !DEVICES.contains(&device.as_str()) {
                problems.push(format!("{at}: devices are integrated or discrete"));
            }
            if budget["source"].as_str().is_none_or(str::is_empty) {
                problems.push(format!(
                    "{at}: source says where its measured values came from"
                ));
            }
            let Some(metrics) = budget["metrics"].as_object() else {
                problems.push(format!("{at}: metrics is an object of limits"));
                continue;
            };
            for (metric, limits) in metrics {
                let at = format!("{at} {metric}");
                if !covered(metric) {
                    problems.push(format!("{at}: no metric of bench/metrics.json"));
                }
                let [measured, soft, hard] =
                    ["measured", "soft", "hard"].map(|field| limits[field].as_f64());
                match (measured, soft, hard) {
                    (Some(measured), Some(soft), Some(hard))
                        if measured >= 0.0 && measured <= soft && soft <= hard => {}
                    _ => problems.push(format!(
                        "{at}: measured, soft and hard are numbers, each at most the next"
                    )),
                }
            }
        }
    }
    for (index, waiver) in budgets["waivers"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        let at = format!("bench/budgets.json waivers[{index}]");
        let named = |field: &str| waiver[field].as_str().unwrap_or_default().to_owned();
        let (scene, device, metric) = (named("scene"), named("device"), named("metric"));
        if budgets["scenes"][&scene][&device]["metrics"][&metric].is_null() {
            problems.push(format!("{at}: names no budget ({scene} {device} {metric})"));
        }
        if waiver["reason"].as_str().is_none_or(str::is_empty) {
            problems.push(format!("{at}: a reason"));
        }
        if !waiver["until"].as_str().is_some_and(is_date) {
            problems.push(format!("{at}: until is the date it ends, YYYY-MM-DD"));
        }
    }
    Ok(problems)
}

/// The median of `values`, which must not be empty.
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    }
}

/// Today's date in UTC, `YYYY-MM-DD`, from the system clock.
fn today() -> String {
    crate::measure::iso(std::time::SystemTime::now())[..10].to_owned()
}

/// `value` rounded up to a multiple of `step`, without the step's float
/// error.
fn up_to(value: f64, step: f64) -> f64 {
    ((value / step - 1e-9).ceil() * step * 1000.0).round() / 1000.0
}

/// The value of `metric` in `run`: its timings and memory, or its work
/// counters.
fn value(run: &Value, metric: &str) -> Option<f64> {
    run["metrics"][metric]
        .as_f64()
        .or_else(|| run["work"][metric].as_f64())
}

/// A workload's settled runs of `arm`.
fn settled<'a>(workload: &'a Value, arm: &str) -> Vec<&'a Value> {
    workload["runs"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|run| run["arm"] == arm && run["conditions"]["settled"]["held"] == true)
        .collect()
}

/// The metrics `runs` measured that budgets are written for.
fn budgeted(runs: &[&Value]) -> Vec<String> {
    let mut metrics: Vec<String> = runs
        .iter()
        .flat_map(|run| run["metrics"].as_object().into_iter().flatten())
        .map(|(metric, _)| metric)
        .filter(|metric| {
            *metric == FRAME_GPU
                || (metric.starts_with(STAGES) && !NO_PASSES.contains(&metric.as_str()))
        })
        .cloned()
        .collect();
    metrics.extend(AMOUNTS.iter().map(|&metric| metric.to_owned()));
    metrics.sort();
    metrics.dedup();
    metrics
}

/// Where `runs` of `arm` in `report`'s `workload` were measured, in words.
fn source(report: &Value, workload: &Value, arm: &str, runs: &[&Value]) -> String {
    let setup = &report["setup"];
    let gpu = report["profile_start"]["setups"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|known| known["id"] == *setup)
        .and_then(|known| known["gpu"].as_str())
        .or_else(|| setup.as_str())
        .unwrap_or("an unnamed setup");
    let size = &workload["method"]["size"];
    let dates: Vec<&str> = runs
        .iter()
        .filter_map(|run| run["started_at"].as_str().map(|at| &at[..at.len().min(10)]))
        .collect();
    let load = |field: &str| {
        let values: Vec<f64> = runs
            .iter()
            .filter_map(|run| run["conditions"][field].as_f64())
            .collect();
        match (
            values.iter().copied().reduce(f64::min),
            values.iter().copied().reduce(f64::max),
        ) {
            (Some(low), Some(high)) => format!("{low:.0}-{high:.0}%"),
            _ => "unknown".to_owned(),
        }
    };
    format!(
        "{count} settled {arm} runs of {sha} ({name}) on {gpu}, {runtime} {width}x{height}, {first} to {last}; other processes' load CPU {cpu}, GPU {gpu_load}",
        count = runs.len(),
        sha = report[format!("{arm}_sha")]
            .as_str()
            .unwrap_or("an unnamed commit"),
        name = workload["name"].as_str().unwrap_or_default(),
        runtime = workload["method"]["runtime"].as_str().unwrap_or("native"),
        width = size[0],
        height = size[1],
        first = dates.iter().min().copied().unwrap_or("?"),
        last = dates.iter().max().copied().unwrap_or("?"),
        cpu = load("foreign_cpu_load_pct"),
        gpu_load = load("foreign_gpu_load_pct"),
    )
}

/// The budget `runs` of `arm` in `report`'s `workload` call for: each
/// budgeted metric's median with its headroom.
fn written(report: &Value, workload: &Value, arm: &str, runs: &[&Value]) -> Value {
    let mut metrics = Map::new();
    for metric in budgeted(runs) {
        let values: Vec<f64> = runs.iter().filter_map(|run| value(run, &metric)).collect();
        if values.is_empty() {
            continue;
        }
        let measured = median(values);
        let limits = if metric == FRAME_GPU || metric.starts_with(STAGES) {
            if measured < SMALLEST_STAGE_MS {
                continue;
            }
            let limit =
                |(scale, least): (f64, f64)| up_to((measured * scale).max(measured + least), 0.05);
            json!({
                "measured": (measured * 1000.0).round() / 1000.0,
                "soft": limit((TIME_HEADROOM.0, TIME_HEADROOM_MS.0)),
                "hard": limit((TIME_HEADROOM.1, TIME_HEADROOM_MS.1)),
            })
        } else {
            if measured <= 0.0 {
                continue;
            }
            json!({
                "measured": measured.round() as u64,
                "soft": (measured * AMOUNT_HEADROOM.0).ceil() as u64,
                "hard": (measured * AMOUNT_HEADROOM.1).ceil() as u64,
            })
        };
        metrics.insert(metric, limits);
    }
    json!({ "source": source(report, workload, arm, runs), "metrics": metrics })
}

/// Judges a measure report against the budgets, or with `--write` writes
/// the budgets of the scenes it measured from it: for each workload named
/// after a bench scene, the median of each budgeted metric over the chosen
/// arm's settled runs against (or setting) the device class's limits.
/// Judging prints every verdict as JSON and fails when a hard limit is
/// broken and no waiver in force excuses it.
pub fn judge(root: &Path, options: &[String]) -> Result<(), crate::Failure> {
    let mut named = BTreeMap::new();
    let mut write = false;
    let mut given = options.iter();
    while let Some(option) = given.next() {
        match option.as_str() {
            "--write" => write = true,
            flag if flag.starts_with("--") => {
                let value = given.next().ok_or(USAGE)?;
                named.insert(flag[2..].to_owned(), value.clone());
            }
            _ => return Err(USAGE.into()),
        }
    }
    let (Some(report), Some(device)) = (named.get("report"), named.get("device")) else {
        return Err(USAGE.into());
    };
    let arm = named.get("arm").map_or("candidate", String::as_str);
    if !DEVICES.contains(&device.as_str()) || !["base", "candidate"].contains(&arm) {
        return Err(USAGE.into());
    }
    let today = named.get("today").cloned().unwrap_or_else(today);
    let mut budgets = read(root)?;
    let report = read_json(Path::new(report))?;
    let scenes: Vec<(&str, &Value)> = report["workloads"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|workload| Some((workload["name"].as_str()?, workload)))
        .filter(|(scene, _)| {
            root.join("bench/scenes")
                .join(format!("{scene}.json"))
                .is_file()
        })
        .collect();
    if write {
        let mut wrote = Vec::new();
        for (scene, workload) in scenes {
            let runs = settled(workload, arm);
            if runs.is_empty() {
                continue;
            }
            budgets["scenes"][scene][device.as_str()] = written(&report, workload, arm, &runs);
            wrote.push(scene);
        }
        if wrote.is_empty() {
            return Err("the report has no settled runs of a bench scene: name workloads after bench/scenes files".into());
        }
        let text = serde_json::to_string_pretty(&budgets).expect("budgets serialize") + "\n";
        fs::write(root.join("bench/budgets.json"), text)
            .map_err(|error| format!("cannot write bench/budgets.json: {error}"))?;
        println!(
            "wrote the {device} budgets of {} from {arm} runs into bench/budgets.json",
            wrote.join(", ")
        );
        return Ok(());
    }
    let waived = |scene: &str, metric: &str| {
        budgets["waivers"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|waiver| {
                waiver["scene"] == scene
                    && waiver["device"] == device.as_str()
                    && waiver["metric"] == metric
                    && waiver["until"]
                        .as_str()
                        .is_some_and(|until| until >= today.as_str())
            })
    };
    let mut verdicts = Vec::new();
    let mut broken = 0;
    for (scene, workload) in scenes {
        let Some(limits) = budgets["scenes"][scene][device.as_str()]["metrics"].as_object() else {
            continue;
        };
        let runs = settled(workload, arm);
        for (metric, limit) in limits {
            let values: Vec<f64> = runs.iter().filter_map(|run| value(run, metric)).collect();
            let (soft, hard) = (limit["soft"].as_f64(), limit["hard"].as_f64());
            let (verdict, measured) = match (values.is_empty(), soft, hard) {
                (false, Some(soft), Some(hard)) => {
                    let measured = median(values);
                    let verdict = if measured <= soft {
                        "within"
                    } else if measured <= hard {
                        "soft"
                    } else if waived(scene, metric).is_some() {
                        "waived"
                    } else {
                        broken += 1;
                        "hard"
                    };
                    (verdict, Some(measured))
                }
                _ => ("unmeasured", None),
            };
            verdicts.push(json!({
                "scene": scene, "device": device, "metric": metric, "value": measured,
                "soft": soft, "hard": hard, "verdict": verdict,
                "waiver": waived(scene, metric).map(|waiver| waiver["reason"].clone()),
            }));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"arm": arm, "device": device, "verdicts": verdicts}))
            .expect("verdicts serialize")
    );
    if verdicts.is_empty() {
        return Err(format!("the report measured no scene with {device} budgets: name workloads after bench/budgets.json's scenes").into());
    }
    if broken > 0 {
        return Err(crate::Failure::found(format!(
            "{broken} hard budget(s) broken without a waiver"
        )));
    }
    Ok(())
}
