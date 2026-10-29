use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::time::Instant;

use engine::host::HostSettings;
use engine::{Engine, Error, ErrorCode, Expected, Game};
use serde_json::{Map, Value, json};

use crate::Offscreen;
use crate::jobs::JobOptions;

#[derive(Clone, Debug)]
pub struct Method {
    pub settle: Settle,
    pub warmup_frames: u32,
    pub frames: u32,
    pub frames_in_flight: u32,
    pub capture: Option<(f64, PathBuf)>,
}

#[derive(Clone, Debug)]
pub struct Settle {
    pub stats: Map<String, Value>,
    pub quiet_frames: u32,
    pub below: BTreeMap<String, f64>,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl Method {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let method: Value = engine::bricks::parse_json(text)?;
        let number = |value: &Value, path: &str| {
            value
                .as_f64()
                .filter(|number| number.is_finite() && *number >= 0.0)
                .ok_or_else(|| {
                    Error::new(ErrorCode::InvalidValue, format!("$.{path}"))
                        .expected(Expected::Number)
                        .at_least(0)
                        .got(value)
                })
        };
        let count = |value: &Value, path: &str| {
            value
                .as_u64()
                .and_then(|count| u32::try_from(count).ok())
                .ok_or_else(|| {
                    Error::new(ErrorCode::InvalidValue, format!("$.{path}"))
                        .expected(Expected::Integer)
                        .range(0.0, f64::from(u32::MAX))
                        .got(value)
                })
        };
        let settle = &method["settle"];
        let stats = match &settle["stats"] {
            Value::Object(stats) => stats.clone(),
            Value::Null => Map::new(),
            stats @ (Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_)) => {
                return Err(Error::new(ErrorCode::InvalidType, "$.settle.stats")
                    .expected(Expected::Object)
                    .got(stats));
            }
        };
        let mut below = BTreeMap::new();
        for (metric, threshold) in settle["metrics"].as_object().into_iter().flatten() {
            if !metric.starts_with("cpu:") {
                return Err(
                    Error::new(ErrorCode::InvalidValue, format!("$.settle.metrics.{metric}"))
                        .expected(Expected::Supported)
                        .got(metric),
                );
            }
            below.insert(
                metric.clone(),
                number(
                    &threshold["below"],
                    &format!("settle.metrics.{metric}.below"),
                )?,
            );
        }
        let sample = &method["sample"];
        if sample["paced"] != false {
            return Err(Error::new(ErrorCode::InvalidValue, "$.sample.paced")
                .got(&sample["paced"])
                .nearest(false));
        }
        let capture = match &method["capture"] {
            Value::Null => None,
            capture @ (Value::Bool(_)
            | Value::Number(_)
            | Value::String(_)
            | Value::Array(_)
            | Value::Object(_)) => {
                if capture["overlays"] == true {
                    return Err(Error::new(ErrorCode::InvalidValue, "$.capture.overlays")
                        .got(true)
                        .nearest(false));
                }
                let path = capture["path"].as_str().ok_or_else(|| {
                    Error::new(ErrorCode::InvalidType, "$.capture.path")
                        .expected(Expected::Path)
                        .got(&capture["path"])
                })?;
                Some((
                    number(&capture["timestamp_ms"], "capture.timestamp_ms")?,
                    PathBuf::from(path),
                ))
            }
        };
        let frames = count(&sample["frames"], "sample.frames")?;
        let frames_in_flight = count(&method["frames_in_flight"], "frames_in_flight")?;
        if frames == 0 || frames_in_flight == 0 {
            let zero = if frames == 0 {
                "$.sample.frames"
            } else {
                "$.frames_in_flight"
            };
            return Err(Error::new(ErrorCode::OutOfRange, zero).at_least(1).got(0));
        }
        Ok(Self {
            settle: Settle {
                stats,
                quiet_frames: count(&settle["quiet_frames"], "settle.quiet_frames")?,
                below,
                min_ms: number(&settle["min_ms"], "settle.min_ms")?,
                max_ms: number(&settle["max_ms"], "settle.max_ms")?,
            },
            warmup_frames: count(&sample["warmup_frames"], "sample.warmup_frames")?,
            frames,
            frames_in_flight,
            capture,
        })
    }
}

fn at(frame: u32) -> f64 {
    f64::from(frame) * 1000.0 / 60.0
}

fn mark(name: &str) {
    println!("{}", json!({ "mark": name }));
}

pub fn run<G: Game, E: Into<engine::Error>>(
    method: &Method,
    size: (u32, u32),
    gpu: engine::GpuPreference,
    jobs: JobOptions,
    settings: HostSettings,
    build: impl FnOnce() -> Result<(Engine, G), E>,
) -> engine::Result<()> {
    let launched = Instant::now();
    let mut game = Offscreen::launch(size, gpu, &jobs, build)?;
    settings.apply(game.host_mut());
    game.prepare(0.0)?;
    let prepare_ms = launched.elapsed().as_secs_f64() * 1000.0;
    let (held, after_ms) = settle(&mut game, &method.settle, launched)?;
    game.host_mut().suspend();
    let mut frame = 0;
    for _ in 0..method.warmup_frames {
        game.frame(at(frame))?;
        frame += 1;
    }
    let mut times = Vec::with_capacity(method.frames as usize);
    let mut costs = Vec::with_capacity(method.frames as usize);
    let (mut loading, mut compiling) = (false, false);
    mark("serialized-start");
    for _ in 0..method.frames {
        let started = Instant::now();
        game.frame(at(frame))?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        let stats = game.host().frame_stats();
        loading |= stats.loading;
        compiling |= stats.pipelines_compiling;
        costs.extend(stats.bricks_cost);
        frame += 1;
    }
    mark("serialized-end");
    let mut work: Value =
        serde_json::from_str(&game.host().stats()).expect("stats_json");
    work["loading"] = loading.into();
    work["pipelines_compiling"] = compiling.into();
    let capture = match &method.capture {
        Some((timestamp, path)) => {
            if *timestamp < at(frame) {
                return Err(
                    Error::new(ErrorCode::InvalidMeasurement, "$.capture.timestamp_ms")
                        .got(*timestamp)
                        .nearest(at(frame)),
                );
            }
            while at(frame + 1) <= *timestamp {
                game.frame(at(frame))?;
                frame += 1;
            }
            game.frame(*timestamp)?;
            frame += 1;
            game.save_png(path)?;
            Some(path.display().to_string())
        }
        None => None,
    };
    let mut marks = VecDeque::new();
    mark("pipelined-start");
    for _ in 0..method.frames {
        while marks.len() >= method.frames_in_flight as usize {
            game.host()
                .wait_for_mark(marks.pop_front().expect("frame_in_flight"))?;
        }
        game.frame_in_flight(at(frame))?;
        marks.push_back(game.host().gpu_mark());
        frame += 1;
    }
    game.host().wait_for_gpu()?;
    mark("pipelined-end");
    let mut metrics = Map::new();
    metrics.insert("frame_ms_mean".into(), json!(mean(&times)));
    metrics.insert("load:prepare_ms".into(), json!(prepare_ms));
    if let Some(bytes) = engine_profiler::stack::peak() {
        metrics.insert("mem:main_stack_peak_bytes".into(), json!(bytes));
    }
    let timings = timings(&mut game, method.frames, &mut frame)?;
    let gpu_timestamps = if timings.gpu { "exact" } else { "none" };
    metrics.extend(timings.metrics);
    let adapter = game.host().adapter();
    let bricks = (!costs.is_empty()).then(|| engine::bricks::BrickCosts::table(&costs));
    let report = json!({
        "report": {
            "bricks": bricks,
            "size": [size.0, size.1],
            "frames": method.frames,
            "settled": {"held": held, "after_ms": after_ms},
            "metrics": metrics,
            "frame_ms": times,
            "work": work,
            "capture": capture,
            "gpu_timestamps": gpu_timestamps,
            "unavailable": timings.unavailable,
            "adapter": {"name": adapter.name, "backend": adapter.backend},
        }
    });
    println!("{report}");
    Ok(())
}

fn settle<G: Game>(
    game: &mut Offscreen<G>,
    rule: &Settle,
    launched: Instant,
) -> engine::Result<(bool, f64)> {
    let mut quiet = 0;
    loop {
        let elapsed = launched.elapsed().as_secs_f64() * 1000.0;
        if elapsed >= rule.max_ms {
            return Ok((false, elapsed));
        }
        game.host_mut().suspend();
        let spans = if rule.below.is_empty() {
            game.frame(0.0)?;
            BTreeMap::new()
        } else {
            spans_of_frame(game)?
        };
        let stats: Value =
            serde_json::from_str(&game.host().stats()).expect("stats_json");
        let holds =
            rule.stats
                .iter()
                .all(|(path, value)| stat(&stats, path) == Some(value))
                && rule.below.iter().all(|(metric, threshold)| {
                    spans.get(metric).copied().unwrap_or(0.0) < *threshold
                });
        quiet = if holds { quiet + 1 } else { 0 };
        if quiet >= rule.quiet_frames && elapsed >= rule.min_ms {
            return Ok((true, launched.elapsed().as_secs_f64() * 1000.0));
        }
    }
}

fn stat<'a>(stats: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(stats, |value, key| value.get(key))
}

#[cfg(feature = "profiling")]
fn spans_of_frame<G: Game>(game: &mut Offscreen<G>) -> engine::Result<BTreeMap<String, f64>> {
    game.host_mut()
        .profiling_start(engine::profiling::MAX_EVENTS)?;
    let frame = game.frame(0.0);
    let report = game.host_mut().profiling_stop()?;
    frame?;
    let mut spans = BTreeMap::new();
    for (name, durations) in cpu_spans(&report) {
        spans.insert(name, durations.values().sum());
    }
    Ok(spans)
}

#[cfg(not(feature = "profiling"))]
fn spans_of_frame<G: Game>(game: &mut Offscreen<G>) -> engine::Result<BTreeMap<String, f64>> {
    game.frame(0.0)?;
    Ok(BTreeMap::new())
}

#[cfg(feature = "profiling")]
fn cpu_spans(report: &Value) -> BTreeMap<String, BTreeMap<u64, f64>> {
    let mut spans: BTreeMap<String, BTreeMap<u64, f64>> = BTreeMap::new();
    for event in report["cpu"]["events"].as_array().into_iter().flatten() {
        if !(event["kind"] == "span" || event["kind"] == "aggregate") || event["complete"] != true {
            continue;
        }
        let (Some(name), Some(frame), Some(duration)) = (
            event["name"].as_str(),
            event["frame"].as_u64(),
            event["duration_ms"].as_f64(),
        ) else {
            continue;
        };
        *spans
            .entry(format!("cpu:{name}"))
            .or_default()
            .entry(frame)
            .or_default() += duration;
    }
    spans
}

struct Timings {
    metrics: Map<String, Value>,
    gpu: bool,
    unavailable: Vec<Value>,
}

#[cfg(feature = "profiling")]
fn timings<G: Game>(
    game: &mut Offscreen<G>,
    frames: u32,
    frame: &mut u32,
) -> engine::Result<Timings> {
    const FIRST_CAPTURE: u32 = 30;
    let budget = engine::profiling::MAX_EVENTS;
    let mut passes: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut spans: BTreeMap<String, BTreeMap<u64, f64>> = BTreeMap::new();
    let mut memory: BTreeMap<&str, f64> = BTreeMap::new();
    let (mut left, mut capture, mut complete) = (frames, FIRST_CAPTURE, true);
    while left > 0 {
        let count = capture.min(left);
        game.host_mut().profiling_start(budget)?;
        let mut drawn = Ok(());
        for _ in 0..count {
            drawn = game.frame(at(*frame));
            *frame += 1;
            if drawn.is_err() {
                break;
            }
        }
        let mut report = game.host_mut().profiling_stop()?;
        drawn?;
        game.host().wait_for_gpu()?;
        game.host_mut().profiling_poll();
        report["gpu"] = game.host_mut().profiling_report();
        complete &= report["cpu"]["complete"] == true;
        for sample in report["gpu"]["samples"].as_array().into_iter().flatten() {
            if let (Some(name), Some(duration)) =
                (sample["name"].as_str(), sample["duration_ms"].as_f64())
            {
                passes
                    .entry(format!("gpu:{name}"))
                    .or_default()
                    .push(duration);
            }
        }
        for (name, per_frame) in cpu_spans(&report) {
            spans.entry(name).or_default().extend(per_frame);
        }
        let events = report["cpu"]["events"]
            .as_array()
            .map_or(&[][..], Vec::as_slice);
        for (counter, metric) in [
            ("memory.live_requested_bytes", "mem:live_requested_bytes"),
            (
                "memory.lifetime_peak_requested_bytes",
                "mem:peak_requested_bytes",
            ),
        ] {
            let first = events
                .iter()
                .filter(|event| event["kind"] == "counter" && event["name"] == counter)
                .find_map(|event| event["value"].as_f64());
            if let (Some(bytes), false) = (first, memory.contains_key(metric)) {
                memory.insert(metric, bytes);
            }
        }
        let per_frame = (events.len() / count as usize).max(1);
        capture = u32::try_from(budget * 3 / 4 / per_frame)
            .unwrap_or(u32::MAX)
            .max(1);
        left -= count;
    }
    let mut metrics = Map::new();
    let gpu = !passes.is_empty();
    for (name, durations) in passes {
        metrics.insert(name, json!(engine::host::median(&durations)));
    }
    for (name, per_frame) in spans {
        let mut durations: Vec<f64> = per_frame.into_values().collect();
        durations.resize(durations.len().max(frames as usize), 0.0);
        metrics.insert(name, json!(engine::host::median(&durations)));
    }
    for (metric, bytes) in memory {
        metrics.insert(metric.into(), json!(bytes));
    }
    let mut unavailable = Vec::new();
    if !gpu {
        unavailable.push(json!({
            "metric": "gpu:*",
            "reason": Error::new(ErrorCode::ProfilingUnavailable, "$.gpu.timestamps")
                .expected(Expected::Supported)
                .got(game.host().adapter().backend),
        }));
    }
    if !complete {
        unavailable.push(json!({
            "metric": "cpu:*",
            "reason": Error::new(ErrorCode::ProfilingUnavailable, "$.cpu.events")
                .expected(Expected::Present),
        }));
    }
    Ok(Timings {
        metrics,
        gpu,
        unavailable,
    })
}

#[cfg(not(feature = "profiling"))]
fn timings<G: Game>(_: &mut Offscreen<G>, _: u32, _: &mut u32) -> engine::Result<Timings> {
    let reason =
        Error::new(ErrorCode::ProfilingUnavailable, "$").expected(Expected::Supported);
    Ok(Timings {
        metrics: Map::new(),
        gpu: false,
        unavailable: ["gpu:*", "cpu:*", "mem:live_requested_bytes"]
            .into_iter()
            .map(|metric| json!({"metric": metric, "reason": reason}))
            .collect(),
    })
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}
