//! A measured run (`--measure FILE`): the game offscreen, settled by the
//! workload's rule with time held still, then played from time zero at 60 Hz
//! without pacing: warm-up frames, the sampled frames one at a time (each
//! waits for its GPU work), the frame at the capture's timestamp kept as a
//! PNG, the sampled frames again with frames in flight, and in profiling
//! builds once more with the engine's capture, for each GPU pass's and CPU
//! span's time. Lines on standard output mark where the timed frames start
//! and end, so the program that runs this one times them with its own
//! clock; the last line is the report, which holds each sampled frame's
//! time for the measuring tool to take their spread from. One method file,
//! which the measuring tool writes from the workload, says how.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::time::Instant;

use engine::host::TerrainClock;
use engine::{Engine, Game};
use serde_json::{Map, Value, json};

use crate::Offscreen;
use crate::jobs::JobOptions;

/// How a measured run goes, from its method file.
#[derive(Clone, Debug)]
pub struct Method {
    pub settle: Settle,
    pub warmup_frames: u32,
    pub frames: u32,
    /// The most frames the GPU may still be drawing when the next is
    /// submitted, in the second timing.
    pub frames_in_flight: u32,
    /// The frame kept as a PNG: at that many milliseconds of play.
    pub capture: Option<(f64, PathBuf)>,
}

/// When a run has settled: `quiet_frames` frames in a row whose frame
/// statistics hold the `stats` values and whose CPU spans stay below their
/// thresholds, and at least `min_ms` after launch; a run that has not by
/// `max_ms` is measured anyway, as unsettled.
#[derive(Clone, Debug)]
pub struct Settle {
    pub stats: Map<String, Value>,
    pub quiet_frames: u32,
    pub below: BTreeMap<String, f64>,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl Method {
    /// The method `text` (JSON) gives: the workload's `settle`, `sample`
    /// and `capture` as the measuring tool's task names them, with the
    /// capture's `path`, and `frames_in_flight`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let method: Value =
            serde_json::from_str(text).map_err(|error| format!("method: {error}"))?;
        let number = |value: &Value, path: &str| {
            value
                .as_f64()
                .filter(|number| number.is_finite() && *number >= 0.0)
                .ok_or_else(|| format!("method: {path} is a number of at least 0"))
        };
        let count = |value: &Value, path: &str| {
            value
                .as_u64()
                .and_then(|count| u32::try_from(count).ok())
                .ok_or_else(|| format!("method: {path} is a count"))
        };
        let settle = &method["settle"];
        let stats = match &settle["stats"] {
            Value::Object(stats) => stats.clone(),
            Value::Null => Map::new(),
            Value::Bool(_) | Value::Number(_) | Value::String(_) | Value::Array(_) => {
                return Err("method: settle.stats is an object of statistics' values".into());
            }
        };
        let mut below = BTreeMap::new();
        for (metric, threshold) in settle["metrics"].as_object().into_iter().flatten() {
            if !metric.starts_with("cpu:") {
                return Err(format!(
                    "method: settle.metrics.{metric}: native runs settle on CPU spans (cpu:NAME)"
                ));
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
            return Err("method: sample.paced is false: offscreen frames are unpaced".into());
        }
        let capture = match &method["capture"] {
            Value::Null => None,
            capture @ (Value::Bool(_)
            | Value::Number(_)
            | Value::String(_)
            | Value::Array(_)
            | Value::Object(_)) => {
                if capture["overlays"] == true {
                    return Err("method: capture.overlays: offscreen frames have none".into());
                }
                let path = capture["path"]
                    .as_str()
                    .ok_or("method: capture.path is the PNG file's path")?;
                Some((
                    number(&capture["timestamp_ms"], "capture.timestamp_ms")?,
                    PathBuf::from(path),
                ))
            }
        };
        let frames = count(&sample["frames"], "sample.frames")?;
        let frames_in_flight = count(&method["frames_in_flight"], "frames_in_flight")?;
        if frames == 0 || frames_in_flight == 0 {
            return Err("method: sample.frames and frames_in_flight are above 0".into());
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

/// The milliseconds of play frame `n` draws at: 60 Hz from zero.
fn at(frame: u32) -> f64 {
    f64::from(frame) * 1000.0 / 60.0
}

/// A line on standard output marking a moment the measuring tool times.
fn mark(name: &str) {
    println!("{}", json!({ "mark": name }));
}

/// Runs the game `build` makes as `method` says, `size` pixels on the `gpu`
/// preferred, its terrain timed by `terrain_clock`, and prints the marks and
/// the report.
pub fn run<G: Game, E: Into<engine::Error>>(
    method: &Method,
    size: (u32, u32),
    gpu: engine::GpuPreference,
    jobs: JobOptions,
    terrain_clock: TerrainClock,
    build: impl FnOnce() -> Result<(Engine, G), E>,
) -> engine::Result<()> {
    let launched = Instant::now();
    let mut game = Offscreen::launch(size, gpu, jobs, build)?;
    game.host_mut().set_terrain_clock(terrain_clock);
    game.prepare(0.0)?;
    let prepare_ms = launched.elapsed().as_secs_f64() * 1000.0;
    let (held, after_ms) = settle(&mut game, &method.settle, launched)?;
    // Play starts at time zero, as after preparing.
    game.host_mut().suspend();
    let mut frame = 0;
    for _ in 0..method.warmup_frames {
        game.frame(at(frame))?;
        frame += 1;
    }
    // The sampled frames one at a time: CPU and GPU in series.
    let mut times = Vec::with_capacity(method.frames as usize);
    let (mut loading, mut compiling) = (false, false);
    mark("serialized-start");
    for _ in 0..method.frames {
        let started = Instant::now();
        game.frame(at(frame))?;
        times.push(started.elapsed().as_secs_f64() * 1000.0);
        let stats = game.host().frame_stats();
        loading |= stats.loading;
        compiling |= stats.pipelines_compiling;
        frame += 1;
    }
    mark("serialized-end");
    // The work of the last sampled frame; while any was still loading or
    // compiling, so was the run.
    let mut work: Value =
        serde_json::from_str(&game.host().stats()).expect("frame statistics are JSON");
    work["loading"] = loading.into();
    work["pipelines_compiling"] = compiling.into();
    let capture = match &method.capture {
        Some((timestamp, path)) => {
            if *timestamp < at(frame) {
                return Err(engine::Error::from(format!(
                    "capture.timestamp_ms {timestamp} falls before the end of the warm-up and \
                     sampled frames, {} ms",
                    at(frame)
                )));
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
    // The sampled frames again, each submitted once the frame
    // `frames_in_flight` before it is drawn.
    let mut marks = VecDeque::new();
    mark("pipelined-start");
    for _ in 0..method.frames {
        while marks.len() >= method.frames_in_flight as usize {
            game.host()
                .wait_for_mark(marks.pop_front().expect("frames in flight"))?;
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
    let report = json!({
        "report": {
            "size": [size.0, size.1],
            "frames": method.frames,
            "settled": {"held": held, "after_ms": after_ms},
            "metrics": metrics,
            // Each serialized sampled frame's milliseconds, in order.
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

/// Draws frames with time held still until the run settles as `rule`
/// says, or `rule.max_ms` after `launched`: whether it settled, and when.
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
            serde_json::from_str(&game.host().stats()).expect("frame statistics are JSON");
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

/// The frame statistic at the dotted `path`.
fn stat<'a>(stats: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(stats, |value, key| value.get(key))
}

/// One frame at time zero, and with the engine's spans each `cpu:NAME`
/// span's milliseconds in it, summed (profiling builds).
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

/// Each `cpu:NAME` span of a capture's CPU events: its milliseconds per
/// frame.
#[cfg(feature = "profiling")]
fn cpu_spans(report: &Value) -> BTreeMap<String, BTreeMap<u64, f64>> {
    let mut spans: BTreeMap<String, BTreeMap<u64, f64>> = BTreeMap::new();
    for event in report["cpu"]["events"].as_array().into_iter().flatten() {
        // Spans, and aggregates: a span's time summed over a loop.
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

/// What the engine's capture times of the sampled frames.
struct Timings {
    metrics: Map<String, Value>,
    /// Whether the GPU timed its passes.
    gpu: bool,
    unavailable: Vec<Value>,
}

/// The sampled frames once more with the engine's capture (profiling
/// builds): each GPU pass's median time per frame, each CPU span's median
/// per frame, and the program's allocated memory as the capture began. The
/// capture holds a
/// bounded number of events, so the frames are captured a few at a time:
/// first `FIRST_CAPTURE` frames, then as many as three quarters of the
/// budget holds at the events per frame seen.
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
        // GPU read-backs of the last frames finish once the GPU did.
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
            // As the first captured frame starts: before the capture's
            // own reports allocate.
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
        // Frames the span did not run in count as none of it.
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
            "reason": format!(
                "the GPU ({}) offers no timestamp queries inside passes",
                game.host().adapter().backend
            ),
        }));
    }
    if !complete {
        unavailable.push(json!({
            "metric": "cpu:*",
            "reason": "the engine's capture dropped events: some frames' CPU spans are missing",
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
    let reason = "the program was built without the profiling feature";
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
