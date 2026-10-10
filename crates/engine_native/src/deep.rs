//! Deep profiles, the default native profile on any machine without
//! privileges or setup: the engine's capture (its CPU spans and every
//! frame's GPU passes and opaque draws) together with CPU samples of the
//! game's threads, symbolicated in process to functions, inlined
//! functions, source lines and instructions. One capture writes three
//! files:
//!
//! - the engine's report, as JSON;
//! - a Firefox Profiler profile (open it at profiler.firefox.com, or in
//!   Perfetto at ui.perfetto.dev): call trees and flame graphs per thread,
//!   the engine's spans as markers and as a flame graph of their own, the
//!   GPU's passes and draws on GPU tracks placed at their frame's
//!   submission, and memory counters;
//! - a summary for agents and people: each thread's hottest functions,
//!   the hottest source lines and annotated instructions, GPU pass and draw
//!   costs, what was captured, and the deeper layers recommended to set up.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use engine::{Game, GameHost};
use fxprof_processed_profile::{
    CategoryColor, CategoryHandle, CpuDelta, Frame, FrameFlags, FrameInfo, MarkerFieldFlags,
    MarkerFieldFormat, MarkerTiming, Profile, ReferenceTimestamp, SamplingInterval,
    StaticSchemaMarker, StaticSchemaMarkerField, StringHandle, Timestamp,
};
use serde_json::{Value, json};

use crate::sampler::{SampledThread, Sampler, Samples};

/// How often threads are sampled.
const INTERVAL: Duration = Duration::from_millis(1);

/// A deep profile running.
pub struct DeepCapture {
    sampler: Option<Sampler>,
}

/// The files a deep profile wrote.
pub struct ProfileFiles {
    pub report: PathBuf,
    pub firefox: PathBuf,
    pub summary: PathBuf,
    pub complete: bool,
}

impl DeepCapture {
    /// Starts the engine's capture of at most `max_events` CPU events and
    /// every frame's GPU spans, and samples of `threads`.
    pub(crate) fn start<G: Game>(
        host: &mut GameHost<G>,
        threads: Vec<SampledThread>,
        max_events: usize,
    ) -> Result<Self, String> {
        host.profiling_start(max_events).map_err(String::from)?;
        let sampler = (!threads.is_empty())
            .then(|| Sampler::start(threads, INTERVAL, engine::host::now_ms))
            .flatten();
        Ok(Self { sampler })
    }

    /// Stops capturing and writes the files into `folder` as `stem` with
    /// their extensions; `context` (what ran, on what) goes into the
    /// summary.
    pub(crate) fn finish<G: Game>(
        self,
        host: &mut GameHost<G>,
        folder: &Path,
        stem: &str,
        context: Value,
    ) -> Result<ProfileFiles, String> {
        let samples = self.sampler.map(Sampler::finish);
        let mut report = host.profiling_stop().map_err(String::from)?;
        // GPU read-backs of the last frames finish once the GPU did.
        host.wait_for_gpu().map_err(|error| error.to_string())?;
        host.profiling_poll();
        report["gpu"] = host.profiling_report();
        report["context"] = context;
        let symbols = Symbols::of(samples.as_ref());
        let profile = firefox_profile(&report, samples.as_ref(), &symbols, host.adapter());
        let summary = summary(&report, samples.as_ref(), &symbols, host.adapter());
        std::fs::create_dir_all(folder).map_err(|error| error.to_string())?;
        let files = ProfileFiles {
            report: folder.join(format!("{stem}.json")),
            firefox: folder.join(format!("{stem}.firefox.json")),
            summary: folder.join(format!("{stem}.summary.json")),
            complete: report["gpu"]["complete"] == true
                && report["cpu"]["complete"] == true
                && samples.is_some(),
        };
        let write = |path: &Path, value: &dyn erased::Write| -> Result<(), String> {
            let file = std::fs::File::create(path).map_err(|error| error.to_string())?;
            value
                .write(&mut std::io::BufWriter::new(file))
                .map_err(|error| format!("cannot write {}: {error}", path.display()))
        };
        write(&files.report, &report)?;
        write(&files.firefox, &profile)?;
        let mut summary = summary;
        summary["files"] = json!({
            "report": files.report.display().to_string(),
            "firefox": files.firefox.display().to_string(),
        });
        write(&files.summary, &summary)?;
        Ok(files)
    }
}

/// JSON writing of the two value kinds the files hold.
mod erased {
    pub(super) trait Write {
        fn write(&self, out: &mut dyn std::io::Write) -> serde_json::Result<()>;
    }

    impl Write for serde_json::Value {
        fn write(&self, out: &mut dyn std::io::Write) -> serde_json::Result<()> {
            serde_json::to_writer(out, self)
        }
    }

    impl Write for fxprof_processed_profile::Profile {
        fn write(&self, out: &mut dyn std::io::Write) -> serde_json::Result<()> {
            serde_json::to_writer(out, self)
        }
    }
}

/// A code address's function and source line; inlined functions first
/// (innermost first), then the function it was inlined into.
#[derive(Clone, Default)]
struct Resolved {
    frames: Vec<SymbolFrame>,
    // The outermost function's start address.
    start: Option<u64>,
}

#[derive(Clone)]
struct SymbolFrame {
    function: String,
    file: Option<String>,
    line: Option<u32>,
}

/// Every sampled address's symbols, resolved in process from the program's
/// and libraries' debug information.
struct Symbols(HashMap<u64, Resolved>);

impl Symbols {
    fn of(samples: Option<&Samples>) -> Self {
        let mut resolved = HashMap::new();
        for thread in samples.into_iter().flat_map(|samples| &samples.threads) {
            for sample in &thread.samples {
                for (index, &address) in sample.stack.iter().enumerate() {
                    // Return addresses point after their call.
                    let lookup = if index == 0 {
                        address
                    } else {
                        address.saturating_sub(1)
                    };
                    resolved.entry(lookup).or_insert_with(|| resolve(lookup));
                }
            }
        }
        Self(resolved)
    }

    fn get(&self, address: u64, leaf: bool) -> Option<&Resolved> {
        self.0.get(
            &(if leaf {
                address
            } else {
                address.saturating_sub(1)
            }),
        )
    }
}

fn resolve(address: u64) -> Resolved {
    let mut resolved = Resolved::default();
    backtrace::resolve(address as *mut std::ffi::c_void, |symbol| {
        let function = symbol
            .name()
            .map_or_else(|| format!("{address:#x}"), |name| format!("{name:#}"));
        resolved.frames.push(SymbolFrame {
            function,
            file: symbol.filename().map(|file| file.display().to_string()),
            line: symbol.lineno(),
        });
        if let Some(start) = symbol.addr() {
            resolved.start = Some(start as u64);
        }
    });
    resolved
}

/// The functions of `stack` (leaf first) from the root down, inlined
/// functions expanded; unknown code as its address.
fn functions(stack: &[u64], symbols: &Symbols) -> Vec<String> {
    let mut names = Vec::with_capacity(stack.len() * 2);
    for (index, &address) in stack.iter().enumerate().rev() {
        match symbols.get(address, index == 0) {
            Some(resolved) if !resolved.frames.is_empty() => {
                names.extend(
                    resolved
                        .frames
                        .iter()
                        .rev()
                        .map(|frame| frame.function.clone()),
                );
            }
            _ => names.push(format!("{address:#x}")),
        }
    }
    names
}

/// The engine's spans and the GPU's passes and draws as markers.
struct SpanMarker {
    name: StringHandle,
    kind: StringHandle,
    category: CategoryHandle,
}

impl StaticSchemaMarker for SpanMarker {
    const UNIQUE_MARKER_TYPE_NAME: &'static str = "EngineSpan";
    const CHART_LABEL: Option<&'static str> = Some("{marker.name}");
    const TABLE_LABEL: Option<&'static str> = Some("{marker.name} ({marker.data.kind})");
    const FIELDS: &'static [StaticSchemaMarkerField] = &[StaticSchemaMarkerField {
        key: "kind",
        label: "Kind",
        format: MarkerFieldFormat::String,
        flags: MarkerFieldFlags::SEARCHABLE,
    }];

    fn name(&self, _: &mut Profile) -> StringHandle {
        self.name
    }

    fn category(&self, _: &mut Profile) -> CategoryHandle {
        self.category
    }

    fn string_field_value(&self, _: u32) -> StringHandle {
        self.kind
    }

    fn number_field_value(&self, _: u32) -> f64 {
        0.0
    }
}

fn at(ms: f64) -> Timestamp {
    Timestamp::from_millis_since_reference(ms.max(0.0))
}

/// The Firefox Profiler profile of `report` and `samples`.
fn firefox_profile(
    report: &Value,
    samples: Option<&Samples>,
    symbols: &Symbols,
    adapter: &engine::AdapterInfo,
) -> Profile {
    // The host's clock counts from an instant this subtracts from now.
    let now = engine::host::now_ms();
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |since| since.as_secs_f64() * 1000.0)
        - now;
    let interval = samples.map_or(INTERVAL, |samples| {
        Duration::from_secs_f64(samples.interval_ms / 1000.0)
    });
    let product = format!("engine_native on {}", adapter.name);
    let mut profile = Profile::new(
        &product,
        ReferenceTimestamp::from_millis_since_unix_epoch(epoch),
        SamplingInterval::from_nanos(interval.as_nanos() as u64),
    );
    profile.set_os_name(std::env::consts::OS);
    let start = report["cpu"]["start_ms"].as_f64().unwrap_or(0.0);
    let end = report["cpu"]["end_ms"].as_f64().unwrap_or(now);
    let process = profile.add_process(&product, std::process::id(), at(start));
    let native = profile.add_category("Native code", CategoryColor::Blue);
    let engine_category = profile.add_category("Engine spans", CategoryColor::Green);
    let gpu_category = profile.add_category("GPU", CategoryColor::Orange);
    // CPU samples, a thread each; the first draws the game.
    let mut main = None;
    for (index, thread) in samples
        .into_iter()
        .flat_map(|samples| &samples.threads)
        .enumerate()
    {
        let handle = profile.add_thread(process, index as u32 + 1, at(start), index == 0);
        profile.set_thread_name(handle, &thread.name);
        if index == 0 {
            profile.add_initial_selected_thread(handle);
            main = Some(handle);
        }
        profile.add_initial_visible_thread(handle);
        for sample in &thread.samples {
            let names = functions(&sample.stack, symbols);
            let frames: Vec<FrameInfo> = names
                .iter()
                .map(|name| FrameInfo {
                    frame: Frame::Label(profile.intern_string(name)),
                    category_pair: native.into(),
                    flags: FrameFlags::empty(),
                })
                .collect();
            let stack = profile.intern_stack_frames(handle, frames.into_iter());
            let cpu = sample.cpu_ns.map_or_else(
                || CpuDelta::from_nanos(interval.as_nanos() as u64),
                CpuDelta::from_nanos,
            );
            profile.add_sample(handle, at(sample.time_ms), stack, cpu, 1);
        }
    }
    // The engine's spans: markers on the thread that drew the game, beside
    // its samples, and a flame graph of their nesting on a track of its own,
    // sampled at the same interval.
    let events: Vec<&Value> = report["cpu"]["events"]
        .as_array()
        .map(|events| {
            events
                .iter()
                .filter(|event| event["kind"] == "span")
                .collect()
        })
        .unwrap_or_default();
    let spans = profile.add_thread(process, 0, at(start), false);
    profile.set_thread_name(spans, "Engine spans");
    profile.add_initial_visible_thread(spans);
    let span_kind = profile.intern_string("engine span");
    let mut nested: Vec<(f64, f64, u64, String)> = Vec::with_capacity(events.len());
    for event in &events {
        let (Some(begin), Some(duration)) =
            (event["start_ms"].as_f64(), event["duration_ms"].as_f64())
        else {
            continue;
        };
        let name = event["name"].as_str().unwrap_or("span");
        let marker = SpanMarker {
            name: profile.intern_string(name),
            kind: span_kind,
            category: engine_category,
        };
        profile.add_marker(
            main.unwrap_or(spans),
            MarkerTiming::Interval(at(begin), at(begin + duration)),
            marker,
        );
        nested.push((
            begin,
            begin + duration,
            event["depth"].as_u64().unwrap_or(0),
            name.to_owned(),
        ));
    }
    nested.sort_by(|a, b| a.0.total_cmp(&b.0));
    let step = interval.as_secs_f64() * 1000.0;
    let mut active: Vec<&(f64, f64, u64, String)> = Vec::new();
    let mut next = 0;
    let mut time = start;
    while time < end {
        while next < nested.len() && nested[next].0 <= time {
            active.push(&nested[next]);
            next += 1;
        }
        active.retain(|span| span.1 > time);
        if !active.is_empty() {
            let mut stack = active.clone();
            stack.sort_by_key(|span| span.2);
            let frames: Vec<FrameInfo> = stack
                .iter()
                .map(|span| FrameInfo {
                    frame: Frame::Label(profile.intern_string(&span.3)),
                    category_pair: engine_category.into(),
                    flags: FrameFlags::empty(),
                })
                .collect();
            let stack = profile.intern_stack_frames(spans, frames.into_iter());
            profile.add_sample(spans, at(time), stack, CpuDelta::from_millis(step), 1);
        }
        time += step;
    }
    // The GPU's passes and draws, each frame's at its submission: GPU
    // clocks are not the CPU's, so frames sit where they were submitted and
    // their spans at their offsets within the frame.
    let submissions: HashMap<String, f64> = events
        .iter()
        .filter(|event| event["name"] == "submit_and_present")
        .filter_map(|event| {
            Some((
                event["frame"].as_u64()?.to_string(),
                event["start_ms"].as_f64()?,
            ))
        })
        .collect();
    let gpu = &report["gpu"];
    for (track, list, kind_name) in [
        ("GPU passes", &gpu["samples"], "gpu pass"),
        ("GPU draws", &gpu["draw_samples"], "gpu draw"),
    ] {
        let Some(list) = list.as_array().filter(|list| !list.is_empty()) else {
            continue;
        };
        let handle = profile.add_thread(process, 0, at(start), false);
        profile.set_thread_name(handle, track);
        profile.set_thread_show_markers_in_timeline(handle, true);
        profile.add_initial_visible_thread(handle);
        let kind = profile.intern_string(kind_name);
        for sample in list {
            let (Some(frame), Some(offset), Some(duration)) = (
                sample["frame"]
                    .as_str()
                    .and_then(|frame| submissions.get(frame)),
                sample["start_ms"].as_f64(),
                sample["duration_ms"].as_f64(),
            ) else {
                continue;
            };
            let name = sample["name"]
                .as_str()
                .or_else(|| sample["label"].as_str())
                .unwrap_or("gpu");
            let marker = SpanMarker {
                name: profile.intern_string(name),
                kind,
                category: gpu_category,
            };
            let begin = frame + offset;
            profile.add_marker(
                handle,
                MarkerTiming::Interval(at(begin), at(begin + duration)),
                marker,
            );
        }
    }
    // The program's allocations, as the engine counted them; its other
    // counters are per-frame tallies the report keeps.
    let mut counters: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    for event in report["cpu"]["events"].as_array().into_iter().flatten() {
        if event["kind"] == "counter"
            && event["name"].as_str().is_some_and(|name| {
                name == "memory.live_requested_bytes" || name == "memory.allocations_total"
            })
            && let (Some(name), Some(time), Some(value)) = (
                event["name"].as_str(),
                event["start_ms"].as_f64(),
                event["value"].as_f64(),
            )
        {
            counters
                .entry(name.to_owned())
                .or_default()
                .push((time, value));
        }
    }
    for (name, values) in counters {
        let counter = profile.add_counter(process, &name, "Engine", &name);
        let mut last = 0.0;
        for (time, value) in values {
            profile.add_counter_sample(counter, at(time), value - last, 1);
            last = value;
        }
    }
    profile
}

/// Counts of names, most first, with their share of `total`, at most
/// `limit`.
fn ranked(counts: HashMap<String, u64>, total: u64, limit: usize) -> Vec<Value> {
    let mut counts: Vec<(String, u64)> = counts.into_iter().collect();
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    counts
        .into_iter()
        .take(limit)
        .map(|(name, samples)| {
            json!({"name": name, "samples": samples, "share": samples as f64 / total.max(1) as f64})
        })
        .collect()
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

/// A hot function's samples, and its sampled instructions' samples by
/// address.
type Hot = (u64, BTreeMap<u64, u64>);

/// The summary of a capture.
fn summary(
    report: &Value,
    samples: Option<&Samples>,
    symbols: &Symbols,
    adapter: &engine::AdapterInfo,
) -> Value {
    let mut threads = Vec::new();
    let mut deepest = 0;
    for thread in samples.into_iter().flat_map(|samples| &samples.threads) {
        let mut own: HashMap<String, u64> = HashMap::new();
        let mut inclusive: HashMap<String, u64> = HashMap::new();
        for sample in &thread.samples {
            let names = functions(&sample.stack, symbols);
            deepest = deepest.max(names.len());
            if let Some(leaf) = names.last() {
                *own.entry(leaf.clone()).or_default() += 1;
            }
            let mut seen: Vec<&String> = Vec::with_capacity(names.len());
            for name in &names {
                if !seen.contains(&name) {
                    seen.push(name);
                    *inclusive.entry(name.clone()).or_default() += 1;
                }
            }
        }
        let total = thread.samples.len() as u64;
        // Frames in every sample are the thread's roots: no information.
        inclusive.retain(|_, samples| *samples < total);
        threads.push(json!({
            "name": thread.name,
            "samples": total,
            "missed": thread.missed,
            "self": ranked(own, total, 25),
            "total": ranked(inclusive, total, 25),
        }));
    }
    // The main thread's hottest source lines and instructions: where its
    // samples were, inlined code at its own line.
    let main = samples.and_then(|samples| samples.threads.first());
    let mut lines: HashMap<String, u64> = HashMap::new();
    let mut hot: HashMap<u64, Hot> = HashMap::new();
    for sample in main.into_iter().flat_map(|thread| &thread.samples) {
        let Some(&leaf) = sample.stack.first() else {
            continue;
        };
        let resolved = symbols.get(leaf, true);
        if let Some(frame) = resolved.and_then(|resolved| resolved.frames.first()) {
            let line = match (&frame.file, frame.line) {
                (Some(file), Some(line)) => format!("{file}:{line} ({})", frame.function),
                _ => frame.function.clone(),
            };
            *lines.entry(line).or_default() += 1;
        }
        // Instructions of the program's own functions, which have sources.
        let sourced = resolved
            .and_then(|resolved| resolved.frames.first())
            .is_some_and(|frame| frame.file.is_some());
        if let Some(start) = resolved
            .and_then(|resolved| resolved.start)
            .filter(|_| sourced)
        {
            let (count, addresses) = hot.entry(start).or_default();
            *count += 1;
            *addresses.entry(leaf).or_default() += 1;
        }
    }
    let main_total = main.map_or(0, |thread| thread.samples.len() as u64);
    let mut functions_hot: Vec<(u64, Hot)> = hot.into_iter().collect();
    functions_hot.sort_by_key(|(_, (count, _))| std::cmp::Reverse(*count));
    let instructions: Vec<Value> = functions_hot
        .iter()
        .take(3)
        .map(|(start, (count, addresses))| annotated(*start, *count, addresses, symbols))
        .collect();
    // GPU passes and draws: each name's median over the captured frames.
    let gpu = &report["gpu"];
    let mut passes: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for sample in gpu["samples"].as_array().into_iter().flatten() {
        if let (Some(name), Some(duration)) =
            (sample["name"].as_str(), sample["duration_ms"].as_f64())
        {
            passes.entry(name.to_owned()).or_default().push(duration);
        }
    }
    let mut passes: Vec<(String, f64, usize)> = passes
        .into_iter()
        .filter_map(|(name, mut values)| Some((name, median(&mut values)?, values.len())))
        .collect();
    passes.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut draws: BTreeMap<String, (Vec<f64>, Option<Value>)> = BTreeMap::new();
    for draw in gpu["draw_samples"].as_array().into_iter().flatten() {
        let Some(label) = draw["label"].as_str() else {
            continue;
        };
        let entry = draws.entry(label.to_owned()).or_default();
        if let Some(duration) = draw["duration_ms"].as_f64() {
            entry.0.push(duration);
        }
        if entry.1.is_none() && !draw["vertex_invocations"].is_null() {
            entry.1 = Some(json!({
                "vertex_invocations": draw["vertex_invocations"],
                "primitives": draw["primitives"],
                "fragment_invocations": draw["fragment_invocations"],
            }));
        }
    }
    let mut draws: Vec<(String, f64, Option<Value>)> = draws
        .into_iter()
        .filter_map(|(label, (mut values, statistics))| {
            Some((label, median(&mut values)?, statistics))
        })
        .collect();
    draws.sort_by(|a, b| b.1.total_cmp(&a.1));
    let sampled = samples.is_some();
    json!({
        "machine": crate::capabilities::machine(adapter),
        "captured": {
            "cpu_samples": samples.map(|samples| json!({
                "method": samples.method,
                "interval_ms": samples.interval_ms,
                "threads": samples.threads.iter().map(|thread| json!({
                    "name": thread.name, "samples": thread.samples.len(), "missed": thread.missed,
                })).collect::<Vec<_>>(),
                "symbols": "in process, from the program's and libraries' debug information",
            })),
            "cpu_samples_unavailable": (!sampled).then_some("this operating system's threads cannot be sampled in process"),
            "engine_spans": {
                "events": report["cpu"]["events"].as_array().map_or(0, Vec::len),
                "complete": report["cpu"]["complete"],
            },
            "gpu": {
                "frames": gpu["completed"],
                "complete": gpu["complete"],
                "draws": gpu["draws"],
            },
            "privileges": "none: no admin, no system setting, no installed tool",
        },
        "threads": threads,
        "hot_lines": ranked(lines, main_total, 30),
        "hot_instructions": instructions,
        "gpu_passes_ms": passes.iter().take(40).map(|(name, ms, frames)| json!({"name": name, "median_ms": ms, "frames": frames})).collect::<Vec<_>>(),
        "gpu_draws_ms": draws.iter().take(25).map(|(label, ms, statistics)| json!({"draw": label, "median_ms": ms, "statistics": statistics})).collect::<Vec<_>>(),
        "recommended": crate::capabilities::recommended(adapter, sampled && deepest < 8),
        "open": "Load the .firefox.json file at https://profiler.firefox.com (Load a profile from file; it stays on this machine) or at https://ui.perfetto.dev",
    })
}

/// A hot function's instructions from its start to past its last sampled
/// one, each with its samples and the sampled ones' source lines; on x86
/// disassembled, elsewhere as addresses.
fn annotated(start: u64, count: u64, addresses: &BTreeMap<u64, u64>, symbols: &Symbols) -> Value {
    let function = resolve(start)
        .frames
        .last()
        .map_or_else(|| format!("{start:#x}"), |frame| frame.function.clone());
    let last = addresses.keys().next_back().copied().unwrap_or(start);
    let line_of = |address: u64| {
        symbols
            .get(address, true)
            .and_then(|resolved| resolved.frames.first())
            .and_then(|frame| Some(format!("{}:{}", frame.file.as_deref()?, frame.line?)))
    };
    let instructions = disassemble(start, last, addresses, &line_of);
    json!({
        "function": function,
        "samples": count,
        "start": format!("{start:#x}"),
        "instructions": instructions,
    })
}

#[cfg(target_arch = "x86_64")]
fn disassemble(
    start: u64,
    last: u64,
    addresses: &BTreeMap<u64, u64>,
    line_of: &dyn Fn(u64) -> Option<String>,
) -> Vec<Value> {
    use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};
    // Through the last sampled instruction, at most 256 KiB of a function.
    let length = ((last.saturating_sub(start)) as usize + 16).min(1 << 18);
    let code = crate::sampler::code(start, length);
    let mut decoder = Decoder::with_ip(64, &code, start, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    let decoded: Vec<_> = std::iter::from_fn(|| decoder.can_decode().then(|| decoder.decode()))
        .take_while(|instruction| instruction.ip() <= last)
        .collect();
    // The sampled instructions with the four before and after each.
    let near = |index: usize| {
        decoded[index.saturating_sub(4)..(index + 5).min(decoded.len())]
            .iter()
            .any(|instruction| addresses.contains_key(&instruction.ip()))
    };
    let mut text = String::new();
    let mut out = Vec::new();
    let mut skipped = false;
    for (index, instruction) in decoded.iter().enumerate() {
        if !near(index) {
            skipped = true;
            continue;
        }
        if std::mem::take(&mut skipped) {
            out.push(json!({"skipped": true}));
        }
        text.clear();
        formatter.format(instruction, &mut text);
        let samples = addresses.get(&instruction.ip()).copied().unwrap_or(0);
        out.push(json!({
            "offset": instruction.ip() - start,
            "samples": samples,
            "text": text,
            "line": (samples > 0).then(|| line_of(instruction.ip())).flatten(),
        }));
    }
    out
}

#[cfg(not(target_arch = "x86_64"))]
fn disassemble(
    start: u64,
    _: u64,
    addresses: &BTreeMap<u64, u64>,
    line_of: &dyn Fn(u64) -> Option<String>,
) -> Vec<Value> {
    addresses
        .iter()
        .map(|(&address, &samples)| {
            json!({"offset": address - start, "samples": samples, "line": line_of(address)})
        })
        .collect()
}
