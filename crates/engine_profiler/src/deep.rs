use engine::collections::HashMap;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use engine::host::median;
use engine::{Error, ErrorCode, Expected, Game, GameHost};
use fxprof_processed_profile::{
    CategoryColor, CategoryHandle, CpuDelta, Frame, FrameFlags, FrameInfo, MarkerFieldFlags,
    MarkerFieldFormat, MarkerTiming, Profile, ReferenceTimestamp, SamplingInterval,
    StaticSchemaMarker, StaticSchemaMarkerField, StringHandle, Timestamp,
};
use serde_json::{Value, json};

use crate::sampler::{SampledThread, Sampler, Samples};

const INTERVAL: Duration = Duration::from_millis(1);
const HEAP_SAMPLES: usize = 65_536;
const HEAP_FRAMES: usize = 48;
const HEAP_STACKS: usize = 12;
const SPANS_PER_BRICK: usize = 15;
const ALLOCATOR_FRAMES: &[&str] = &[
    "backtrace::",
    "engine_profiler::deep::",
    "engine_profile::",
    "__rust_",
    "__rdl_",
    "__rg_",
    "alloc::alloc::",
    "alloc::raw_vec::",
    "std::alloc::",
    "core::alloc::",
];

struct HeapSample {
    brick: u16,
    bytes: usize,
    stack: Vec<u64>,
}

static HEAP: Mutex<Vec<HeapSample>> = Mutex::new(Vec::new());

fn sample_heap(brick: u16, bytes: usize) {
    let mut frames = [0_u64; HEAP_FRAMES];
    let mut depth = 0;
    backtrace::trace(|frame| {
        frames[depth] = frame.ip() as u64;
        depth += 1;
        depth < HEAP_FRAMES
    });
    let Ok(mut samples) = HEAP.try_lock() else {
        return;
    };
    if samples.len() < HEAP_SAMPLES {
        samples.push(HeapSample {
            brick,
            bytes,
            stack: frames[..depth].to_vec(),
        });
    }
}

pub struct DeepCapture {
    sampler: Option<Sampler>,
}

pub struct ProfileFiles {
    pub report: PathBuf,
    pub firefox: PathBuf,
    pub summary: PathBuf,
    pub complete: bool,
}

impl DeepCapture {
    pub fn start<G: Game>(
        host: &mut GameHost<G>,
        threads: Vec<SampledThread>,
        max_events: usize,
    ) -> Result<Self, Error> {
        host.profiling_start(max_events)?;
        HEAP.lock().unwrap_or_else(PoisonError::into_inner).clear();
        engine::profiling::sample_allocations(Some(sample_heap));
        crate::counters::begin();
        let sampler = (!threads.is_empty())
            .then(|| Sampler::start(threads, INTERVAL, engine::host::now_ms))
            .flatten();
        Ok(Self { sampler })
    }

    pub fn finish<G: Game>(
        self,
        host: &mut GameHost<G>,
        folder: &Path,
        stem: &str,
        context: Value,
    ) -> Result<ProfileFiles, Error> {
        let samples = self.sampler.map(Sampler::finish);
        engine::profiling::sample_allocations(None);
        let heap = std::mem::take(&mut *HEAP.lock().unwrap_or_else(PoisonError::into_inner));
        let mut report = host.profiling_stop()?;
        host.wait_for_gpu()?;
        host.profiling_poll();
        report["gpu"] = host.profiling_report();
        report["context"] = context;
        if let Some(counters) = crate::counters::end() {
            report["gpu_counters"] = counters;
        }
        let symbols = Symbols::of(samples.as_ref(), &heap);
        report["heap_samples_by_brick"] = heap_by_brick(&heap, &symbols);
        let profile = firefox_profile(&report, samples.as_ref(), &symbols, host.adapter());
        let summary = summary(&report, samples.as_ref(), &symbols, host.adapter());
        let unwritable = |path: &Path| {
            Error::new(ErrorCode::SourceIo, "$.folder")
                .expected(Expected::Writable)
                .got(path.display().to_string())
        };
        std::fs::create_dir_all(folder).map_err(|_| unwritable(folder))?;
        let files = ProfileFiles {
            report: folder.join(format!("{stem}.json")),
            firefox: folder.join(format!("{stem}.firefox.json")),
            summary: folder.join(format!("{stem}.summary.json")),
            complete: report["gpu"]["complete"] == true
                && report["cpu"]["complete"] == true
                && samples.is_some(),
        };
        let write = |path: &Path, value: &dyn erased::Write| -> Result<(), Error> {
            let file = std::fs::File::create(path).map_err(|_| unwritable(path))?;
            value
                .write(&mut std::io::BufWriter::new(file))
                .map_err(|_| unwritable(path))
        };
        write(&files.report, &report)?;
        write(&files.firefox, &profile)?;
        let mut summary = summary;
        summary["bricks"] = report["bricks"].clone();
        summary["heap_samples_by_brick"] = report["heap_samples_by_brick"].clone();
        summary["spans_by_brick"] = spans_by_brick(&report);
        summary["files"] = json!({
            "report": files.report.display().to_string(),
            "firefox": files.firefox.display().to_string(),
        });
        write(&files.summary, &summary)?;
        Ok(files)
    }
}

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

#[derive(Clone, Default)]
struct Resolved {
    frames: Vec<SymbolFrame>,
    start: Option<u64>,
}

#[derive(Clone)]
struct SymbolFrame {
    function: String,
    file: Option<String>,
    line: Option<u32>,
}

struct Symbols(HashMap<u64, Resolved>);

impl Symbols {
    fn of(samples: Option<&Samples>, heap: &[HeapSample]) -> Self {
        let mut resolved = HashMap::default();
        let stacks = samples
            .into_iter()
            .flat_map(|samples| &samples.threads)
            .flat_map(|thread| thread.samples.iter().map(|sample| &sample.stack))
            .chain(heap.iter().map(|sample| &sample.stack));
        for stack in stacks {
            for (index, &address) in stack.iter().enumerate() {
                let lookup = if index == 0 {
                    address
                } else {
                    address.saturating_sub(1)
                };
                resolved.entry(lookup).or_insert_with(|| resolve(lookup));
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

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn firefox_profile(
    report: &Value,
    samples: Option<&Samples>,
    symbols: &Symbols,
    adapter: &engine::AdapterInfo,
) -> Profile {
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

fn heap_by_brick(samples: &[HeapSample], symbols: &Symbols) -> Value {
    type Stacks = HashMap<Vec<String>, (u64, u64)>;
    let mut bricks: BTreeMap<&'static str, (u64, u64, Stacks)> = BTreeMap::new();
    for sample in samples {
        let mut names = functions(&sample.stack, symbols);
        while names.last().is_some_and(|name| {
            ALLOCATOR_FRAMES
                .iter()
                .any(|frame| name.starts_with(frame) || name.contains(&format!("<{frame}")))
        }) {
            names.pop();
        }
        let callers = names.split_off(names.len().saturating_sub(HEAP_STACKS));
        let brick = engine::profiling::brick_name(sample.brick);
        let (count, bytes, stacks) = bricks.entry(brick).or_default();
        *count += 1;
        *bytes += sample.bytes as u64;
        let (stack_count, stack_bytes) = stacks.entry(callers).or_default();
        *stack_count += 1;
        *stack_bytes += sample.bytes as u64;
    }
    let mut rows: Vec<(&str, u64, u64, Stacks)> = bricks
        .into_iter()
        .map(|(brick, (count, bytes, stacks))| (brick, count, bytes, stacks))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    json!({
        "every": engine::profiling::SAMPLE_EVERY,
        "bricks": rows
            .into_iter()
            .map(|(brick, count, bytes, stacks)| {
                let mut stacks: Vec<(Vec<String>, (u64, u64))> = stacks.into_iter().collect();
                stacks.sort_by(|a, b| b.1.0.cmp(&a.1.0).then_with(|| a.0.cmp(&b.0)));
                json!({
                    "brick": brick,
                    "samples": count,
                    "sampled_bytes": bytes,
                    "stacks": stacks
                        .into_iter()
                        .take(10)
                        .map(|(frames, (samples, bytes))| json!({
                            "samples": samples,
                            "sampled_bytes": bytes,
                            "frames": frames,
                        }))
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>(),
    })
}

fn spans_by_brick(report: &Value) -> Value {
    let mut open: Vec<Option<String>> = Vec::new();
    let mut frames: Vec<u64> = Vec::new();
    let mut totals: BTreeMap<String, BTreeMap<String, (f64, u64)>> = BTreeMap::new();
    for event in report["cpu"]["events"].as_array().into_iter().flatten() {
        if event["kind"] != "span" {
            continue;
        }
        let (Some(name), Some(depth), Some(duration)) = (
            event["name"].as_str(),
            event["depth"].as_u64(),
            event["duration_ms"].as_f64(),
        ) else {
            continue;
        };
        if let Some(frame) = event["frame"].as_u64()
            && !frames.contains(&frame)
        {
            frames.push(frame);
        }
        open.truncate(depth as usize);
        let enclosing = open.iter().rev().find_map(Clone::clone);
        if event["category"] == "brick" {
            open.push(Some(name.to_owned()));
            continue;
        }
        if let Some(brick) = &enclosing {
            let (total, count) = totals
                .entry(brick.clone())
                .or_default()
                .entry(name.to_owned())
                .or_default();
            *total += duration;
            *count += 1;
        }
        open.push(enclosing);
    }
    let count = frames.len().max(1) as f64;
    let bricks: serde_json::Map<String, Value> = totals
        .into_iter()
        .map(|(brick, spans)| {
            let mut spans: Vec<(String, (f64, u64))> = spans.into_iter().collect();
            spans.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            let rows: Vec<Value> = spans
                .into_iter()
                .take(SPANS_PER_BRICK)
                .map(|(name, (total, calls))| {
                    json!({
                        "name": name,
                        "ms_per_frame": total / count,
                        "calls": calls,
                    })
                })
                .collect();
            (brick, Value::Array(rows))
        })
        .collect();
    json!({
        "frames": frames.len(),
        "bricks": bricks,
    })
}

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

type Hot = (u64, BTreeMap<u64, u64>);

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn summary(
    report: &Value,
    samples: Option<&Samples>,
    symbols: &Symbols,
    adapter: &engine::AdapterInfo,
) -> Value {
    let mut threads = Vec::new();
    let mut deepest = 0;
    for thread in samples.into_iter().flat_map(|samples| &samples.threads) {
        let mut own: HashMap<String, u64> = HashMap::default();
        let mut inclusive: HashMap<String, u64> = HashMap::default();
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
        inclusive.retain(|_, samples| *samples < total);
        threads.push(json!({
            "name": thread.name,
            "samples": total,
            "missed": thread.missed,
            "self": ranked(own, total, 25),
            "total": ranked(inclusive, total, 25),
        }));
    }
    let main = samples.and_then(|samples| samples.threads.first());
    let mut lines: HashMap<String, u64> = HashMap::default();
    let mut hot: HashMap<u64, Hot> = HashMap::default();
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
        .filter_map(|(name, values)| Some((name, median(&values)?, values.len())))
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
        .filter_map(|(label, (values, statistics))| Some((label, median(&values)?, statistics)))
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
                "symbols": "in_process",
            })),
            "cpu_samples_unavailable": (!sampled).then(|| {
                Error::new(ErrorCode::ProfilingUnavailable, "$.cpu_samples")
                    .expected(Expected::Supported)
            }),
            "engine_spans": {
                "events": report["cpu"]["events"].as_array().map_or(0, Vec::len),
                "complete": report["cpu"]["complete"],
            },
            "gpu": {
                "frames": gpu["completed"],
                "complete": gpu["complete"],
                "draws": gpu["draws"],
            },
            "privileges": "none",
        },
        "threads": threads,
        "hot_lines": ranked(lines, main_total, 30),
        "hot_instructions": instructions,
        "gpu_passes_ms": passes.iter().take(40).map(|(name, ms, frames)| json!({"name": name, "median_ms": ms, "frames": frames})).collect::<Vec<_>>(),
        "gpu_draws_ms": draws.iter().take(25).map(|(label, ms, statistics)| json!({"draw": label, "median_ms": ms, "statistics": statistics})).collect::<Vec<_>>(),
        "gpu_counters": report["gpu_counters"],
        "recommended": crate::capabilities::recommended(adapter, sampled && deepest < 8),
        "viewers": ["https://profiler.firefox.com", "https://ui.perfetto.dev"],
    })
}

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
    let length = ((last.saturating_sub(start)) as usize + 16).min(1 << 18);
    let code = crate::sampler::code(start, length);
    let mut decoder = Decoder::with_ip(64, &code, start, DecoderOptions::NONE);
    let mut formatter = IntelFormatter::new();
    let decoded: Vec<_> = std::iter::from_fn(|| decoder.can_decode().then(|| decoder.decode()))
        .take_while(|instruction| instruction.ip() <= last)
        .collect();
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
