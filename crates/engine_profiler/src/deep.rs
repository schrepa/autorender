use engine::collections::HashMap;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine::{Error, ErrorCode, Expected};
use engine_profile::{Event, Facts};
use engine_registry::{Kind, Lands, NameEntry};
use fxprof_processed_profile::{
    CategoryColor, CategoryHandle, CpuDelta, Frame, FrameFlags, FrameInfo, MarkerFieldFlags,
    MarkerFieldFormat, MarkerTiming, Profile, ReferenceTimestamp, SamplingInterval,
    StaticSchemaMarker, StaticSchemaMarkerField, StringHandle, ThreadHandle, Timestamp,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::sampler::{SampledThread, Sampler, Samples};

const INTERVAL: Duration = Duration::from_millis(1);
const HEAP_FRAMES: usize = 48;
const HEAP_STACKS: usize = 24;
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
    bytes: usize,
    stack: Vec<u64>,
}

struct Heap {
    samples: Vec<HeapSample>,
    limit: usize,
    dropped: u64,
}

static HEAP: Mutex<Heap> = Mutex::new(Heap {
    samples: Vec::new(),
    limit: 0,
    dropped: 0,
});
static HEAP_BUSY: AtomicU64 = AtomicU64::new(0);

fn sample_heap(bytes: usize) {
    let mut frames = [0_u64; HEAP_FRAMES];
    let mut depth = 0;
    backtrace::trace(|frame| {
        frames[depth] = frame.ip() as u64;
        depth += 1;
        depth < HEAP_FRAMES
    });
    let Ok(mut heap) = HEAP.try_lock() else {
        HEAP_BUSY.fetch_add(1, Relaxed);
        return;
    };
    if heap.samples.len() < heap.limit {
        heap.samples.push(HeapSample {
            bytes,
            stack: frames[..depth].to_vec(),
        });
    } else {
        heap.dropped += 1;
    }
}

const UNSET: &str = "settings.deep_not_set";

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeepSizes {
    pub heap_samples: usize,
    pub thread_samples: usize,
    pub events: usize,
    pub rows: usize,
}

#[derive(Resource, Default)]
struct DeepSetting(Option<DeepSizes>);

pub fn deep_setting(world: &mut World) {
    world.init_resource::<DeepSetting>();
    let entry = NameEntry::read("settings.deep", Kind::Setting, |reach, _| {
        Ok(reach
            .world
            .get_resource::<DeepSetting>()
            .and_then(|setting| setting.0)
            .map_or_else(|| json!({ "unavailable": UNSET }), |sizes| json!(sizes)))
    })
    .args(
        json!({ "value": "heap_samples, thread_samples, events, rows (set, all four)" }),
        json!({ "value": { "heap_samples": 65_536, "thread_samples": 60_000, "events": 65_536, "rows": 30 } }),
    )
    .set(Lands::ThisFrame, |reach, args| {
        let sizes: DeepSizes = args
            .get("value")
            .cloned()
            .ok_or_else(|| Error::new(ErrorCode::InvalidArguments, "$.args.value").expected(Expected::Present))
            .and_then(|value| {
                serde_json::from_value(value).map_err(|error| {
                    Error::new(ErrorCode::InvalidArguments, "$.args.value").got(error.to_string())
                })
            })?;
        reach.world.insert_resource(DeepSetting(Some(sizes)));
        Ok(json!(sizes))
    });
    engine_registry::register(world, entry);
}

pub fn deep_sizes(world: &World) -> Result<DeepSizes, &'static str> {
    world
        .get_resource::<DeepSetting>()
        .and_then(|setting| setting.0)
        .ok_or(UNSET)
}

pub struct DeepCapture {
    sampler: Result<Sampler, &'static str>,
    sizes: DeepSizes,
    recorded: bool,
}

pub struct DeepFiles {
    pub viewer: PathBuf,
    pub digest: PathBuf,
    pub complete: bool,
}

impl DeepCapture {
    pub fn start(threads: Vec<SampledThread>, sizes: DeepSizes) -> Self {
        {
            let mut heap = HEAP.lock().unwrap_or_else(PoisonError::into_inner);
            heap.samples = Vec::with_capacity(sizes.heap_samples);
            heap.limit = sizes.heap_samples;
            heap.dropped = 0;
        }
        HEAP_BUSY.store(0, Relaxed);
        engine_profile::sample_allocations(Some(sample_heap));
        engine_profile::keep(sizes.events);
        let sampler = if threads.is_empty() {
            Err("no_threads_named")
        } else {
            Sampler::start(
                threads,
                INTERVAL,
                engine::host::now_ms,
                sizes.thread_samples,
            )
            .ok_or("sampler_unavailable_on_this_host")
        };
        Self {
            sampler,
            sizes,
            recorded: engine_profile::recording(),
        }
    }

    pub fn finish(self, folder: &Path, stem: &str) -> Result<DeepFiles, Error> {
        let (events, events_dropped) = engine_profile::take_kept();
        let samples = self.sampler.map(Sampler::finish);
        let unsampled = samples.as_ref().err().copied();
        let samples = samples.ok();
        engine_profile::sample_allocations(None);
        let (heap, dropped) = {
            let mut heap = HEAP.lock().unwrap_or_else(PoisonError::into_inner);
            let dropped = heap.dropped + HEAP_BUSY.load(Relaxed);
            (std::mem::take(&mut heap.samples), dropped)
        };
        let symbols = Symbols::of(samples.as_ref(), &heap);
        let viewer = firefox_profile(&events, samples.as_ref(), &symbols);
        let mut digest = digest(samples.as_ref(), &symbols, self.sizes);
        digest["events"] = if self.recorded {
            json!({ "kept": events.len(), "dropped": events_dropped })
        } else {
            json!({ "unavailable": "no_record_running" })
        };
        if let Some(reason) = unsampled {
            digest["cpu_samples"] = json!({ "unavailable": reason });
        }
        digest["heap"] = heap_samples(&heap, dropped, &symbols, self.sizes);
        std::fs::create_dir_all(folder).map_err(|_| unwritable(folder))?;
        let files = DeepFiles {
            viewer: folder.join(format!("{stem}.firefox.json")),
            digest: folder.join(format!("{stem}.digest.json")),
            complete: samples.as_ref().is_some_and(|samples| {
                samples.threads.iter().all(|thread| thread.missed == 0)
            }) && dropped == 0
                && events_dropped == 0
                && self.recorded,
        };
        write(&files.viewer, &viewer)?;
        write(&files.digest, &digest)?;
        Ok(files)
    }
}

fn unwritable(path: &Path) -> Error {
    Error::new(ErrorCode::SourceIo, "$.folder")
        .expected(Expected::Writable)
        .got(path.display().to_string())
}

fn write(path: &Path, value: &dyn erased::Write) -> Result<(), Error> {
    let file = std::fs::File::create(path).map_err(|_| unwritable(path))?;
    value
        .write(&mut std::io::BufWriter::new(file))
        .map_err(|_| unwritable(path))
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
    cause: StringHandle,
    category: CategoryHandle,
}

impl StaticSchemaMarker for SpanMarker {
    const UNIQUE_MARKER_TYPE_NAME: &'static str = "EngineSpan";
    const CHART_LABEL: Option<&'static str> = Some("{marker.name}");
    const TABLE_LABEL: Option<&'static str> = Some("{marker.name} ({marker.data.kind})");
    const FIELDS: &'static [StaticSchemaMarkerField] = &[
        StaticSchemaMarkerField {
            key: "kind",
            label: "Kind",
            format: MarkerFieldFormat::String,
            flags: MarkerFieldFlags::SEARCHABLE,
        },
        StaticSchemaMarkerField {
            key: "cause",
            label: "Cause",
            format: MarkerFieldFormat::String,
            flags: MarkerFieldFlags::SEARCHABLE,
        },
    ];

    fn name(&self, _: &mut Profile) -> StringHandle {
        self.name
    }

    fn category(&self, _: &mut Profile) -> CategoryHandle {
        self.category
    }

    fn string_field_value(&self, field: u32) -> StringHandle {
        match field {
            1 => self.cause,
            _ => self.kind,
        }
    }

    fn number_field_value(&self, _: u32) -> f64 {
        0.0
    }
}

fn at(ms: f64) -> Timestamp {
    Timestamp::from_millis_since_reference(ms.max(0.0))
}

struct Spanned<'a> {
    start: f64,
    end: f64,
    depth: u32,
    cause: u32,
    category: &'static str,
    name: &'a str,
}

fn spans(events: &[Event]) -> Vec<Spanned<'_>> {
    let mut spans: Vec<Spanned<'_>> = events
        .iter()
        .filter_map(|event| match event.facts {
            Facts::Span {
                category,
                ms,
                depth,
            } => Some(Spanned {
                start: event.at_ms,
                end: event.at_ms + ms,
                depth,
                cause: event.cause,
                category,
                name: &event.name,
            }),
            Facts::Counter(_) | Facts::Listed { .. } => None,
        })
        .collect();
    spans.sort_by(|a, b| a.start.total_cmp(&b.start));
    spans
}

fn bounds(spans: &[Spanned<'_>], samples: Option<&Samples>) -> (f64, f64) {
    let times = spans
        .iter()
        .flat_map(|span| [span.start, span.end])
        .chain(
            samples
                .into_iter()
                .flat_map(|samples| &samples.threads)
                .flat_map(|thread| thread.samples.iter().map(|sample| sample.time_ms)),
        );
    times.fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), time| {
        (low.min(time), high.max(time))
    })
}

fn markers(
    profile: &mut Profile,
    spans: &[Spanned<'_>],
    (threads, main): (&[(&str, ThreadHandle)], ThreadHandle),
    category: CategoryHandle,
) {
    for span in spans {
        let on = threads
            .iter()
            .find(|(name, _)| *name == span.category)
            .map_or(main, |&(_, handle)| handle);
        let cause = match span.cause {
            0 => String::new(),
            cause => cause.to_string(),
        };
        let marker = SpanMarker {
            name: profile.intern_string(span.name),
            kind: profile.intern_string(span.category),
            cause: profile.intern_string(&cause),
            category,
        };
        profile.add_marker(
            on,
            MarkerTiming::Interval(at(span.start), at(span.end)),
            marker,
        );
    }
}

fn firefox_profile(events: &[Event], samples: Option<&Samples>, symbols: &Symbols) -> Profile {
    let now = engine::host::now_ms();
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |since| since.as_secs_f64() * 1000.0)
        - now;
    let interval = samples.map_or(INTERVAL, |samples| {
        Duration::from_secs_f64(samples.interval_ms / 1000.0)
    });
    let mut profile = Profile::new(
        "engine_native",
        ReferenceTimestamp::from_millis_since_unix_epoch(epoch),
        SamplingInterval::from_nanos(interval.as_nanos() as u64),
    );
    profile.set_os_name(std::env::consts::OS);
    let spans = spans(events);
    let (start, end) = bounds(&spans, samples);
    let start = if start.is_finite() { start } else { now };
    let process = profile.add_process("engine_native", std::process::id(), at(start));
    let native = profile.add_category("Native code", CategoryColor::Blue);
    let engine_category = profile.add_category("Engine spans", CategoryColor::Green);
    let mut main = None;
    let mut threads: Vec<(&str, ThreadHandle)> = Vec::new();
    for (index, thread) in samples
        .into_iter()
        .flat_map(|samples| &samples.threads)
        .enumerate()
    {
        let handle = profile.add_thread(process, index as u32 + 1, at(start), index == 0);
        profile.set_thread_name(handle, &thread.name);
        threads.push((&thread.name, handle));
        if index == 0 {
            profile.add_initial_selected_thread(handle);
            main = Some(handle);
        }
        profile.add_initial_visible_thread(handle);
        for sample in &thread.samples {
            let frames: Vec<FrameInfo> = functions(&sample.stack, symbols)
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
    let track = profile.add_thread(process, 0, at(start), false);
    profile.set_thread_name(track, "Engine spans");
    profile.add_initial_visible_thread(track);
    markers(
        &mut profile,
        &spans,
        (&threads, main.unwrap_or(track)),
        engine_category,
    );
    let stacked: Vec<&Spanned<'_>> = spans
        .iter()
        .filter(|span| !threads.iter().any(|(name, _)| *name == span.category))
        .collect();
    let step = interval.as_secs_f64() * 1000.0;
    let mut active: Vec<&Spanned<'_>> = Vec::new();
    let mut next = 0;
    let mut time = start;
    while time < end {
        while next < stacked.len() && stacked[next].start <= time {
            active.push(stacked[next]);
            next += 1;
        }
        active.retain(|span| span.end > time);
        if !active.is_empty() {
            let mut stack = active.clone();
            stack.sort_by_key(|span| span.depth);
            let frames: Vec<FrameInfo> = stack
                .iter()
                .map(|span| FrameInfo {
                    frame: Frame::Label(profile.intern_string(span.name)),
                    category_pair: engine_category.into(),
                    flags: FrameFlags::empty(),
                })
                .collect();
            let stack = profile.intern_stack_frames(track, frames.into_iter());
            profile.add_sample(track, at(time), stack, CpuDelta::from_millis(step), 1);
        }
        time += step;
    }
    let mut counters: BTreeMap<&str, Vec<(f64, f64)>> = BTreeMap::new();
    for event in events {
        if let Facts::Counter(value) = event.facts {
            counters
                .entry(&event.name)
                .or_default()
                .push((event.at_ms, value));
        }
    }
    for (name, values) in counters {
        let counter = profile.add_counter(process, name, "Engine", name);
        let mut last = 0.0;
        for (time, value) in values {
            profile.add_counter_sample(counter, at(time), value - last, 1);
            last = value;
        }
    }
    profile
}

fn heap_samples(samples: &[HeapSample], dropped: u64, symbols: &Symbols, sizes: DeepSizes) -> Value {
    let mut stacks: HashMap<Vec<String>, (u64, u64)> = HashMap::default();
    let mut bytes = 0;
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
        bytes += sample.bytes as u64;
        let (stack_count, stack_bytes) = stacks.entry(callers).or_default();
        *stack_count += 1;
        *stack_bytes += sample.bytes as u64;
    }
    let mut stacks: Vec<(Vec<String>, (u64, u64))> = stacks.into_iter().collect();
    stacks.sort_by(|a, b| b.1.1.cmp(&a.1.1).then_with(|| a.0.cmp(&b.0)));
    json!({
        "every": engine_profile::SAMPLE_EVERY,
        "always_at_bytes": engine_profile::ALWAYS_SAMPLED_BYTES,
        "samples": samples.len(),
        "dropped": dropped,
        "sampled_bytes": bytes,
        "stacks": stacks
            .into_iter()
            .take(sizes.rows)
            .map(|(frames, (samples, bytes))| json!({
                "samples": samples,
                "sampled_bytes": bytes,
                "frames": frames,
            }))
            .collect::<Vec<_>>(),
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

fn digest(samples: Option<&Samples>, symbols: &Symbols, sizes: DeepSizes) -> Value {
    let mut threads = Vec::new();
    for thread in samples.into_iter().flat_map(|samples| &samples.threads) {
        let mut own: HashMap<String, u64> = HashMap::default();
        let mut inclusive: HashMap<String, u64> = HashMap::default();
        for sample in &thread.samples {
            let names = functions(&sample.stack, symbols);
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
            "self": ranked(own, total, sizes.rows),
            "total": ranked(inclusive, total, sizes.rows),
        }));
    }
    let main = samples.and_then(|samples| samples.threads.first());
    let mut lines: HashMap<String, u64> = HashMap::default();
    for sample in main.into_iter().flat_map(|thread| &thread.samples) {
        let Some(&leaf) = sample.stack.first() else {
            continue;
        };
        if let Some(frame) = symbols
            .get(leaf, true)
            .and_then(|resolved| resolved.frames.first())
        {
            let line = match (&frame.file, frame.line) {
                (Some(file), Some(line)) => format!("{file}:{line} ({})", frame.function),
                _ => frame.function.clone(),
            };
            *lines.entry(line).or_default() += 1;
        }
    }
    let main_total = main.map_or(0, |thread| thread.samples.len() as u64);
    json!({
        "sizes": {
            "heap_samples": sizes.heap_samples,
            "thread_samples": sizes.thread_samples,
            "events": sizes.events,
            "rows": sizes.rows,
        },
        "cpu_samples": samples.map(|samples| json!({
            "method": samples.method,
            "interval_ms": samples.interval_ms,
            "symbols": "in_process",
        })),
        "threads": threads,
        "hot_lines": ranked(lines, main_total, sizes.rows),
    })
}
