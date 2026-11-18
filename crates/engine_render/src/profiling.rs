#[derive(Clone, Copy)]
pub struct Span<'a> {
    query_set: &'a wgpu::QuerySet,
    begin: u32,
}

impl Span<'_> {
    pub fn mark(self, encoder: &mut wgpu::CommandEncoder, end: bool) {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/profile-timestamp"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: self.query_set,
                beginning_of_pass_write_index: (!end).then_some(self.begin),
                end_of_pass_write_index: end.then_some(self.begin + 1),
            }),
        });
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DrawKey {
    pub pass: &'static str,
    pub part: &'static str,
    pub pipeline: &'static str,
}

#[cfg(feature = "profiling")]
pub const FEATURES: wgpu::Features = wgpu::Features::PIPELINE_STATISTICS_QUERY;

#[cfg(any(feature = "dev", feature = "profiling"))]
pub const INSTRUMENTS: [&str; 4] = [
    "gpu.passes",
    "gpu.passes.split",
    "gpu.draws.shaders",
    crate::linker::counting::INSTRUMENT,
];

#[cfg(any(feature = "dev", feature = "profiling"))]
const MARKS: &str = "adds an empty compute pass with two time stamps at each group and batch boundary";

#[cfg(any(feature = "dev", feature = "profiling"))]
const SPLIT: &str = concat!(
    "pass merging off, so every group of passes is its own render pass; ",
    "adds an empty compute pass with two time stamps at each group and batch boundary"
);

#[cfg(any(feature = "dev", feature = "profiling"))]
pub use timer::GpuProfiler;

#[cfg(not(any(feature = "dev", feature = "profiling")))]
pub struct GpuProfiler;

#[cfg(not(any(feature = "dev", feature = "profiling")))]
impl GpuProfiler {
    pub fn new(_features: wgpu::Features, _period_ns: f64) -> Self {
        Self
    }

    #[inline(always)]
    pub fn begin_draw(
        &self,
        _render: &mut wgpu::RenderPass<'_>,
        _key: impl FnOnce() -> DrawKey,
    ) -> bool {
        false
    }

    #[inline(always)]
    pub fn end_draw(&self, _render: &mut wgpu::RenderPass<'_>, _began: bool) {}

    #[inline(always)]
    pub fn begin_frame(&mut self, _device: &wgpu::Device, _settings: &crate::RenderSettings) {}

    #[inline(always)]
    pub fn splits(&self) -> bool {
        false
    }

    #[inline(always)]
    pub fn ran(&self, _span: Option<&'static str>, _pass: &'static str) {}

    #[inline(always)]
    pub fn span(&self, _name: &'static str) -> Option<Span<'_>> {
        None
    }

    #[inline(always)]
    pub fn span_in(&self, _group: &'static str, _name: &'static str) -> Option<Span<'_>> {
        None
    }

    #[inline(always)]
    pub fn busy(&self, _name: &'static str) -> Option<Span<'_>> {
        None
    }

    #[inline(always)]
    pub(crate) fn end_frame(
        &mut self,
        _recording: (&wgpu::Device, &mut wgpu::CommandEncoder),
        _readbacks: &mut crate::readback::Readbacks,
        _written: engine_core::FrameId,
    ) {
    }

    #[inline(always)]
    pub(crate) fn arrived(&mut self, _slot: u32, _arrival: crate::readback::Arrival<'_>) {}
}

#[cfg(any(feature = "dev", feature = "profiling"))]
mod timer {
    use std::cell::{Cell, RefCell};

    use engine_core::FrameId;
    use serde_json::{Value, json};

    use super::{DrawKey, Span};
    use crate::readback::{Arrival, Keep, Readbacks, Reader};

    const BUSY: &str = "busy";
    const STATISTICS: wgpu::PipelineStatisticsTypes =
        wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
            .union(wgpu::PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT)
            .union(wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS);
    const RUNS: usize = 3;

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct Sizes {
        slots: usize,
        spans: u32,
        draws: u32,
    }

    impl Sizes {
        fn of(settings: &crate::RenderSettings) -> Self {
            Self {
                slots: settings.timer_slots.max(1) as usize,
                spans: settings.timer_spans.max(1),
                draws: settings.timer_draws.max(1),
            }
        }

        fn draws_at(self) -> u64 {
            (2 * u64::from(self.spans) * 8).next_multiple_of(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT)
        }

        fn bytes(self) -> u64 {
            self.draws_at() + u64::from(self.draws) * RUNS as u64 * 8
        }
    }

    struct Slot {
        queries: wgpu::QuerySet,
        statistics: Option<wgpu::QuerySet>,
        resolve: wgpu::Buffer,
        written: Option<FrameId>,
        spans: RefCell<Vec<(&'static str, Option<&'static str>)>>,
        ran: RefCell<Vec<(usize, &'static str)>>,
        draws: RefCell<Vec<DrawKey>>,
    }

    impl Slot {
        fn new(device: &wgpu::Device, sizes: Sizes, statistics: bool) -> Self {
            Self {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("engine/profile-timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2 * sizes.spans,
                }),
                statistics: statistics.then(|| {
                    device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("engine/profile-draw-statistics"),
                        ty: wgpu::QueryType::PipelineStatistics(STATISTICS),
                        count: sizes.draws,
                    })
                }),
                resolve: crate::gpu_calls::create_buffer(device, &wgpu::BufferDescriptor {
                    label: Some("engine/profile-resolve"),
                    size: sizes.bytes(),
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                written: None,
                spans: RefCell::new(Vec::new()),
                ran: RefCell::new(Vec::new()),
                draws: RefCell::new(Vec::new()),
            }
        }
    }

    struct Timed {
        name: &'static str,
        of: Option<&'static str>,
        passes: Vec<&'static str>,
        frames: u64,
        ns: u64,
    }

    struct Counted {
        key: DrawKey,
        draws: u64,
        runs: [u64; RUNS],
    }

    #[derive(Default)]
    struct Switches {
        passes: bool,
        split: bool,
        shaders: bool,
        timed_once: bool,
        counted_once: bool,
    }

    pub struct GpuProfiler {
        timestamps: bool,
        statistics: bool,
        period_ns: f64,
        on: Switches,
        sizes: Sizes,
        slots: Vec<Slot>,
        current: Option<usize>,
        through: Option<u64>,
        frames: u64,
        no_slot: u64,
        failed: u64,
        spans_dropped: Cell<u64>,
        draws_dropped: Cell<u64>,
        timed: Vec<Timed>,
        counted: Vec<Counted>,
    }

    impl GpuProfiler {
        pub fn new(features: wgpu::Features, period_ns: f64) -> Self {
            Self {
                timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
                statistics: features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY),
                period_ns,
                on: Switches::default(),
                sizes: Sizes {
                    slots: 0,
                    spans: 0,
                    draws: 0,
                },
                slots: Vec::new(),
                current: None,
                through: None,
                frames: 0,
                no_slot: 0,
                failed: 0,
                spans_dropped: Cell::new(0),
                draws_dropped: Cell::new(0),
                timed: Vec::new(),
                counted: Vec::new(),
            }
        }

        pub fn switch(&mut self, name: &str, on: bool) -> Result<&'static str, &'static str> {
            match name {
                "gpu.passes" | "gpu.passes.split" => {
                    if on && !self.timestamps {
                        return Err("timestamp_query_unavailable");
                    }
                    self.on.timed_once |= on;
                    if name == "gpu.passes" {
                        self.on.passes = on;
                        Ok(super::MARKS)
                    } else {
                        self.on.split = on;
                        Ok(super::SPLIT)
                    }
                }
                "gpu.draws.shaders" => {
                    if on && !cfg!(feature = "profiling") {
                        return Err("profiling_build_only");
                    }
                    if on && !self.statistics {
                        return Err("pipeline_statistics_query_unavailable");
                    }
                    self.on.shaders = on;
                    self.on.counted_once |= on;
                    Ok("every draw is wrapped in a pipeline statistics query")
                }
                _ => Err("unknown_instrument"),
            }
        }

        fn timing(&self) -> bool {
            self.on.passes || self.on.split
        }

        pub fn splits(&self) -> bool {
            self.on.split
        }

        pub fn begin_frame(&mut self, device: &wgpu::Device, settings: &crate::RenderSettings) {
            self.current = None;
            if !self.timing() && !self.on.shaders {
                return;
            }
            let sizes = Sizes::of(settings);
            if sizes != self.sizes && self.slots.iter().all(|slot| slot.written.is_none()) {
                self.sizes = sizes;
                self.slots = (0..sizes.slots)
                    .map(|_| Slot::new(device, sizes, self.statistics))
                    .collect();
            }
            let Some(index) = self.slots.iter().position(|slot| slot.written.is_none()) else {
                self.no_slot += 1;
                return;
            };
            let slot = &mut self.slots[index];
            slot.spans.get_mut().clear();
            slot.ran.get_mut().clear();
            slot.draws.get_mut().clear();
            self.current = Some(index);
        }

        pub fn span(&self, name: &'static str) -> Option<Span<'_>> {
            self.recorded(None, name)
        }

        pub fn span_in(&self, group: &'static str, name: &'static str) -> Option<Span<'_>> {
            self.recorded(Some(group), name)
        }

        pub fn busy(&self, name: &'static str) -> Option<Span<'_>> {
            self.recorded(Some(BUSY), name)
        }

        fn recorded(&self, group: Option<&'static str>, name: &'static str) -> Option<Span<'_>> {
            if !self.timing() {
                return None;
            }
            let slot = &self.slots[self.current?];
            let mut spans = slot.spans.borrow_mut();
            let index = match spans.iter().position(|&(span, _)| span == name) {
                Some(index) => index,
                None if spans.len() < self.sizes.spans as usize => {
                    spans.push((name, group));
                    spans.len() - 1
                }
                None => {
                    self.spans_dropped.set(self.spans_dropped.get() + 1);
                    return None;
                }
            };
            Some(Span {
                query_set: &slot.queries,
                begin: 2 * index as u32,
            })
        }

        pub fn ran(&self, span: Option<&'static str>, pass: &'static str) {
            let (Some(current), Some(span)) = (self.current, span) else {
                return;
            };
            let slot = &self.slots[current];
            let Some(at) = slot.spans.borrow().iter().position(|&(name, _)| name == span) else {
                return;
            };
            let mut ran = slot.ran.borrow_mut();
            if !ran.contains(&(at, pass)) {
                ran.push((at, pass));
            }
        }

        pub fn begin_draw(
            &self,
            render: &mut wgpu::RenderPass<'_>,
            key: impl FnOnce() -> DrawKey,
        ) -> bool {
            if !self.on.shaders {
                return false;
            }
            let Some(slot) = self.current.map(|current| &self.slots[current]) else {
                return false;
            };
            let Some(statistics) = &slot.statistics else {
                return false;
            };
            let mut draws = slot.draws.borrow_mut();
            if draws.len() >= self.sizes.draws as usize {
                self.draws_dropped.set(self.draws_dropped.get() + 1);
                return false;
            }
            render.begin_pipeline_statistics_query(statistics, draws.len() as u32);
            draws.push(key());
            true
        }

        pub fn end_draw(&self, render: &mut wgpu::RenderPass<'_>, began: bool) {
            if began {
                render.end_pipeline_statistics_query();
            }
        }

        pub(crate) fn end_frame(
            &mut self,
            (device, encoder): (&wgpu::Device, &mut wgpu::CommandEncoder),
            readbacks: &mut Readbacks,
            written: FrameId,
        ) {
            let Some(index) = self.current.take() else {
                return;
            };
            let sizes = self.sizes;
            let slot = &mut self.slots[index];
            let stamps = 2 * slot.spans.get_mut().len() as u32;
            let draws = slot.draws.get_mut().len() as u32;
            if stamps == 0 && draws == 0 {
                return;
            }
            if stamps != 0 {
                encoder.resolve_query_set(&slot.queries, 0..stamps, &slot.resolve, 0);
            }
            if let Some(statistics) = slot.statistics.as_ref().filter(|_| draws != 0) {
                encoder.resolve_query_set(statistics, 0..draws, &slot.resolve, sizes.draws_at());
            }
            let recorded = readbacks.record(
                (device, encoder),
                (&slot.resolve, 0..sizes.bytes()),
                (Reader::Profiler(index as u32), written, None),
                Keep::Ring(1),
            );
            if recorded {
                slot.written = Some(written);
            } else {
                self.failed += 1;
            }
        }

        pub(crate) fn arrived(&mut self, index: u32, arrival: Arrival<'_>) {
            let Self {
                slots,
                timed,
                counted,
                sizes,
                period_ns,
                frames,
                failed,
                through,
                ..
            } = self;
            let Some(slot) = slots.get_mut(index as usize) else {
                return;
            };
            let Some(written) = slot.written.take() else {
                return;
            };
            let Arrival::Read(bytes) = arrival else {
                *failed += 1;
                return;
            };
            let word = |at: usize| {
                bytes
                    .get(at..at + 8)
                    .and_then(|word| word.try_into().ok())
                    .map_or(0, u64::from_le_bytes)
            };
            let spans = slot.spans.get_mut();
            let ran = slot.ran.get_mut();
            for (at, &(name, of)) in spans.iter().enumerate() {
                let (begin, end) = (word(16 * at), word(16 * at + 8));
                if begin == 0 || end < begin {
                    continue;
                }
                let ns = ((end - begin) as f64 * *period_ns).round() as u64;
                let found = match timed.iter().position(|known| known.name == name) {
                    Some(found) => found,
                    None => {
                        timed.push(Timed {
                            name,
                            of,
                            passes: Vec::new(),
                            frames: 0,
                            ns: 0,
                        });
                        timed.len() - 1
                    }
                };
                let entry = &mut timed[found];
                entry.frames += 1;
                entry.ns += ns;
                for &(_, pass) in ran.iter().filter(|(span, _)| *span == at) {
                    if !entry.passes.contains(&pass) {
                        entry.passes.push(pass);
                    }
                }
            }
            let draws_at = sizes.draws_at() as usize;
            for (at, &key) in slot.draws.get_mut().iter().enumerate() {
                let runs: [u64; RUNS] =
                    std::array::from_fn(|run| word(draws_at + (at * RUNS + run) * 8));
                let found = match counted.iter().position(|known| known.key == key) {
                    Some(found) => found,
                    None => {
                        counted.push(Counted {
                            key,
                            draws: 0,
                            runs: [0; RUNS],
                        });
                        counted.len() - 1
                    }
                };
                let entry = &mut counted[found];
                entry.draws += 1;
                for (held, run) in entry.runs.iter_mut().zip(runs) {
                    *held += run;
                }
            }
            if !spans.is_empty() {
                *frames += 1;
            }
            *through = Some(through.map_or(written.0, |through| through.max(written.0)));
        }


        fn unavailable(&self, timing: bool) -> Option<&'static str> {
            if timing {
                return (!self.timestamps)
                    .then_some("timestamp_query_unavailable")
                    .or((!self.on.timed_once).then_some("never_switched_on"));
            }
            if !cfg!(feature = "profiling") {
                return Some("profiling_build_only");
            }
            (!self.statistics)
                .then_some("pipeline_statistics_query_unavailable")
                .or((!self.on.counted_once).then_some("never_switched_on"))
        }

        pub fn read(&self, name: &str) -> Value {
            let timing = name.starts_with("gpu.passes.");
            if let Some(reason) = self.unavailable(timing) {
                return json!({ "unavailable": reason });
            }
            let through = self.through.map_or_else(
                || json!({ "unavailable": "nothing_arrived" }),
                |through| json!(through),
            );
            let timed = |value: fn(&Timed) -> Value| -> Value {
                Value::Object(
                    self.timed
                        .iter()
                        .map(|timed| (timed.name.to_owned(), value(timed)))
                        .collect(),
                )
            };
            let counted = |value: fn(&Counted) -> u64| -> Value {
                Value::Object(
                    self.counted
                        .iter()
                        .map(|counted| {
                            let DrawKey {
                                pass,
                                part,
                                pipeline,
                            } = counted.key;
                            (format!("{pass}/{part}/{pipeline}"), json!(value(counted)))
                        })
                        .collect(),
                )
            };
            match name {
                "gpu.passes.time" => timed(|timed| json!(timed.ns)),
                "gpu.passes.timed" => timed(|timed| json!(timed.frames)),
                "gpu.passes.groups" => json!({
                    "on": self.timing(),
                    "split": self.on.split,
                    "period_ns": self.period_ns,
                    "through_frame": through,
                    "spans": timed(|timed| json!({"in": timed.of, "passes": timed.passes})),
                }),
                "gpu.passes.frames" => json!({
                    "timed": self.frames,
                    "without_slot": self.no_slot,
                    "failed": self.failed,
                    "spans_dropped": self.spans_dropped.get(),
                }),
                "gpu.draws.vertex_runs" => counted(|counted| counted.runs[0]),
                "gpu.draws.clipped" => counted(|counted| counted.runs[1]),
                "gpu.draws.fragment_runs" => counted(|counted| counted.runs[2]),
                "gpu.draws.measured" => counted(|counted| counted.draws),
                "gpu.draws.unmeasured" => json!(self.draws_dropped.get()),
                "gpu.draws.measuring" => json!({
                    "on": self.on.shaders,
                    "through_frame": through,
                }),
                _ => json!({ "unavailable": "unknown_name" }),
            }
        }
    }
}
