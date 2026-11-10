#[derive(Clone, Copy)]
pub struct Span<'a> {
    query_set: &'a wgpu::QuerySet,
    begin: u32,
}

impl<'a> Span<'a> {
    pub fn compute_part(self, begin: bool, end: bool) -> wgpu::ComputePassTimestampWrites<'a> {
        wgpu::ComputePassTimestampWrites {
            query_set: self.query_set,
            beginning_of_pass_write_index: begin.then_some(self.begin),
            end_of_pass_write_index: end.then_some(self.begin + 1),
        }
    }

    pub fn mark(self, encoder: &mut wgpu::CommandEncoder, end: bool) {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/profile-timestamp"),
            timestamp_writes: Some(self.compute_part(!end, end)),
        });
    }
}

#[cfg_attr(
    not(feature = "profiling"),
    expect(dead_code, reason = "only profiling builds time draws")
)]
pub struct Draw(u32);

#[cfg(feature = "profiling")]
pub const FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY
    .union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    .union(wgpu::Features::PIPELINE_STATISTICS_QUERY);

#[cfg(feature = "profiling")]
pub use capture::GpuProfiler;

#[cfg(not(feature = "profiling"))]
pub struct GpuProfiler;

#[cfg(not(feature = "profiling"))]
impl GpuProfiler {
    pub fn new(_features: wgpu::Features, _period_ns: f64) -> Self {
        Self
    }

    #[inline(always)]
    pub fn begin_draw(
        &self,
        _pass: &mut wgpu::RenderPass<'_>,
        _label: impl FnOnce() -> String,
    ) -> Option<Draw> {
        None
    }

    #[inline(always)]
    pub fn end_draw(&self, _pass: &mut wgpu::RenderPass<'_>, _draw: Option<Draw>) {}

    #[inline(always)]
    pub fn begin_frame(&mut self, _device: &wgpu::Device) {}

    #[inline(always)]
    pub fn capturing(&self) -> bool {
        false
    }

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

#[cfg(feature = "profiling")]
mod capture {
    use std::cell::{Cell, RefCell};

    use engine_core::FrameId;
    use serde_json::{Value, json};

    use super::{Draw, Span};
    use crate::readback::{Arrival, Keep, Readbacks, Reader};

    const SLOTS: usize = 3;
    const SAMPLE_LIMIT: usize = 4_096;
    const SPANS: usize = 128;
    const QUERIES: u32 = 2 * SPANS as u32;
    const QUERY_BYTES: u64 = QUERIES as u64 * 8;
    const DRAWS: usize = 2_048;
    const STATISTICS: wgpu::PipelineStatisticsTypes =
        wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
            .union(wgpu::PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT)
            .union(wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS);
    const DRAW_STAMPS_AT: u64 = QUERY_BYTES;
    const DRAW_STATISTICS_AT: u64 = DRAW_STAMPS_AT + 2 * DRAWS as u64 * 8;
    const SLOT_BYTES: u64 = DRAW_STATISTICS_AT + DRAWS as u64 * 24;
    const OPAQUE: &str = "opaque_scene";
    const BUSY: &str = "busy";

    #[derive(Clone, Copy)]
    struct Origin {
        capture: u64,
        frame: u64,
        submission: u64,
    }

    struct Slot {
        queries: wgpu::QuerySet,
        resolve: wgpu::Buffer,
        origin: Option<Origin>,
        spans: RefCell<Vec<(&'static str, Option<&'static str>)>>,
        draw_stamps: Option<wgpu::QuerySet>,
        draw_statistics: Option<wgpu::QuerySet>,
        draws: RefCell<Vec<String>>,
    }

    impl Slot {
        fn new(device: &wgpu::Device, draws: Support) -> Self {
            let draw_stamps = draws.timestamps.then(|| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("engine/profile-draw-timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2 * DRAWS as u32,
                })
            });
            let draw_statistics = draws.statistics.then(|| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("engine/profile-draw-statistics"),
                    ty: wgpu::QueryType::PipelineStatistics(STATISTICS),
                    count: DRAWS as u32,
                })
            });
            Self {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("engine/profile-timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: QUERIES,
                }),
                resolve: crate::memory::buffer(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("engine/profile-resolve"),
                        size: SLOT_BYTES,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    },
                ),
                origin: None,
                spans: RefCell::new(Vec::with_capacity(SPANS)),
                draw_stamps,
                draw_statistics,
                draws: RefCell::new(Vec::new()),
            }
        }
    }

    fn read(bytes: &[u8], spans: usize, draws: usize) -> Readback {
        let word = |at: usize| {
            bytes
                .get(at..at + 8)
                .and_then(|word| word.try_into().ok())
                .map_or(0, u64::from_le_bytes)
        };
        let stamp = |query: usize| word(query * 8);
        let draw_stamp = |query: usize| word(DRAW_STAMPS_AT as usize + query * 8);
        let statistic = |draw: usize, counter: usize| {
            word(DRAW_STATISTICS_AT as usize + draw * 24 + counter * 8)
        };
        (
            (0..spans)
                .map(|span| (stamp(2 * span), stamp(2 * span + 1)))
                .collect(),
            (0..draws)
                .map(|draw| (draw_stamp(2 * draw), draw_stamp(2 * draw + 1)))
                .collect(),
            (0..draws)
                .map(|draw| [0, 1, 2].map(|counter| statistic(draw, counter)))
                .collect(),
        )
    }

    fn idle(
        recorded: &[(&'static str, Option<&'static str>)],
        stamps: &[(u64, u64)],
        (first, last): (u64, u64),
    ) -> Option<u64> {
        let mut busy: Vec<(u64, u64)> = recorded
            .iter()
            .zip(stamps)
            .filter(|((_, group), _)| *group == Some(BUSY))
            .map(|(_, &stamps)| stamps)
            .filter(|&(begin, end)| begin != 0 && end >= begin)
            .collect();
        busy.sort_unstable();
        let (mut reach, mut covered) = (first, 0);
        for &(begin, end) in &busy {
            covered += end.saturating_sub(begin.max(reach));
            reach = reach.max(end);
        }
        (!busy.is_empty()).then(|| last.saturating_sub(first).saturating_sub(covered))
    }

    #[derive(Clone, Copy)]
    struct Support {
        timestamps: bool,
        statistics: bool,
    }

    struct DrawSample {
        label: String,
        start_ms: Option<f64>,
        duration_ms: Option<f64>,
        statistics: Option<[u64; 3]>,
    }

    type Readback = (Vec<(u64, u64)>, Vec<(u64, u64)>, Vec<[u64; 3]>);

    struct Sample {
        origin: Origin,
        spans: Vec<(&'static str, f64)>,
        starts: Vec<(&'static str, f64)>,
        draws: Vec<DrawSample>,
    }

    pub struct Latest {
        pub frame_ms: f64,
        pub idle_ms: f64,
        pub spans: Vec<(&'static str, f64)>,
    }

    impl Latest {
        fn of(
            frame_ms: f64,
            idle_ms: Option<f64>,
            names: &[&'static str],
            stamps: &[(u64, u64)],
            duration: impl Fn((u64, u64)) -> Option<f64>,
        ) -> Self {
            Self {
                frame_ms,
                idle_ms: idle_ms.unwrap_or(0.0),
                spans: names
                    .iter()
                    .zip(stamps)
                    .filter_map(|(&name, &stamps)| Some((name, duration(stamps)?)))
                    .collect(),
            }
        }
    }

    pub struct GpuProfiler {
        supported: bool,
        draw_support: Support,
        period_ns: f64,
        slots: Vec<Slot>,
        current: Option<usize>,
        drawing: bool,
        latest: Option<Latest>,
        capture: Option<u64>,
        submission: u64,
        attempted: u64,
        submitted: u64,
        dropped: u64,
        spans_dropped: Cell<u64>,
        draws_dropped: Cell<u64>,
        errors: u64,
        sample_limit_reached: bool,
        samples: Vec<Sample>,
        last_span: Option<(u64, u64, u64)>,
    }

    impl GpuProfiler {
        pub fn new(features: wgpu::Features, period_ns: f64) -> Self {
            let supported = features.contains(wgpu::Features::TIMESTAMP_QUERY);
            Self {
                supported,
                draw_support: Support {
                    timestamps: supported
                        && features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES),
                    statistics: features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY),
                },
                period_ns,
                slots: Vec::new(),
                current: None,
                drawing: false,
                latest: None,
                capture: None,
                submission: 0,
                attempted: 0,
                submitted: 0,
                dropped: 0,
                spans_dropped: Cell::new(0),
                draws_dropped: Cell::new(0),
                errors: 0,
                sample_limit_reached: false,
                samples: Vec::new(),
                last_span: None,
            }
        }

        pub fn begin(&mut self, device: &wgpu::Device, capture: u64) {
            self.capture = Some(capture);
            self.attempted = 0;
            self.submitted = 0;
            self.dropped = 0;
            self.spans_dropped.set(0);
            self.draws_dropped.set(0);
            self.errors = 0;
            self.sample_limit_reached = false;
            self.samples.clear();
            if self.supported {
                self.make_slots(device);
                self.samples.reserve_exact(SAMPLE_LIMIT);
            }
        }

        fn make_slots(&mut self, device: &wgpu::Device) {
            if !self.slots.is_empty() {
                return;
            }
            let timed = self.draw_support;
            let untimed = Support {
                timestamps: false,
                statistics: false,
            };
            self.slots = (0..2 * SLOTS)
                .map(|index| Slot::new(device, if index < SLOTS { timed } else { untimed }))
                .collect();
        }

        pub fn begin_frame(&mut self, device: &wgpu::Device) {
            self.current = None;
            self.drawing = false;
            self.submission = self.submission.saturating_add(1);
            let capture = engine_profile::capture_id();
            if let Some(capture) = capture {
                if self.capture != Some(capture) {
                    self.begin(device, capture);
                }
                self.attempted = self.attempted.saturating_add(1);
            }
            if !self.supported {
                return;
            }
            self.make_slots(device);
            if capture.is_some() && self.submitted as usize >= SAMPLE_LIMIT {
                self.sample_limit_reached = true;
                self.dropped = self.dropped.saturating_add(1);
                return;
            }
            let mut pool = if capture.is_some() {
                0..SLOTS
            } else {
                SLOTS..2 * SLOTS
            };
            let Some(index) = pool.find(|&index| self.slots[index].origin.is_none()) else {
                if capture.is_some() {
                    self.dropped = self.dropped.saturating_add(1);
                }
                return;
            };
            let slot = &mut self.slots[index];
            slot.origin = Some(Origin {
                capture: capture.unwrap_or(0),
                frame: engine_profile::current_frame(),
                submission: self.submission,
            });
            slot.spans.get_mut().clear();
            slot.draws.get_mut().clear();
            if capture.is_some() {
                self.submitted += 1;
                self.drawing = true;
            }
            self.current = Some(index);
        }

        pub fn latest(&self) -> Option<&Latest> {
            self.latest.as_ref()
        }

        pub fn capturing(&self) -> bool {
            self.current.is_some()
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
            let slot = &self.slots[self.current?];
            let mut spans = slot.spans.borrow_mut();
            let index = match spans.iter().position(|&(span, _)| span == name) {
                Some(index) => index,
                None if spans.len() < SPANS => {
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

        pub fn begin_draw(
            &self,
            pass: &mut wgpu::RenderPass<'_>,
            label: impl FnOnce() -> String,
        ) -> Option<Draw> {
            let slot = &self.slots[self.current?];
            if !self.drawing || (slot.draw_stamps.is_none() && slot.draw_statistics.is_none()) {
                return None;
            }
            let mut draws = slot.draws.borrow_mut();
            if draws.len() >= DRAWS {
                self.draws_dropped.set(self.draws_dropped.get() + 1);
                return None;
            }
            let index = draws.len() as u32;
            draws.push(label());
            if let Some(stamps) = &slot.draw_stamps {
                pass.write_timestamp(stamps, 2 * index);
            }
            if let Some(statistics) = &slot.draw_statistics {
                pass.begin_pipeline_statistics_query(statistics, index);
            }
            Some(Draw(index))
        }

        pub fn end_draw(&self, pass: &mut wgpu::RenderPass<'_>, draw: Option<Draw>) {
            let (Some(Draw(index)), Some(current)) = (draw, self.current) else {
                return;
            };
            let slot = &self.slots[current];
            if slot.draw_statistics.is_some() {
                pass.end_pipeline_statistics_query();
            }
            if let Some(stamps) = &slot.draw_stamps {
                pass.write_timestamp(stamps, 2 * index + 1);
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
            let slot = &mut self.slots[index];
            let queries = 2 * slot.spans.borrow().len() as u32;
            if queries != 0 {
                encoder.resolve_query_set(&slot.queries, 0..queries, &slot.resolve, 0);
            }
            let draws = slot.draws.borrow().len() as u32;
            if draws != 0 {
                if let Some(stamps) = &slot.draw_stamps {
                    encoder.resolve_query_set(stamps, 0..2 * draws, &slot.resolve, DRAW_STAMPS_AT);
                }
                if let Some(statistics) = &slot.draw_statistics {
                    encoder.resolve_query_set(
                        statistics,
                        0..draws,
                        &slot.resolve,
                        DRAW_STATISTICS_AT,
                    );
                }
            }
            let recorded = readbacks.record(
                (device, encoder),
                (&slot.resolve, 0..SLOT_BYTES),
                (Reader::Profiler(index as u32), written, None),
                Keep::Ring(1),
            );
            if !recorded {
                slot.origin = None;
                self.dropped = self.dropped.saturating_add(1);
            }
        }

        #[expect(
            clippy::too_many_lines,
            reason = "longer than the line ceiling; it may only shrink"
        )]
        pub(crate) fn arrived(&mut self, index: u32, arrival: Arrival<'_>) {
            let Some(slot) = self.slots.get_mut(index as usize) else {
                return;
            };
            let Some(origin) = slot.origin.take() else {
                return;
            };
            let Arrival::Read(bytes) = arrival else {
                if Some(origin.capture) == self.capture {
                    self.errors = self.errors.saturating_add(1);
                }
                return;
            };
            let recorded = slot.spans.get_mut().clone();
            let labels = std::mem::take(slot.draws.get_mut());
            let (timed, counted) = (slot.draw_stamps.is_some(), slot.draw_statistics.is_some());
            let names: Vec<&'static str> = recorded.iter().map(|&(name, _)| name).collect();
            let mut groups: Vec<&'static str> = Vec::new();
            for group in recorded.iter().filter_map(|&(_, group)| group) {
                if group != BUSY && !groups.contains(&group) {
                    groups.push(group);
                }
            }
            let (stamps, draw_stamps, draw_statistics) = read(bytes, names.len(), labels.len());
            let period_ns = self.period_ns;
            let to_ms = |ticks: u64| ticks as f64 * period_ns / 1_000_000.0;
            let duration = |(begin, end): (u64, u64)| {
                end.checked_sub(begin)
                    .map(to_ms)
                    .filter(|value| value.is_finite())
            };
            let across = |group: &str| {
                let spans: Vec<(u64, u64)> = recorded
                    .iter()
                    .zip(&stamps)
                    .filter(|((_, of), _)| *of == Some(group))
                    .map(|(_, &stamps)| stamps)
                    .collect();
                let begin = spans.iter().map(|&(begin, _)| begin).min()?;
                let end = spans.iter().map(|&(_, end)| end).max()?;
                duration((begin, end))
            };
            let span = stamps
                .iter()
                .flat_map(|&(begin, end)| [begin, end])
                .min()
                .zip(stamps.iter().flat_map(|&(begin, end)| [begin, end]).max());
            let frame_ms = span.and_then(|(first, last)| duration((first, last)));
            let previous = self.last_span.filter(|&(frame, _, _)| frame < origin.frame);
            let gap_ms = span
                .zip(previous)
                .and_then(|((first, _), (frame, _, end))| {
                    (frame + 1 == origin.frame)
                        .then(|| first.checked_sub(end))
                        .flatten()
                        .map(to_ms)
                });
            let interval_ms = span
                .zip(previous)
                .and_then(|((first, _), (frame, start, _))| {
                    first
                        .checked_sub(start)
                        .map(|ticks| to_ms(ticks) / (origin.frame - frame) as f64)
                });
            if let Some((first, last)) = span {
                self.last_span = Some((origin.frame, first, last));
            }
            let idle_ms = span.and_then(|span| idle(&recorded, &stamps, span).map(to_ms));
            self.latest = frame_ms.map(|ms| Latest::of(ms, idle_ms, &names, &stamps, duration));
            if Some(origin.capture) != self.capture {
                return;
            }
            let opaque = names
                .iter()
                .position(|name| *name == OPAQUE)
                .and_then(|index| duration(stamps[index]));
            let Some(opaque) = opaque else {
                self.errors = self.errors.saturating_add(1);
                return;
            };
            let mut spans = vec![(OPAQUE, opaque)];
            spans.extend(
                names
                    .iter()
                    .zip(&stamps)
                    .filter(|(name, _)| **name != OPAQUE)
                    .filter_map(|(name, &stamps)| Some((*name, duration(stamps)?))),
            );
            spans.extend(
                groups
                    .iter()
                    .filter_map(|&group| across(group).map(|whole| (group, whole))),
            );
            let mut stages: Vec<(&str, u64, u64)> = names
                .iter()
                .zip(&stamps)
                .filter(|(name, _)| name.starts_with("stage_"))
                .map(|(name, &(begin, end))| (*name, begin, end))
                .filter(|&(_, begin, end)| begin != 0 && end >= begin)
                .collect();
            stages.sort_by_key(|&(_, begin, _)| begin);
            if let (Some((first, last)), Some(&(_, begin, _)), Some(&(_, _, end))) =
                (span, stages.first(), stages.last())
            {
                spans.extend(duration((first, begin)).map(|lead| ("stage_lead", lead)));
                spans.extend(duration((end, last)).map(|tail| ("stage_tail", tail)));
                let gaps = stages
                    .windows(2)
                    .filter_map(|pair| duration((pair[0].2, pair[1].1)))
                    .sum::<f64>();
                spans.push(("stage_gaps", gaps));
            }
            spans.extend(frame_ms.map(|frame| ("frame_gpu", frame)));
            spans.extend(idle_ms.map(|idle| ("gpu_idle", idle)));
            spans.extend(
                gap_ms
                    .filter(|value| value.is_finite())
                    .map(|gap| ("frame_gpu_gap", gap)),
            );
            spans.extend(
                interval_ms
                    .filter(|value| value.is_finite())
                    .map(|interval| ("frame_gpu_interval", interval)),
            );
            let start_of = |begin: u64| {
                span.filter(|_| begin != 0)
                    .and_then(|(first, _)| begin.checked_sub(first))
                    .map(to_ms)
            };
            let starts = names
                .iter()
                .zip(&stamps)
                .filter_map(|(name, &(begin, _))| Some((*name, start_of(begin)?)))
                .collect();
            let draws = labels
                .into_iter()
                .zip(draw_stamps.into_iter().zip(draw_statistics))
                .map(|(label, (stamps, statistics))| DrawSample {
                    label,
                    start_ms: timed.then(|| start_of(stamps.0)).flatten(),
                    duration_ms: timed.then(|| duration(stamps)).flatten(),
                    statistics: counted.then_some(statistics),
                })
                .collect();
            self.samples.push(Sample {
                origin,
                spans,
                starts,
                draws,
            });
        }

        pub fn status(&self) -> Value {
            let active = self.capture.is_some() && engine_profile::capture_id() == self.capture;
            let pending = self
                .slots
                .iter()
                .filter(|slot| {
                    slot.origin
                        .is_some_and(|origin| Some(origin.capture) == self.capture)
                })
                .count();
            json!({
                "supported": self.supported,
                "reason": if self.supported { Value::Null } else { json!("timestamp_query_unavailable") },
                "capture_id": self.capture.map(|id| id.to_string()),
                "active": active,
                "timestamp_period_ns": if self.supported { Some(self.period_ns) } else { None },
                "ring_slots": SLOTS,
                "sample_limit": SAMPLE_LIMIT,
                "span_limit": SPANS,
                "attempted": self.attempted,
                "submitted": self.submitted,
                "completed": self.samples.len(),
                "pending": pending,
                "dropped": self.dropped,
                "spans_dropped": self.spans_dropped.get(),
                "draws": {
                    "timestamps": self.draw_support.timestamps,
                    "statistics": self.draw_support.statistics,
                    "limit": DRAWS,
                    "dropped": self.draws_dropped.get(),
                },
                "errors": self.errors,
                "sample_limit_reached": self.sample_limit_reached,
                "complete": self.capture.is_some() && !active && self.supported && pending == 0 && self.dropped == 0 && self.spans_dropped.get() == 0 && self.errors == 0,
                "readback_buffer_bytes": self.slots.len() as u64 * SLOT_BYTES * 2,
                "cpu_sample_capacity_bytes": self.samples.capacity() * size_of::<Sample>(),
            })
        }

        pub fn report(&self) -> Value {
            let mut report = self.status();
            report["samples"] = Value::Array(
                self.samples
                    .iter()
                    .flat_map(|sample| {
                        sample.spans.iter().map(|&(name, duration_ms)| {
                            let start = sample.starts.iter().find(|&&(of, _)| of == name);
                            json!({
                                "capture_id": sample.origin.capture.to_string(),
                                "frame": sample.origin.frame.to_string(),
                                "submission": sample.origin.submission.to_string(),
                                "name": name,
                                "start_ms": start.map(|&(_, start)| start),
                                "duration_ms": duration_ms,
                            })
                        })
                    })
                    .collect(),
            );
            report["draw_samples"] = Value::Array(
                self.samples
                    .iter()
                    .flat_map(|sample| {
                        sample.draws.iter().map(|draw| {
                            json!({
                                "frame": sample.origin.frame.to_string(),
                                "label": draw.label,
                                "start_ms": draw.start_ms,
                                "duration_ms": draw.duration_ms,
                                "vertex_invocations": draw.statistics.map(|counts| counts[0]),
                                "primitives": draw.statistics.map(|counts| counts[1]),
                                "fragment_invocations": draw.statistics.map(|counts| counts[2]),
                            })
                        })
                    })
                    .collect(),
            );
            report
        }
    }
}
