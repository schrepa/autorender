//! GPU time of the frame's passes during profiling captures, as named
//! spans: each frame allocates two timestamp queries for every span its
//! passes record, a pass writes a span's begin, its end, or both, and the
//! report names samples by span. Spans covering several passes (the opaque
//! phases, the near cascades, occlusion culling's pyramid and test) are the
//! same span asked for again within the frame. Readback is bounded and
//! never waits on the GPU: WebGPU completes maps through its event loop.
//! Where the device times work inside passes and counts pipeline
//! statistics (native GPU interfaces), captured frames also time each draw
//! of the opaque passes and count its vertex shader invocations, the
//! primitives it rasterized and its fragment shader invocations.
//! Builds without the `profiling` feature compile a stand-in that records
//! nothing, so passes ask for spans without conditional compilation.

/// A span of GPU time this frame: the query set and its begin query (the
/// end query follows it).
#[derive(Clone, Copy)]
pub(crate) struct Span<'a> {
    query_set: &'a wgpu::QuerySet,
    begin: u32,
}

impl<'a> Span<'a> {
    /// Writes of a render pass the span covers whole.
    pub(crate) fn render(self) -> wgpu::RenderPassTimestampWrites<'a> {
        self.render_part(true, true)
    }

    /// Writes of a compute pass the span covers whole.
    pub(crate) fn compute(self) -> wgpu::ComputePassTimestampWrites<'a> {
        self.compute_part(true, true)
    }

    /// Writes of a render pass that begins the span (`begin`), ends it
    /// (`end`), or both.
    pub(crate) fn render_part(self, begin: bool, end: bool) -> wgpu::RenderPassTimestampWrites<'a> {
        wgpu::RenderPassTimestampWrites {
            query_set: self.query_set,
            beginning_of_pass_write_index: begin.then_some(self.begin),
            end_of_pass_write_index: end.then_some(self.begin + 1),
        }
    }

    /// Writes of a compute pass that begins the span (`begin`), ends it
    /// (`end`), or both.
    pub(crate) fn compute_part(
        self,
        begin: bool,
        end: bool,
    ) -> wgpu::ComputePassTimestampWrites<'a> {
        wgpu::ComputePassTimestampWrites {
            query_set: self.query_set,
            beginning_of_pass_write_index: begin.then_some(self.begin),
            end_of_pass_write_index: end.then_some(self.begin + 1),
        }
    }

    /// Writes the span's begin, or its `end`, from an empty compute pass:
    /// for work recorded elsewhere between the two.
    pub(crate) fn mark(self, encoder: &mut wgpu::CommandEncoder, end: bool) {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/profile-timestamp"),
            timestamp_writes: Some(self.compute_part(!end, end)),
        });
    }
}

/// A draw a captured frame times: its index among the frame's.
#[cfg_attr(not(feature = "profiling"), allow(dead_code))]
pub(crate) struct Draw(u32);

/// What profiling builds ask devices for: timestamps, and inside passes
/// with pipeline statistics where the device has them.
#[cfg(feature = "profiling")]
pub(crate) const FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY
    .union(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    .union(wgpu::Features::PIPELINE_STATISTICS_QUERY);

#[cfg(feature = "profiling")]
pub(crate) use capture::GpuProfiler;

/// The stand-in of builds without profiling: no frame records spans.
#[cfg(not(feature = "profiling"))]
pub(crate) struct GpuProfiler;

#[cfg(not(feature = "profiling"))]
impl GpuProfiler {
    pub(crate) fn new(_features: wgpu::Features, _period_ns: f64) -> Self {
        Self
    }

    #[inline(always)]
    pub(crate) fn begin_draw(
        &self,
        _pass: &mut wgpu::RenderPass<'_>,
        _label: impl FnOnce() -> String,
    ) -> Option<Draw> {
        None
    }

    #[inline(always)]
    pub(crate) fn end_draw(&self, _pass: &mut wgpu::RenderPass<'_>, _draw: Option<Draw>) {}

    #[inline(always)]
    pub(crate) fn begin_frame(&mut self, _device: &wgpu::Device) {}

    #[inline(always)]
    pub(crate) fn capturing(&self) -> bool {
        false
    }

    #[inline(always)]
    pub(crate) fn span(&self, _name: &'static str) -> Option<Span<'_>> {
        None
    }

    #[inline(always)]
    pub(crate) fn span_in(&self, _group: &'static str, _name: &'static str) -> Option<Span<'_>> {
        None
    }

    #[inline(always)]
    pub(crate) fn end_frame(&mut self, _encoder: &mut wgpu::CommandEncoder) {}

    #[inline(always)]
    pub(crate) fn submitted(&mut self) {}
}

#[cfg(feature = "profiling")]
mod capture {
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
    };

    use serde_json::{Value, json};

    use super::{Draw, Span};

    const SLOTS: usize = 3;
    const SAMPLE_LIMIT: usize = 4_096;
    /// Spans one frame records at most, a bound of the dev protocol; frames
    /// record about sixty with their stages.
    const SPANS: usize = 128;
    const QUERIES: u32 = 2 * SPANS as u32;
    const QUERY_BYTES: u64 = QUERIES as u64 * 8;
    /// Draws one frame times at most: two timestamps each fill a query
    /// set's 4096 queries.
    const DRAWS: usize = 2_048;
    /// What each draw counts, in the order queries resolve them: vertex
    /// shader invocations, primitives out of clipping, fragment shader
    /// invocations.
    const STATISTICS: wgpu::PipelineStatisticsTypes =
        wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
            .union(wgpu::PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT)
            .union(wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS);
    // Where draws' timestamps and statistics resolve, after the spans'.
    const DRAW_STAMPS_AT: u64 = QUERY_BYTES;
    const DRAW_STATISTICS_AT: u64 = DRAW_STAMPS_AT + 2 * DRAWS as u64 * 8;
    const SLOT_BYTES: u64 = DRAW_STATISTICS_AT + DRAWS as u64 * 24;
    /// The span every captured frame records: its opaque passes.
    const OPAQUE: &str = "opaque_meshes";

    #[derive(Clone, Copy)]
    struct Origin {
        capture: u64,
        frame: u64,
        submission: u64,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum MapState {
        Pending,
        Ready,
        Failed,
    }

    struct Slot {
        queries: wgpu::QuerySet,
        resolve: wgpu::Buffer,
        readback: wgpu::Buffer,
        mapped: Rc<Cell<MapState>>,
        origin: Option<Origin>,
        // The spans this frame recorded, in order, by name and begin query,
        // each with the group whose first begin to last end the report
        // also times.
        spans: RefCell<Vec<(&'static str, Option<&'static str>)>>,
        // Draws' timestamps and statistics, where the device has them, and
        // the draws this frame timed, by label.
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
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("engine/profile-resolve"),
                    size: SLOT_BYTES,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("engine/profile-readback"),
                    size: SLOT_BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                mapped: Rc::new(Cell::new(MapState::Pending)),
                origin: None,
                spans: RefCell::new(Vec::with_capacity(SPANS)),
                draw_stamps,
                draw_statistics,
                draws: RefCell::new(Vec::new()),
            }
        }
    }

    /// What a device measures of each draw.
    #[derive(Clone, Copy)]
    struct Support {
        timestamps: bool,
        statistics: bool,
    }

    /// A draw's measurements in a captured frame.
    struct DrawSample {
        label: String,
        duration_ms: Option<f64>,
        // Vertex shader invocations, primitives out of clipping, fragment
        // shader invocations.
        statistics: Option<[u64; 3]>,
    }

    /// A read-back frame's span timestamps, draw timestamps and draw
    /// statistics.
    type Readback = (Vec<(u64, u64)>, Vec<(u64, u64)>, Vec<[u64; 3]>);

    struct Sample {
        origin: Origin,
        // Each recorded span's duration, then the aggregates: in report order.
        spans: Vec<(&'static str, f64)>,
        draws: Vec<DrawSample>,
    }

    /// Retains one capture and a small reusable ring. A new capture clears
    /// samples, while in-flight old maps finish normally and are ignored by
    /// capture ID.
    pub(crate) struct GpuProfiler {
        supported: bool,
        draw_support: Support,
        period_ns: f64,
        slots: Vec<Slot>,
        // The slot this frame records into.
        current: Option<usize>,
        capture: Option<u64>,
        submission: u64,
        attempted: u64,
        submitted: u64,
        dropped: u64,
        // Spans frames asked for beyond `SPANS`, and draws beyond `DRAWS`.
        spans_dropped: Cell<u64>,
        draws_dropped: Cell<u64>,
        errors: u64,
        sample_limit_reached: bool,
        samples: Vec<Sample>,
        // The last read frame and its first and last GPU timestamps.
        last_span: Option<(u64, u64, u64)>,
    }

    impl GpuProfiler {
        pub(crate) fn new(features: wgpu::Features, period_ns: f64) -> Self {
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

        pub(crate) fn begin(&mut self, device: &wgpu::Device, capture: u64) {
            self.poll();
            self.capture = Some(capture);
            self.attempted = 0;
            self.submitted = 0;
            self.dropped = 0;
            self.spans_dropped.set(0);
            self.draws_dropped.set(0);
            self.errors = 0;
            self.sample_limit_reached = false;
            self.samples.clear();
            // Allocate at the explicit capture boundary, never grow in the
            // frame loop.
            if self.supported && self.slots.is_empty() {
                let draws = self.draw_support;
                self.slots = (0..SLOTS).map(|_| Slot::new(device, draws)).collect();
                self.samples.reserve_exact(SAMPLE_LIMIT);
            }
        }

        /// Starts a frame: while a capture runs, a free slot records its
        /// spans.
        pub(crate) fn begin_frame(&mut self, device: &wgpu::Device) {
            self.current = None;
            self.submission = self.submission.saturating_add(1);
            self.poll();
            let Some(capture) = engine_profile::capture_id() else {
                return;
            };
            if self.capture != Some(capture) {
                self.begin(device, capture);
            }
            self.attempted = self.attempted.saturating_add(1);
            if !self.supported {
                return;
            }
            if self.submitted as usize >= SAMPLE_LIMIT {
                self.sample_limit_reached = true;
                self.dropped = self.dropped.saturating_add(1);
                return;
            }
            let Some(index) = self.slots.iter().position(|slot| slot.origin.is_none()) else {
                self.dropped = self.dropped.saturating_add(1);
                return;
            };
            let slot = &mut self.slots[index];
            slot.origin = Some(Origin {
                capture,
                frame: engine_profile::current_frame(),
                submission: self.submission,
            });
            slot.mapped.set(MapState::Pending);
            slot.spans.get_mut().clear();
            slot.draws.get_mut().clear();
            self.submitted += 1;
            self.current = Some(index);
        }

        /// Whether this frame records spans: passes timed in parts then run
        /// each part as its own pass.
        pub(crate) fn capturing(&self) -> bool {
            self.current.is_some()
        }

        /// Span `name` of this frame, allocated the first time a pass asks
        /// for it; `None` unless this frame is captured. Every span asked for
        /// must have its begin and end written.
        pub(crate) fn span(&self, name: &'static str) -> Option<Span<'_>> {
            self.recorded(None, name)
        }

        /// [`Self::span`] of a span in `group`: the report also times the
        /// group, from its spans' first begin to their last end.
        pub(crate) fn span_in(&self, group: &'static str, name: &'static str) -> Option<Span<'_>> {
            self.recorded(Some(group), name)
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

        /// Begins timing the draw `label` names in `pass`, where this frame
        /// is captured and the device measures work inside passes.
        pub(crate) fn begin_draw(
            &self,
            pass: &mut wgpu::RenderPass<'_>,
            label: impl FnOnce() -> String,
        ) -> Option<Draw> {
            let slot = &self.slots[self.current?];
            if slot.draw_stamps.is_none() && slot.draw_statistics.is_none() {
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

        /// Ends timing `draw`.
        pub(crate) fn end_draw(&self, pass: &mut wgpu::RenderPass<'_>, draw: Option<Draw>) {
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

        /// Resolves this frame's queries into its read-back buffer.
        pub(crate) fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
            let Some(index) = self.current else {
                return;
            };
            let slot = &self.slots[index];
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
            if queries != 0 || draws != 0 {
                encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, SLOT_BYTES);
            }
        }

        /// Maps the frame's read-back after its submission.
        pub(crate) fn submitted(&mut self) {
            let Some(index) = self.current.take() else {
                return;
            };
            let slot = &self.slots[index];
            let mapped = Rc::clone(&slot.mapped);
            slot.readback.map_async(
                wgpu::MapMode::Read,
                ..,
                crate::platform::map_callback(move |result| {
                    mapped.set(if result.is_ok() {
                        MapState::Ready
                    } else {
                        MapState::Failed
                    });
                }),
            );
        }

        pub(crate) fn poll(&mut self) {
            // Browser GPU maps complete asynchronously. This only consumes
            // flags and mapped bytes; it makes no device.poll / queue-wait
            // call.
            for index in 0..self.slots.len() {
                if Some(index) == self.current {
                    continue;
                }
                let slot = &mut self.slots[index];
                let Some(origin) = slot.origin else { continue };
                let state = slot.mapped.get();
                if state == MapState::Pending {
                    continue;
                }
                slot.origin = None;
                if state == MapState::Failed {
                    if Some(origin.capture) == self.capture {
                        self.errors = self.errors.saturating_add(1);
                    }
                    continue;
                }
                let recorded = slot.spans.get_mut().clone();
                let labels = std::mem::take(slot.draws.get_mut());
                let (timed, counted) = (slot.draw_stamps.is_some(), slot.draw_statistics.is_some());
                let names: Vec<&'static str> = recorded.iter().map(|&(name, _)| name).collect();
                // Each group in the order its first span was recorded.
                let mut groups: Vec<&'static str> = Vec::new();
                for group in recorded.iter().filter_map(|&(_, group)| group) {
                    if !groups.contains(&group) {
                        groups.push(group);
                    }
                }
                let read: Option<Readback> = slot.readback.get_mapped_range(..).ok().map(|bytes| {
                    let word = |at: usize| {
                        bytes
                            .get(at..at + 8)
                            .and_then(|word| word.try_into().ok())
                            .map_or(0, u64::from_le_bytes)
                    };
                    let stamp = |query: usize| word(query * 8);
                    let spans = (0..names.len())
                        .map(|span| (stamp(2 * span), stamp(2 * span + 1)))
                        .collect();
                    let draw_stamp = |query: usize| word(DRAW_STAMPS_AT as usize + query * 8);
                    let draws = (0..labels.len())
                        .map(|draw| (draw_stamp(2 * draw), draw_stamp(2 * draw + 1)))
                        .collect();
                    let statistic = |draw: usize, counter: usize| {
                        word(DRAW_STATISTICS_AT as usize + draw * 24 + counter * 8)
                    };
                    let statistics = (0..labels.len())
                        .map(|draw| [0, 1, 2].map(|counter| statistic(draw, counter)))
                        .collect();
                    (spans, draws, statistics)
                });
                slot.readback.unmap();
                let Some((stamps, draw_stamps, draw_statistics)) = read else {
                    self.errors = self.errors.saturating_add(1);
                    continue;
                };
                let period_ns = self.period_ns;
                let to_ms = |ticks: u64| ticks as f64 * period_ns / 1_000_000.0;
                let duration = |(begin, end): (u64, u64)| {
                    end.checked_sub(begin)
                        .map(to_ms)
                        .filter(|value| value.is_finite())
                };
                // The first begin to the last end of the spans in `group`.
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
                // The frame's GPU span: every query it wrote.
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
                if Some(origin.capture) != self.capture {
                    continue;
                }
                let opaque = names
                    .iter()
                    .position(|name| *name == OPAQUE)
                    .and_then(|index| duration(stamps[index]));
                let Some(opaque) = opaque else {
                    self.errors = self.errors.saturating_add(1);
                    continue;
                };
                // The opaque span first, then every other span in the order
                // the frame recorded them, each group whole and the frame's
                // GPU span, gap and interval.
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
                // Where the frame's GPU span goes outside the render graph's
                // passes: before the first, after the last, and between
                // each pass and the one before it.
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
                let draws = labels
                    .into_iter()
                    .zip(draw_stamps.into_iter().zip(draw_statistics))
                    .map(|(label, (stamps, statistics))| DrawSample {
                        label,
                        duration_ms: timed.then(|| duration(stamps)).flatten(),
                        statistics: counted.then_some(statistics),
                    })
                    .collect();
                // begin_frame() accounts for completed and pending samples
                // together.
                self.samples.push(Sample {
                    origin,
                    spans,
                    draws,
                });
            }
        }

        pub(crate) fn status(&self) -> Value {
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
                "reason": if self.supported { Value::Null } else { json!("timestamp-query unavailable on adapter") },
                "capture_id": self.capture.map(|id| id.to_string()),
                "active": active,
                "scope": "every span the captured frames recorded, by name: opaque_meshes (the opaque passes), the passes and dispatch groups the renderer times, each group of spans whole under the group's name from its first begin to its last end, stage_<pass> for every render graph pass whole (with the copies and passes it records, timed or not), stage_lead, stage_gaps and stage_tail (the frame's GPU span before, between and after the passes: stalls inside passes with almost no gaps mean other work shares the GPU), pre_<what> for the command buffers submitted before the frame's, frame_gpu (the frame's first timed begin to last timed end, including other work between its passes), frame_gpu_gap (from the previous frame's last end, when consecutive) and frame_gpu_interval (first begin to first begin per frame); excludes uploads, queue latency, presentation and timestamp readback",
                "clock": "GPU duration only; frame/submission IDs correlate origin, clocks are not aligned",
                "timestamp_period_ns": if self.supported { Some(self.period_ns) } else { None },
                "precision": "browser may quantize timestamp values; zero durations remain valid samples",
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
                    "scope": "each draw of the opaque passes in captured frames, where the device times work inside passes (timestamps) and counts pipeline statistics: vertex shader invocations, primitives out of clipping and fragment shader invocations; browsers have neither",
                },
                "errors": self.errors,
                "sample_limit_reached": self.sample_limit_reached,
                "complete": self.capture.is_some() && !active && self.supported && pending == 0 && self.dropped == 0 && self.spans_dropped.get() == 0 && self.errors == 0,
                "readback_buffer_bytes": self.slots.len() as u64 * SLOT_BYTES * 2,
                "cpu_sample_capacity_bytes": self.samples.capacity() * size_of::<Sample>(),
            })
        }

        pub(crate) fn report(&mut self) -> Value {
            self.poll();
            let mut report = self.status();
            report["samples"] = Value::Array(
                self.samples
                    .iter()
                    .flat_map(|sample| {
                        sample.spans.iter().map(|&(name, duration_ms)| {
                            json!({
                                "capture_id": sample.origin.capture.to_string(),
                                "frame": sample.origin.frame.to_string(),
                                "submission": sample.origin.submission.to_string(),
                                "name": name,
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
