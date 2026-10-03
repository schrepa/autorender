//! GPU time of the frame's passes during profiling captures, as named
//! spans: each frame allocates two timestamp queries for every span its
//! passes record, a pass writes a span's begin, its end, or both, and the
//! report names samples by span. Spans covering several passes (the opaque
//! phases, the near cascades, occlusion culling's pyramid and test) are the
//! same span asked for again within the frame. Readback is bounded and
//! never waits on the GPU: WebGPU completes maps through its event loop.
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

#[cfg(feature = "profiling")]
pub(crate) use capture::GpuProfiler;

/// The stand-in of builds without profiling: no frame records spans.
#[cfg(not(feature = "profiling"))]
pub(crate) struct GpuProfiler;

#[cfg(not(feature = "profiling"))]
impl GpuProfiler {
    pub(crate) fn new(_supported: bool, _period_ns: f64) -> Self {
        Self
    }

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

    use super::Span;

    const SLOTS: usize = 3;
    const SAMPLE_LIMIT: usize = 4_096;
    /// Spans one frame records at most, a bound of the dev protocol; frames
    /// record about thirty.
    const SPANS: usize = 64;
    const QUERIES: u32 = 2 * SPANS as u32;
    const QUERY_BYTES: u64 = QUERIES as u64 * 8;
    /// The span every captured frame records: its opaque passes.
    const OPAQUE: &str = "opaque_meshes";
    /// Presentation stages of a frame's own compute passes, whose first
    /// begin to last end is `presentation_compute`; the previous-pose shift
    /// is submitted before the frame's uploads.
    const PRESENTATION: [&str; 4] = [
        "presentation_compose",
        "presentation_cull",
        "presentation_scan",
        "presentation_scatter",
    ];
    /// Shadow maps, whose first begin to last end is `shadow_maps`.
    const SHADOW_MAPS: [&str; 3] = ["shadow_near", "shadow_moving", "shadow_kept"];

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
        // The spans this frame recorded, in order, by name and begin query.
        spans: RefCell<Vec<&'static str>>,
    }

    impl Slot {
        fn new(device: &wgpu::Device) -> Self {
            Self {
                queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("engine/profile-timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: QUERIES,
                }),
                resolve: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("engine/profile-resolve"),
                    size: QUERY_BYTES,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("engine/profile-readback"),
                    size: QUERY_BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                mapped: Rc::new(Cell::new(MapState::Pending)),
                origin: None,
                spans: RefCell::new(Vec::with_capacity(SPANS)),
            }
        }
    }

    struct Sample {
        origin: Origin,
        // Each recorded span's duration, then the aggregates: in report order.
        spans: Vec<(&'static str, f64)>,
    }

    /// Retains one capture and a small reusable ring. A new capture clears
    /// samples, while in-flight old maps finish normally and are ignored by
    /// capture ID.
    pub(crate) struct GpuProfiler {
        supported: bool,
        period_ns: f64,
        slots: Vec<Slot>,
        // The slot this frame records into.
        current: Option<usize>,
        capture: Option<u64>,
        submission: u64,
        attempted: u64,
        submitted: u64,
        dropped: u64,
        // Spans frames asked for beyond `SPANS`.
        spans_dropped: Cell<u64>,
        errors: u64,
        sample_limit_reached: bool,
        samples: Vec<Sample>,
        // The last read frame and its first and last GPU timestamps.
        last_span: Option<(u64, u64, u64)>,
    }

    impl GpuProfiler {
        pub(crate) fn new(supported: bool, period_ns: f64) -> Self {
            Self {
                supported,
                period_ns,
                slots: Vec::new(),
                current: None,
                capture: None,
                submission: 0,
                attempted: 0,
                submitted: 0,
                dropped: 0,
                spans_dropped: Cell::new(0),
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
            self.errors = 0;
            self.sample_limit_reached = false;
            self.samples.clear();
            // Allocate at the explicit capture boundary, never grow in the
            // frame loop.
            if self.supported && self.slots.is_empty() {
                self.slots = (0..SLOTS).map(|_| Slot::new(device)).collect();
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
            let slot = &self.slots[self.current?];
            let mut spans = slot.spans.borrow_mut();
            let index = match spans.iter().position(|span| *span == name) {
                Some(index) => index,
                None if spans.len() < SPANS => {
                    spans.push(name);
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

        /// Resolves this frame's queries into its read-back buffer.
        pub(crate) fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
            let Some(index) = self.current else {
                return;
            };
            let slot = &self.slots[index];
            let queries = 2 * slot.spans.borrow().len() as u32;
            if queries != 0 {
                encoder.resolve_query_set(&slot.queries, 0..queries, &slot.resolve, 0);
                encoder.copy_buffer_to_buffer(
                    &slot.resolve,
                    0,
                    &slot.readback,
                    0,
                    u64::from(queries) * 8,
                );
            }
        }

        /// Maps the frame's read-back after its submission.
        pub(crate) fn submitted(&mut self) {
            let Some(index) = self.current.take() else {
                return;
            };
            let slot = &self.slots[index];
            let mapped = Rc::clone(&slot.mapped);
            slot.readback
                .map_async(wgpu::MapMode::Read, .., move |result| {
                    mapped.set(if result.is_ok() {
                        MapState::Ready
                    } else {
                        MapState::Failed
                    });
                });
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
                let names = slot.spans.get_mut().clone();
                let stamps: Option<Vec<(u64, u64)>> =
                    slot.readback.get_mapped_range(..).ok().map(|bytes| {
                        let stamp = |query: usize| {
                            bytes
                                .get(query * 8..query * 8 + 8)
                                .and_then(|word| word.try_into().ok())
                                .map_or(0, u64::from_le_bytes)
                        };
                        (0..names.len())
                            .map(|span| (stamp(2 * span), stamp(2 * span + 1)))
                            .collect()
                    });
                slot.readback.unmap();
                let Some(stamps) = stamps else {
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
                // The first begin to the last end of the spans named `set`.
                let across = |set: &[&str]| {
                    let spans: Vec<(u64, u64)> = names
                        .iter()
                        .zip(&stamps)
                        .filter(|(name, _)| set.contains(name))
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
                // The opaque span first, then the frame's compute passes,
                // every other span in the order the frame recorded them, the
                // shadow maps together and the frame's GPU span, gap and
                // interval.
                let mut spans = vec![(OPAQUE, opaque)];
                spans
                    .extend(across(&PRESENTATION).map(|compute| ("presentation_compute", compute)));
                spans.extend(
                    names
                        .iter()
                        .zip(&stamps)
                        .filter(|(name, _)| **name != OPAQUE)
                        .filter_map(|(name, &stamps)| Some((*name, duration(stamps)?))),
                );
                spans.extend(across(&SHADOW_MAPS).map(|maps| ("shadow_maps", maps)));
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
                // begin_frame() accounts for completed and pending samples
                // together.
                self.samples.push(Sample { origin, spans });
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
                "scope": "opaque render pass, plus presentation_compute (the frame's compute passes) and per-stage presentation samples for GPU-presented frames (decision 014; stages run as separate passes while capturing), taa_resolve for temporally antialiased frames (decision 017), shadow_select and shadow_maps (decision 020), ambient_occlusion, bloom and post_composite (decision 021), reflections and sun_shafts (decisions 024 and 025), light_bins (decision 023), decals (decision 039), transparent and transparent_composite (decision 026), shadow_near, shadow_moving and shadow_kept per map, occlusion (decision 040: the depth pyramid, test and second compaction, with virtual geometry's cluster retest; opaque_meshes then spans both drawing phases), probe_captures (decision 043) and virtual_selection (decision 044: every cluster selection dispatch), effects_filter, and frame_gpu (the frame's first timed begin to last timed end, including GPU time of other work between its passes), frame_gpu_gap (from the previous frame's last end, when consecutive) and frame_gpu_interval (first begin to first begin per frame); excludes uploads, queue latency, presentation and timestamp readback",
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
                "errors": self.errors,
                "sample_limit_reached": self.sample_limit_reached,
                "complete": self.capture.is_some() && !active && self.supported && pending == 0 && self.dropped == 0 && self.spans_dropped.get() == 0 && self.errors == 0,
                "readback_buffer_bytes": self.slots.len() as u64 * QUERY_BYTES * 2,
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
            report
        }
    }
}
