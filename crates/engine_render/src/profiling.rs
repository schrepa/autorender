//! Optional, bounded GPU timestamp readback. WebGPU completes maps through its
//! event loop; neither rendering nor capture settlement ever waits on the GPU.

use std::{cell::Cell, rc::Rc};

use serde_json::{Value, json};

use crate::gpu_presentation::{STAGE_COUNT, StageStamps};

const SLOTS: usize = 3;
const SAMPLE_LIMIT: usize = 4_096;
/// Names of the GPU presentation stages, indexed like `StageStamps` stages.
/// Captures run each stage as its own pass (see `GpuPresentationPass::encode`).
pub(crate) const STAGES: [&str; STAGE_COUNT] = [
    "presentation_shift",
    "presentation_compose",
    "presentation_cull",
    "presentation_scan",
    "presentation_scatter",
];
/// Passes of shadows (decision 020), screen effects (decisions 021, 024 and
/// 025), local light binning (decision 023), decals (decision 039),
/// transparency (decision 026) and reflection probe captures (decision 043),
/// each timed from its begin to its end query. `shadow_maps` spans the near
/// cascades' begin (or the moving map's) to the kept map's end (or the moving
/// map's), from the per-map queries; `probe_captures` every face captured in
/// a frame, with its culling, light binning and blurring.
pub(crate) const EFFECTS: [&str; 16] = [
    "shadow_select",
    "shadow_maps",
    "ambient_occlusion",
    "bloom",
    "post_composite",
    "reflections",
    "sun_shafts",
    "light_bins",
    "decals",
    "transparent",
    "transparent_composite",
    "shadow_near",
    "shadow_moving",
    "shadow_kept",
    // Occlusion culling (decision 040): the depth pyramid, the test and
    // phase two's compaction.
    "occlusion",
    "probe_captures",
];
const SHADOW_MAPS: usize = 1;
const SHADOW_NEAR: usize = 11;
const SHADOW_MOVING: usize = 12;
const SHADOW_KEPT: usize = 13;
// Render pass begin/end, begin/end of each presentation stage, then of the
// temporal antialiasing resolve (decision 017), then of each effect.
const RESOLVE_QUERY: u32 = 2 + 2 * STAGE_COUNT as u32;
const EFFECT_QUERY: u32 = RESOLVE_QUERY + 2;
const QUERIES: u32 = EFFECT_QUERY + 2 * EFFECTS.len() as u32;
const QUERY_BYTES: u64 = QUERIES as u64 * 8;

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
    // Presentation stages this submission timed, one bit per `STAGES` entry.
    stages: u8,
    // Whether this submission timed a temporal antialiasing resolve.
    resolved: bool,
    // Effects this submission timed, one bit per `EFFECTS` entry.
    effects: u16,
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
            stages: 0,
            resolved: false,
            effects: 0,
        }
    }
}

struct Sample {
    origin: Origin,
    duration_ms: f64,
    // First to last timed stage of the frame's own compute passes.
    compute_ms: Option<f64>,
    stages: [Option<f64>; STAGES.len()],
    resolve_ms: Option<f64>,
    effects: [Option<f64>; EFFECTS.len()],
}

/// Retains one capture and a small reusable ring. A new capture clears samples,
/// while in-flight old maps finish normally and are ignored by capture ID.
pub(crate) struct GpuProfiler {
    supported: bool,
    period_ns: f64,
    slots: Vec<Slot>,
    capture: Option<u64>,
    submission: u64,
    attempted: u64,
    submitted: u64,
    dropped: u64,
    errors: u64,
    sample_limit_reached: bool,
    samples: Vec<Sample>,
}

impl GpuProfiler {
    pub(crate) fn new(supported: bool, period_ns: f64) -> Self {
        Self {
            supported,
            period_ns,
            slots: Vec::new(),
            capture: None,
            submission: 0,
            attempted: 0,
            submitted: 0,
            dropped: 0,
            errors: 0,
            sample_limit_reached: false,
            samples: Vec::new(),
        }
    }

    pub(crate) fn begin(&mut self, device: &wgpu::Device, capture: u64) {
        self.poll();
        self.capture = Some(capture);
        self.attempted = 0;
        self.submitted = 0;
        self.dropped = 0;
        self.errors = 0;
        self.sample_limit_reached = false;
        self.samples.clear();
        // Allocate at the explicit capture boundary, never grow in the frame loop.
        if self.supported && self.slots.is_empty() {
            self.slots = (0..SLOTS).map(|_| Slot::new(device)).collect();
            self.samples.reserve_exact(SAMPLE_LIMIT);
        }
    }

    pub(crate) fn reserve(&mut self, device: &wgpu::Device) -> Option<usize> {
        self.submission = self.submission.saturating_add(1);
        self.poll();
        let capture = engine_profile::capture_id()?;
        if self.capture != Some(capture) {
            self.begin(device, capture);
        }
        self.attempted = self.attempted.saturating_add(1);
        if !self.supported {
            return None;
        }
        if self.submitted as usize >= SAMPLE_LIMIT {
            self.sample_limit_reached = true;
            self.dropped = self.dropped.saturating_add(1);
            return None;
        }
        let Some(index) = self.slots.iter().position(|slot| slot.origin.is_none()) else {
            self.dropped = self.dropped.saturating_add(1);
            return None;
        };
        let slot = &mut self.slots[index];
        slot.origin = Some(Origin {
            capture,
            frame: engine_profile::current_frame(),
            submission: self.submission,
        });
        slot.mapped.set(MapState::Pending);
        slot.stages = 0;
        slot.resolved = false;
        slot.effects = 0;
        self.submitted += 1;
        Some(index)
    }

    pub(crate) fn writes(
        &self,
        index: Option<usize>,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        index.map(|index| wgpu::RenderPassTimestampWrites {
            query_set: &self.slots[index].queries,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        })
    }

    /// Times the presentation stages of a GPU-presented frame, each in its
    /// own pass. Report the stages that ran with [`Self::timed_stages`].
    pub(crate) fn stage_stamps(&self, index: Option<usize>) -> Option<StageStamps<'_>> {
        Some(StageStamps::new(&self.slots[index?].queries))
    }

    /// Times a temporal antialiasing resolve pass; report it with
    /// [`Self::timed_resolve`].
    pub(crate) fn resolve_writes(
        &self,
        index: Option<usize>,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        index.map(|index| wgpu::RenderPassTimestampWrites {
            query_set: &self.slots[index].queries,
            beginning_of_pass_write_index: Some(RESOLVE_QUERY),
            end_of_pass_write_index: Some(RESOLVE_QUERY + 1),
        })
    }

    pub(crate) fn timed_resolve(&mut self, index: Option<usize>) {
        if let Some(index) = index {
            self.slots[index].resolved = true;
        }
    }

    /// The query set and first query of the effect passes: effect `e`
    /// writes queries `first + 2e` and `first + 2e + 1`. Report the effects
    /// that ran with [`Self::timed_effects`].
    pub(crate) fn effect_queries(&self, index: Option<usize>) -> Option<(&wgpu::QuerySet, u32)> {
        Some((&self.slots[index?].queries, EFFECT_QUERY))
    }

    /// Records the effects (bits of `EFFECTS`) whose passes wrote timestamps.
    pub(crate) fn timed_effects(&mut self, index: Option<usize>, effects: u16) {
        if let Some(index) = index {
            self.slots[index].effects |= effects;
        }
    }

    /// Records the stages (bits of `STAGES`) whose passes wrote timestamps.
    pub(crate) fn timed_stages(&mut self, index: Option<usize>, stages: u8) {
        if let Some(index) = index {
            self.slots[index].stages |= stages;
        }
    }

    pub(crate) fn resolve(&self, index: Option<usize>, encoder: &mut wgpu::CommandEncoder) {
        if let Some(index) = index {
            let slot = &self.slots[index];
            // Queries of passes that did not run are never read.
            let queries = if slot.effects != 0 {
                QUERIES
            } else if slot.resolved {
                EFFECT_QUERY
            } else if slot.stages != 0 {
                RESOLVE_QUERY
            } else {
                2
            };
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

    pub(crate) fn map_after_submit(&self, index: Option<usize>) {
        if let Some(index) = index {
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
    }

    pub(crate) fn poll(&mut self) {
        // Browser GPU maps complete asynchronously. This only consumes flags and
        // mapped bytes; it makes no device.poll / queue-wait call.
        for slot in &mut self.slots {
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
            let period_ns = self.period_ns;
            let elapsed = |bytes: &[u8], first: usize| {
                let start = u64::from_le_bytes(bytes.get(first..first + 8)?.try_into().ok()?);
                let end = u64::from_le_bytes(bytes.get(first + 8..first + 16)?.try_into().ok()?);
                end.checked_sub(start)
                    .map(|ticks| ticks as f64 * period_ns / 1_000_000.0)
            };
            let stamp = |bytes: &[u8], query: usize| {
                Some(u64::from_le_bytes(
                    bytes.get(query * 8..query * 8 + 8)?.try_into().ok()?,
                ))
            };
            let mut stages = [None; STAGES.len()];
            let mut resolve_ms = None;
            let mut effects = [None; EFFECTS.len()];
            let (duration, compute_ms) =
                slot.readback
                    .get_mapped_range(..)
                    .ok()
                    .map_or((None, None), |bytes| {
                        let timed = |stage: usize| slot.stages & (1 << stage) != 0;
                        if slot.resolved {
                            resolve_ms = elapsed(&bytes, RESOLVE_QUERY as usize * 8)
                                .filter(|value| value.is_finite());
                        }
                        for (stage, value) in stages.iter_mut().enumerate() {
                            if timed(stage) {
                                *value = elapsed(&bytes, 16 + 16 * stage)
                                    .filter(|value| value.is_finite());
                            }
                        }
                        for (effect, value) in effects.iter_mut().enumerate() {
                            if effect != SHADOW_MAPS && slot.effects & (1 << effect) != 0 {
                                *value = elapsed(&bytes, (EFFECT_QUERY as usize + 2 * effect) * 8)
                                    .filter(|value| value.is_finite());
                            }
                        }
                        // The maps together: first map's begin to last map's end.
                        if slot.effects & (1 << SHADOW_MAPS) != 0 {
                            let query = |effect: usize, end: usize| {
                                EFFECT_QUERY as usize + 2 * effect + end
                            };
                            let timed_map = |effect: usize| slot.effects & (1 << effect) != 0;
                            let first = if timed_map(SHADOW_NEAR) {
                                SHADOW_NEAR
                            } else {
                                SHADOW_MOVING
                            };
                            let last = if timed_map(SHADOW_KEPT) {
                                SHADOW_KEPT
                            } else {
                                SHADOW_MOVING
                            };
                            effects[SHADOW_MAPS] = stamp(&bytes, query(first, 0))
                                .zip(stamp(&bytes, query(last, 1)))
                                .and_then(|(start, end)| end.checked_sub(start))
                                .map(|ticks| ticks as f64 * period_ns / 1_000_000.0)
                                .filter(|value| value.is_finite());
                        }
                        // The frame's own compute passes: every stage but the
                        // shift, which is submitted before the frame's uploads.
                        let first = (1..STAGES.len()).find(|&stage| timed(stage));
                        let last = (1..STAGES.len()).rev().find(|&stage| timed(stage));
                        let compute = first.zip(last).and_then(|(first, last)| {
                            let start = stamp(&bytes, 2 + 2 * first)?;
                            let end = stamp(&bytes, 3 + 2 * last)?;
                            end.checked_sub(start)
                                .map(|ticks| ticks as f64 * period_ns / 1_000_000.0)
                        });
                        (
                            elapsed(&bytes, 0),
                            compute.filter(|value| value.is_finite()),
                        )
                    });
            slot.readback.unmap();
            if Some(origin.capture) != self.capture {
                continue;
            }
            if let Some(duration_ms) = duration.filter(|value| value.is_finite()) {
                // reserve() accounts for completed and pending samples together.
                self.samples.push(Sample {
                    origin,
                    duration_ms,
                    compute_ms,
                    stages,
                    resolve_ms,
                    effects,
                });
            } else {
                self.errors = self.errors.saturating_add(1);
            }
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
            "scope": "opaque render pass, plus presentation_compute (the frame's compute passes) and per-stage presentation samples for GPU-presented frames (decision 014; stages run as separate passes while capturing), taa_resolve for temporally antialiased frames (decision 017), shadow_select and shadow_maps (decision 020), ambient_occlusion, bloom and post_composite (decision 021), reflections and sun_shafts (decisions 024 and 025), light_bins (decision 023), decals (decision 039), transparent and transparent_composite (decision 026), shadow_near, shadow_moving and shadow_kept per map, and occlusion (decision 040: the depth pyramid, test and second compaction; opaque_meshes then spans both drawing phases); excludes uploads, queue latency, presentation and timestamp readback",
            "clock": "GPU duration only; frame/submission IDs correlate origin, clocks are not aligned",
            "timestamp_period_ns": if self.supported { Some(self.period_ns) } else { None },
            "precision": "browser may quantize timestamp values; zero durations remain valid samples",
            "ring_slots": SLOTS,
            "sample_limit": SAMPLE_LIMIT,
            "attempted": self.attempted,
            "submitted": self.submitted,
            "completed": self.samples.len(),
            "pending": pending,
            "dropped": self.dropped,
            "errors": self.errors,
            "sample_limit_reached": self.sample_limit_reached,
            "complete": self.capture.is_some() && !active && self.supported && pending == 0 && self.dropped == 0 && self.errors == 0,
            "readback_buffer_bytes": self.slots.len() as u64 * QUERY_BYTES * 2,
            "cpu_sample_capacity_bytes": self.samples.capacity() * size_of::<Sample>(),
        })
    }

    pub(crate) fn report(&mut self) -> Value {
        self.poll();
        let mut report = self.status();
        let entry = |sample: &Sample, name: &str, duration_ms: f64| {
            json!({
                "capture_id": sample.origin.capture.to_string(),
                "frame": sample.origin.frame.to_string(),
                "submission": sample.origin.submission.to_string(),
                "name": name,
                "duration_ms": duration_ms,
            })
        };
        report["samples"] = Value::Array(
            self.samples
                .iter()
                .flat_map(|sample| {
                    let compute = sample
                        .compute_ms
                        .map(|compute| entry(sample, "presentation_compute", compute));
                    let stages = STAGES
                        .iter()
                        .zip(sample.stages)
                        .filter_map(|(name, value)| Some(entry(sample, name, value?)));
                    let resolve = sample
                        .resolve_ms
                        .map(|resolve| entry(sample, "taa_resolve", resolve));
                    let effects = EFFECTS
                        .iter()
                        .zip(sample.effects)
                        .filter_map(|(name, value)| Some(entry(sample, name, value?)));
                    std::iter::once(entry(sample, "opaque_meshes", sample.duration_ms))
                        .chain(compute)
                        .chain(stages)
                        .chain(resolve)
                        .chain(effects)
                        .collect::<Vec<_>>()
                })
                .collect(),
        );
        report
    }
}
