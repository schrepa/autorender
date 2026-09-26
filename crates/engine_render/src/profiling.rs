//! Optional, bounded GPU timestamp readback. WebGPU completes maps through its
//! event loop; neither rendering nor capture settlement ever waits on the GPU.

use std::{cell::Cell, rc::Rc};

use serde_json::{Value, json};

const SLOTS: usize = 3;
const SAMPLE_LIMIT: usize = 4_096;
// Render pass begin/end, then the optional presentation compute pass.
const QUERIES: u32 = 4;
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
    // Whether this submission also timed the presentation compute pass.
    compute: bool,
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
            compute: false,
        }
    }
}

struct Sample {
    origin: Origin,
    duration_ms: f64,
    compute_ms: Option<f64>,
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
        slot.compute = false;
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

    /// Times the presentation compute pass of a GPU-presented frame.
    pub(crate) fn compute_writes(
        &mut self,
        index: Option<usize>,
    ) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let slot = &mut self.slots[index?];
        slot.compute = true;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &slot.queries,
            beginning_of_pass_write_index: Some(2),
            end_of_pass_write_index: Some(3),
        })
    }

    pub(crate) fn resolve(&self, index: Option<usize>, encoder: &mut wgpu::CommandEncoder) {
        if let Some(index) = index {
            let slot = &self.slots[index];
            let queries = if slot.compute { QUERIES } else { 2 };
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
            let (duration, compute_ms) =
                slot.readback
                    .get_mapped_range(..)
                    .ok()
                    .map_or((None, None), |bytes| {
                        let compute = slot.compute.then(|| elapsed(&bytes, 16)).flatten();
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
            "scope": "opaque render pass, plus separate presentation_compute samples for GPU-presented frames (decision 014); excludes uploads, queue latency, presentation and timestamp readback",
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
                    std::iter::once(entry(sample, "opaque_meshes", sample.duration_ms))
                        .chain(compute)
                })
                .collect(),
        );
        report
    }
}
