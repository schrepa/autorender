use engine_core::{
    RenderReflectionProbe,
    glam::{Mat4, Vec3, Vec4},
    math,
};

use crate::cached::{Bound, Cached};
use crate::gpu_data::STILL_FRAMES;
use crate::gpu_layouts::constants::PROBE_ENTRY_WORDS;
use crate::render_features::{Encode, RenderFeature};
use crate::render_graph::{FrameShape, Node, RenderGraph};

pub(crate) const FACES: usize = 6;
const CAPTURE_NEAR: f32 = 0.05;
const CAPTURE_FAR: f32 = 4_000.0;
const GRID_CELLS: f32 = 64.0;

struct Slot {
    probe: RenderReflectionProbe,
    ready: bool,
    stale: bool,
    age: u32,
}

#[derive(Default)]
pub(crate) struct ProbeSlots {
    slots: Vec<Option<Slot>>,
    capturing: Option<(usize, usize)>,
    changed: bool,
    changed_during: bool,
}

impl ProbeSlots {
    pub(crate) fn plan(
        &mut self,
        probes: &[RenderReflectionProbe],
        capacity: usize,
        lost: bool,
        sky_changed: bool,
    ) {
        if lost {
            self.slots.clear();
            self.capturing = None;
            self.changed = true;
        }
        if self.slots.len() != capacity {
            self.slots.resize_with(capacity, || None);
            if self
                .capturing
                .is_some_and(|(capturing, _)| capturing >= capacity)
            {
                self.capturing = None;
            }
            self.changed = true;
        }
        let wanted = &probes[..probes.len().min(capacity)];
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let keep = slot.as_ref().is_some_and(|slot| {
                wanted.iter().any(|probe| {
                    probe.key == slot.probe.key
                        && probe.center == slot.probe.center
                        && probe.extent == slot.probe.extent
                })
            });
            if !keep && slot.is_some() {
                *slot = None;
                self.changed = true;
                if self
                    .capturing
                    .is_some_and(|(capturing, _)| capturing == index)
                {
                    self.capturing = None;
                }
            }
            if let Some(slot) = slot {
                if sky_changed {
                    slot.stale = true;
                }
                if let Some(probe) = wanted.iter().find(|probe| probe.key == slot.probe.key)
                    && (probe.blend != slot.probe.blend || probe.interior != slot.probe.interior)
                {
                    slot.probe.blend = probe.blend;
                    slot.probe.interior = probe.interior;
                    self.changed = true;
                }
            }
        }
        for probe in wanted {
            if self
                .slots
                .iter()
                .flatten()
                .any(|slot| slot.probe.key == probe.key)
            {
                continue;
            }
            if let Some(free) = self.slots.iter_mut().find(|slot| slot.is_none()) {
                *free = Some(Slot {
                    probe: *probe,
                    ready: false,
                    stale: false,
                    age: 0,
                });
                self.changed |= probe.interior.is_some();
            }
        }
        if sky_changed && self.capturing.is_some() {
            self.changed_during = true;
        }
    }

    pub(crate) fn tick(&mut self) {
        for slot in self.slots.iter_mut().flatten() {
            slot.age = (slot.age + 1).min(STILL_FRAMES);
        }
    }

    pub(crate) fn next_face(
        &mut self,
        probes: &[RenderReflectionProbe],
    ) -> Option<(u32, usize, RenderReflectionProbe)> {
        if self.capturing.is_none() {
            let waiting = probes.iter().find_map(|probe| {
                self.slots.iter().position(|slot| {
                    slot.as_ref().is_some_and(|slot| {
                        slot.probe.key == probe.key
                            && (!slot.ready || slot.stale)
                            && slot.age >= STILL_FRAMES
                    })
                })
            })?;
            self.capturing = Some((waiting, 0));
        }
        let (slot, face) = self.capturing?;
        let probe = self.slots[slot].as_ref()?.probe;
        Some((slot as u32 + 1, face, probe))
    }

    pub(crate) fn face_drawn(&mut self) -> bool {
        let Some((slot, face)) = self.capturing else {
            return false;
        };
        if face + 1 < FACES {
            self.capturing = Some((slot, face + 1));
            return false;
        }
        self.capturing = None;
        let stale = std::mem::take(&mut self.changed_during);
        if let Some(slot) = self.slots[slot].as_mut() {
            slot.ready = true;
            slot.stale = stale;
            self.changed = true;
        }
        true
    }

    pub(crate) fn waiting(&self) -> bool {
        self.capturing.is_some()
            || self
                .slots
                .iter()
                .flatten()
                .any(|slot| !slot.ready || slot.stale)
    }

    pub(crate) fn ready(&self) -> u32 {
        self.slots
            .iter()
            .flatten()
            .filter(|slot| slot.ready)
            .count() as u32
    }

    fn changed_ready(&mut self) -> Option<Vec<(RenderReflectionProbe, u32)>> {
        if !std::mem::take(&mut self.changed) {
            return None;
        }
        Some(
            self.slots
                .iter()
                .enumerate()
                .filter_map(|(index, slot)| {
                    slot.as_ref()
                        .filter(|slot| slot.ready || slot.probe.interior.is_some())
                        .map(|slot| (slot.probe, if slot.ready { index as u32 + 1 } else { 0 }))
                })
                .collect(),
        )
    }
}

pub(crate) struct ProbePass {
    pub(crate) slots: ProbeSlots,
    cameras: [wgpu::Buffer; FACES],
    depth: Cached<u32, wgpu::TextureView>,
    words: crate::buffers::Growable,
    staging: Vec<[u32; 4]>,
    pub(crate) faces: u32,
    pub(crate) dropped: u32,
}

impl ProbePass {
    pub(crate) fn new(device: &wgpu::Device, camera_bytes: u64) -> Self {
        let _owner = crate::memory::owned(crate::render_graph::Pass::ProbeCaptures);
        let cameras = std::array::from_fn(|_| {
            crate::buffers::uniform(device, "engine/probe-camera", camera_bytes)
        });
        Self {
            slots: ProbeSlots {
                changed: true,
                ..ProbeSlots::default()
            },
            cameras,
            depth: Cached::default(),
            words: crate::buffers::Growable::new(
                device,
                GRID_LABEL,
                32,
                wgpu::BufferUsages::empty(),
            ),
            staging: Vec::new(),
            faces: 0,
            dropped: 0,
        }
    }

    pub(crate) fn camera(&self, face: usize) -> &wgpu::Buffer {
        &self.cameras[face]
    }

    pub(crate) fn words(&self) -> Bound<'_> {
        self.words.bound()
    }

    pub(crate) fn depth(
        &mut self,
        device: &wgpu::Device,
        size: u32,
        format: wgpu::TextureFormat,
    ) -> &wgpu::TextureView {
        self.depth.get_or_insert_with(size, || {
            let texture = crate::targets::texture_2d(
                device,
                "engine/probe-depth",
                (size, size),
                1,
                1,
                format,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        })
    }

    pub(crate) fn upload(&mut self, device: &wgpu::Device, queue: &crate::gpu_calls::Queue) {
        let Some(ready) = self.slots.changed_ready() else {
            return;
        };
        grid_words(&ready, &mut self.staging);
        let bytes = (self.staging.len() * 16) as u64;
        self.words.grow(device, bytes);
        queue.write_buffer(&self.words, 0, bytemuck::cast_slice(&self.staging));
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.words.size()
            + self
                .depth
                .key()
                .map_or(0, |size| u64::from(*size) * u64::from(*size) * 4)
    }
}

const GRID_LABEL: &str = "engine/probe-grid";

pub(crate) fn face_view_projection(center: Vec3, face: usize) -> Mat4 {
    let (right, up, back) = match face {
        0 => (Vec3::NEG_Z, Vec3::Y, Vec3::NEG_X),
        1 => (Vec3::Z, Vec3::Y, Vec3::X),
        2 => (Vec3::X, Vec3::NEG_Z, Vec3::NEG_Y),
        3 => (Vec3::X, Vec3::Z, Vec3::Y),
        4 => (Vec3::X, Vec3::Y, Vec3::NEG_Z),
        _ => (Vec3::NEG_X, Vec3::Y, Vec3::Z),
    };
    let row = |axis: Vec3| axis.extend(-axis.dot(center));
    let view = Mat4::from_cols(row(right), row(up), row(back), Vec4::W).transpose();
    Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, CAPTURE_FAR, CAPTURE_NEAR) * view
}

fn grid_words(probes: &[(RenderReflectionProbe, u32)], words: &mut Vec<[u32; 4]>) {
    words.clear();
    if probes.is_empty() {
        words.extend([[0; 4], [0; 4]]);
        return;
    }
    let (mut low, mut high) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let mut smallest = f32::INFINITY;
    for (probe, _) in probes {
        low = low.min(probe.center - probe.extent);
        high = high.max(probe.center + probe.extent);
        smallest = math::min(smallest, math::min(probe.extent.x, probe.extent.z) * 2.0);
    }
    let span = high - low;
    let cell = math::max(smallest * 0.5, math::max(span.x, span.z) / GRID_CELLS);
    let across = |length: f32| ((length / cell).ceil() as u32).max(1);
    let (columns, rows) = (across(span.x), across(span.z));
    let volume = |index: usize| {
        let extent = probes[index].0.extent;
        extent.x * extent.y * extent.z
    };
    let mut order: Vec<usize> = (0..probes.len()).collect();
    order.sort_by(|&a, &b| volume(a).total_cmp(&volume(b)).then(a.cmp(&b)));
    let mut lists = vec![Vec::new(); (columns * rows) as usize];
    for &index in &order {
        let probe = probes[index].0;
        let first = ((probe.center - probe.extent - low) / cell).floor();
        let last = ((probe.center + probe.extent - low) / cell).floor();
        let rows_reached = (math::max(first.z, 0.0) as u32)..=(last.z as u32).min(rows - 1);
        for row in rows_reached {
            for column in (math::max(first.x, 0.0) as u32)..=(last.x as u32).min(columns - 1) {
                lists[(row * columns + column) as usize].push(index as u32);
            }
        }
    }
    let count = probes.len() as u32;
    let cells_at = 2;
    let entries_at = cells_at + columns * rows;
    words.push([low.x.to_bits(), low.z.to_bits(), cell.to_bits(), count]);
    words.push([columns, rows, cells_at, entries_at]);
    let mut start = 0;
    for list in &lists {
        let interiors = list
            .iter()
            .filter(|&&index| probes[index as usize].0.interior.is_some())
            .count() as u32;
        words.push([start, list.len() as u32, interiors, 0]);
        start += list.len() as u32;
    }
    for &index in lists.iter().flatten() {
        let (probe, cube) = probes[index as usize];
        let interior = probe.interior;
        let entry: [[u32; 4]; PROBE_ENTRY_WORDS] = [
            [
                probe.center.x.to_bits(),
                probe.center.y.to_bits(),
                probe.center.z.to_bits(),
                probe.blend.to_bits(),
            ],
            [
                probe.extent.x.to_bits(),
                probe.extent.y.to_bits(),
                probe.extent.z.to_bits(),
                cube,
            ],
            [
                interior.map_or(1.0, |interior| interior.sky).to_bits(),
                interior.map_or(1.0, |interior| interior.sun).to_bits(),
                u32::from(interior.is_some()),
                u32::from(cube != 0),
            ],
        ];
        words.extend(entry);
    }
}

impl RenderFeature for ProbePass {
    type Frame = ();
    type Stats = engine_core::ProbeStats;

    fn declare(_frame: &(), _shape: &FrameShape, _graph: &mut RenderGraph) {}

    fn encode(
        &self,
        _frame: &(),
        _node: &Node,
        _encode: &Encode<'_>,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        0
    }

    fn capacity_bytes(&self) -> u64 {
        ProbePass::capacity_bytes(self)
    }

    fn stats(
        this: Option<&Self>,
        _frame: Option<&()>,
        _snapshot: &engine_core::RenderSnapshot,
    ) -> engine_core::ProbeStats {
        this.map_or_else(engine_core::ProbeStats::default, |probes| {
            engine_core::ProbeStats {
                ready: probes.slots.ready(),
                faces: probes.faces,
                dropped: probes.dropped,
            }
        })
    }
}
