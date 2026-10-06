//! Reflection probes (decision 043): the probes nearest the camera hold
//! cubes of the environment maps after the sky's, captured face by face
//! after frames draw and blurred like the sky's; a grid over the ground plane
//! lists which probes surfaces in each of its cells read.

use engine_core::{
    RenderReflectionProbe,
    glam::{Mat4, Vec3, Vec4},
    math,
};

#[cfg(target_arch = "wasm32")]
use crate::cached::Cached;
use crate::gpu_data::STILL_FRAMES;

/// Cube faces, in the order WebGPU samples them: +X, -X, +Y, -Y, +Z, -Z.
pub(crate) const FACES: usize = 6;
/// Near and far planes of captures, in world units: probes stand close to
/// what they reflect and see the whole city.
const CAPTURE_NEAR: f32 = 0.05;
const CAPTURE_FAR: f32 = 4_000.0;
/// Grid cells across the probes' ground span at most: finer cells list fewer
/// probes each.
const GRID_CELLS: f32 = 64.0;

struct Slot {
    probe: RenderReflectionProbe,
    // All six faces captured and blurred.
    ready: bool,
    // Captured before the sky last changed: reflected until recaptured.
    stale: bool,
    // Frames since the probe took its cube, up to STILL_FRAMES: objects
    // appearing with it count as still only after as many frames.
    age: u32,
}

/// Which probe holds each environment cube after the sky's, and the capture
/// in progress.
#[derive(Default)]
pub(crate) struct ProbeSlots {
    slots: Vec<Option<Slot>>,
    // The slot being captured and the faces drawn.
    capturing: Option<(usize, usize)>,
    // The ready probes changed since the grid was last built.
    changed: bool,
}

impl ProbeSlots {
    /// Assigns the first `capacity` of `probes` (nearest first) to cubes:
    /// probes leaving free theirs, arrivals take free ones and wait for
    /// capture. `lost` discards every capture (the maps were replaced); after
    /// a sky change captures stay reflected until recaptured.
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
                // A new blend distance or interior (decision 079) needs no
                // capture, only the grid.
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
                // Interiors light surfaces before their first capture.
                self.changed |= probe.interior.is_some();
            }
        }
        // A capture the sky changed under starts over.
        if sky_changed && let Some((slot, _)) = self.capturing {
            self.capturing = Some((slot, 0));
        }
    }

    /// Ages every probe holding a cube by a frame in which culling counted
    /// still slots.
    pub(crate) fn tick(&mut self) {
        for slot in self.slots.iter_mut().flatten() {
            slot.age = (slot.age + 1).min(STILL_FRAMES);
        }
    }

    /// The next face to capture: its environment cube, face and probe. The
    /// nearest probe waiting (never captured, or captured before the sky
    /// last changed) that held its cube for STILL_FRAMES frames starts when
    /// none is in progress.
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

    /// Records that the face [`Self::next_face`] returned was drawn; returns
    /// whether it was the last, so the probe's cube is to be blurred and
    /// becomes ready.
    pub(crate) fn face_drawn(&mut self) -> bool {
        let Some((slot, face)) = self.capturing else {
            return false;
        };
        if face + 1 < FACES {
            self.capturing = Some((slot, face + 1));
            return false;
        }
        self.capturing = None;
        if let Some(slot) = self.slots[slot].as_mut() {
            slot.ready = true;
            slot.stale = false;
            self.changed = true;
        }
        true
    }

    /// Whether a probe holding a cube waits for its capture (never
    /// captured, captured before the sky last changed, or in progress).
    pub(crate) fn waiting(&self) -> bool {
        self.capturing.is_some()
            || self
                .slots
                .iter()
                .flatten()
                .any(|slot| !slot.ready || slot.stale)
    }

    /// Probes whose captures surfaces read.
    pub(crate) fn ready(&self) -> u32 {
        self.slots
            .iter()
            .flatten()
            .filter(|slot| slot.ready)
            .count() as u32
    }

    // The ready probes and their cubes, and interiors (decision 079) not
    // captured yet with none (0), once per change.
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

#[cfg(target_arch = "wasm32")]
pub(crate) struct ProbePass {
    pub(crate) slots: ProbeSlots,
    // The camera of each face being captured.
    cameras: [wgpu::Buffer; FACES],
    // Depth of the face being captured, and its size.
    depth: Cached<u32, wgpu::TextureView>,
    // The grid words surfaces read (`meshes/declarations.wgsl`'s `probe_words`).
    words: wgpu::Buffer,
    staging: Vec<[u32; 4]>,
    // Advances when `words` is replaced: groups binding it rebuild.
    generation: u64,
}

#[cfg(target_arch = "wasm32")]
impl ProbePass {
    pub(crate) fn new(device: &wgpu::Device, camera_bytes: u64) -> Self {
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
            words: crate::buffers::storage(device, GRID_LABEL, 32, wgpu::BufferUsages::empty()),
            staging: Vec::new(),
            generation: 0,
        }
    }

    /// The camera uniform buffer of face `face`.
    pub(crate) fn camera(&self, face: usize) -> &wgpu::Buffer {
        &self.cameras[face]
    }

    /// The grid words surfaces read, and their generation.
    pub(crate) fn words(&self) -> (&wgpu::Buffer, u64) {
        (&self.words, self.generation)
    }

    /// The `format` depth target of a face `size` texels across.
    pub(crate) fn depth(
        &mut self,
        device: &wgpu::Device,
        size: u32,
        format: wgpu::TextureFormat,
    ) -> &wgpu::TextureView {
        if self.depth.stale(&size) {
            let texture = crate::targets::texture_2d(
                device,
                "engine/probe-depth",
                (size, size),
                1,
                1,
                format,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.depth.set(size, view);
        }
        self.depth.get().expect("created above")
    }

    /// Uploads the grid of the ready probes when they changed.
    pub(crate) fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(ready) = self.slots.changed_ready() else {
            return;
        };
        grid_words(&ready, &mut self.staging);
        let bytes = (self.staging.len() * 16) as u64;
        if crate::buffers::ensure_storage(
            device,
            &mut self.words,
            GRID_LABEL,
            bytes,
            wgpu::BufferUsages::empty(),
        ) {
            self.generation += 1;
        }
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

#[cfg(target_arch = "wasm32")]
const GRID_LABEL: &str = "engine/probe-grid";

/// The view-projection of cube face `face` (in WebGPU's sampling order and
/// orientation) seen from `center`, with reversed depth (decision 042). Cube
/// faces mirror the world, so it flips triangle winding.
pub(crate) fn face_view_projection(center: Vec3, face: usize) -> Mat4 {
    // Each face's image axes and its backward direction (view z).
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

/// Fills `words` with the grid of `probes` (each with its environment cube,
/// or 0 for an interior not captured yet) that `meshes/shading.wgsl`'s
/// `environment_light` and `interior_light` read.
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
    // Each cell's probes, the smallest boxes first.
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
    // Each cell's list, and how many of its probes are interiors (decision
    // 079), which surfaces elsewhere skip.
    let mut start = 0;
    for list in &lists {
        let interiors = list
            .iter()
            .filter(|&&index| probes[index as usize].0.interior.is_some())
            .count() as u32;
        words.push([start, list.len() as u32, interiors, 0]);
        start += list.len() as u32;
    }
    // Each entry holds its probe's record: surfaces read no index first.
    for &index in lists.iter().flatten() {
        let (probe, cube) = probes[index as usize];
        words.push([
            probe.center.x.to_bits(),
            probe.center.y.to_bits(),
            probe.center.z.to_bits(),
            probe.blend.to_bits(),
        ]);
        words.push([
            probe.extent.x.to_bits(),
            probe.extent.y.to_bits(),
            probe.extent.z.to_bits(),
            cube,
        ]);
        let interior = probe.interior;
        words.push([
            interior.map_or(1.0, |interior| interior.sky).to_bits(),
            interior.map_or(1.0, |interior| interior.sun).to_bits(),
            u32::from(interior.is_some()),
            u32::from(cube != 0),
        ]);
    }
}
