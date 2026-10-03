//! Reflection probes (decision 043): the probes nearest the camera hold
//! cubes of the environment maps after the sky's, captured face by face
//! after frames draw and blurred like the sky's; a grid over the ground plane
//! lists which probes surfaces in each of its cells read.

use engine_core::{
    RenderReflectionProbe,
    glam::{Mat4, Vec3, Vec4},
    math,
};

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
                // A new blend distance needs no capture, only the grid.
                if let Some(probe) = wanted.iter().find(|probe| probe.key == slot.probe.key)
                    && probe.blend != slot.probe.blend
                {
                    slot.probe.blend = probe.blend;
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

    // The ready probes and their cubes, once per change.
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
                        .filter(|slot| slot.ready)
                        .map(|slot| (slot.probe, index as u32 + 1))
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
    // The grid words surfaces read (`meshes.wgsl`'s `probe_words`).
    words: wgpu::Buffer,
    staging: Vec<[u32; 4]>,
    // Advances when `words` is replaced: groups binding it rebuild.
    generation: u64,
}

#[cfg(target_arch = "wasm32")]
impl ProbePass {
    pub(crate) fn new(device: &wgpu::Device, camera_bytes: u64) -> Self {
        let cameras = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/probe-camera"),
                size: camera_bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Self {
            slots: ProbeSlots {
                changed: true,
                ..ProbeSlots::default()
            },
            cameras,
            depth: Cached::default(),
            words: words_buffer(device, 2),
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
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("engine/probe-depth"),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
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
        if bytes > self.words.size() {
            self.words = words_buffer(device, self.staging.len().next_power_of_two());
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
fn words_buffer(device: &wgpu::Device, words: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("engine/probe-grid"),
        size: (words.max(2) * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

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

/// Fills `words` with the grid of `probes` (each with its environment cube)
/// that `meshes.wgsl`'s `environment_light` reads.
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
    let mut start = 0;
    for list in &lists {
        words.push([start, list.len() as u32, 0, 0]);
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(key: u64, center: Vec3, extent: Vec3) -> RenderReflectionProbe {
        RenderReflectionProbe {
            key,
            center,
            extent,
            blend: 1.0,
        }
    }

    #[test]
    fn faces_look_along_webgpu_cube_directions() {
        // WebGPU reads face texel (s, t) along these directions, with u and v
        // running 2s - 1 and 2t - 1.
        let direction = |face: usize, u: f32, v: f32| match face {
            0 => Vec3::new(1.0, -v, -u),
            1 => Vec3::new(-1.0, -v, u),
            2 => Vec3::new(u, 1.0, v),
            3 => Vec3::new(u, -1.0, -v),
            4 => Vec3::new(u, -v, 1.0),
            _ => Vec3::new(-u, -v, -1.0),
        };
        let center = Vec3::new(3.0, 2.0, -7.0);
        for face in 0..FACES {
            let view_projection = face_view_projection(center, face);
            for (u, v) in [(0.0, 0.0), (0.5, -0.25), (-0.75, 0.6)] {
                let point = center + direction(face, u, v) * 10.0;
                let clip = view_projection * point.extend(1.0);
                // Pixel rows run down while NDC y runs up.
                let (x, y) = (clip.x / clip.w, clip.y / clip.w);
                assert!((x - u).abs() < 1.0e-4, "face {face}: x {x} for u {u}");
                assert!((y + v).abs() < 1.0e-4, "face {face}: y {y} for v {v}");
                let depth = clip.z / clip.w;
                assert!(depth > 0.0 && depth < 1.0, "in front, reversed");
            }
        }
    }

    #[test]
    fn faces_wind_front_facing_triangles_clockwise() {
        let forward = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        let center = Vec3::new(1.0, 2.0, 3.0);
        for (face, forward) in forward.into_iter().enumerate() {
            // A triangle ahead of the face, counterclockwise seen from the
            // probe (its normal points back at it).
            let side = forward.any_orthonormal_vector();
            let other = forward.cross(side);
            let ahead = center + forward * 5.0;
            let mut corners = [ahead + side, ahead + other, ahead - side - other];
            let normal = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
            if normal.dot(forward) > 0.0 {
                corners.swap(1, 2);
            }
            let view_projection = face_view_projection(center, face);
            let ndc: Vec<_> = corners
                .iter()
                .map(|corner| {
                    let clip = view_projection * corner.extend(1.0);
                    (clip.x / clip.w, clip.y / clip.w)
                })
                .collect();
            let area = (ndc[1].0 - ndc[0].0) * (ndc[2].1 - ndc[0].1)
                - (ndc[2].0 - ndc[0].0) * (ndc[1].1 - ndc[0].1);
            assert!(
                area < 0.0,
                "face {face}: front faces wind clockwise, area {area}"
            );
        }
    }

    #[test]
    fn plans_keep_captures_of_probes_that_stay() {
        let a = probe(1, Vec3::ZERO, Vec3::splat(10.0));
        let b = probe(2, Vec3::X * 20.0, Vec3::splat(10.0));
        let c = probe(3, Vec3::X * 40.0, Vec3::splat(10.0));
        let mut slots = ProbeSlots::default();
        slots.plan(&[a, b], 2, false, false);
        // Nothing is captured until objects appearing with the probes count
        // as still.
        for _ in 1..STILL_FRAMES {
            slots.tick();
        }
        assert!(slots.next_face(&[a, b]).is_none());
        slots.tick();
        // The nearest waits first, face by face.
        for face in 0..FACES {
            let (cube, drawn, next) = slots.next_face(&[a, b]).unwrap();
            assert_eq!((cube, drawn, next.key), (1, face, 1));
            assert_eq!(slots.face_drawn(), face + 1 == FACES);
        }
        assert_eq!(slots.ready(), 1);
        assert!(slots.waiting(), "b waits");
        // The grid lists ready probes with their cubes, once per change.
        let listed = |slots: &mut ProbeSlots| {
            slots.changed_ready().map(|ready| {
                ready
                    .iter()
                    .map(|(probe, cube)| (probe.key, *cube))
                    .collect::<Vec<_>>()
            })
        };
        assert_eq!(listed(&mut slots), Some(vec![(1, 1)]));
        assert_eq!(listed(&mut slots), None);
        // A newcomer nearer than b takes b's cube; a keeps its capture.
        slots.plan(&[a, c], 2, false, false);
        assert_eq!(slots.ready(), 1);
        for _ in 0..STILL_FRAMES {
            slots.tick();
        }
        let (cube, face, next) = slots.next_face(&[a, c]).unwrap();
        assert_eq!((cube, face, next.key), (2, 0, 3));
        // A sky change keeps a reflected until recaptured, after c.
        slots.plan(&[a, c], 2, false, true);
        assert_eq!(slots.ready(), 1);
        for _ in 0..FACES {
            slots.face_drawn();
        }
        let (cube, face, next) = slots.next_face(&[a, c]).unwrap();
        assert_eq!((cube, face, next.key), (1, 0, 1));
        // Replaced maps lose every capture.
        slots.plan(&[a, c], 2, true, false);
        assert_eq!(slots.ready(), 0);
    }

    #[test]
    fn grids_list_the_probes_reaching_each_cell_smallest_first() {
        let big = probe(1, Vec3::ZERO, Vec3::new(20.0, 5.0, 20.0));
        let small = probe(2, Vec3::new(5.0, 0.0, 5.0), Vec3::splat(2.0));
        let mut words = Vec::new();
        grid_words(&[(big, 1), (small, 2)], &mut words);
        let [x, z, cell, count] = words[0];
        assert_eq!(count, 2);
        let (x, z, cell) = (f32::from_bits(x), f32::from_bits(z), f32::from_bits(cell));
        assert_eq!((x, z), (-20.0, -20.0));
        let [columns, rows, cells_at, entries_at] = words[1];
        assert!(rows >= 1 && columns >= 1);
        // The cell under the small probe lists it before the big one, each
        // entry with its probe's center, blend, extent and cube.
        let column = ((5.0 - x) / cell) as u32;
        let row = ((5.0 - z) / cell) as u32;
        let [start, length, ..] = words[(cells_at + row * columns + column) as usize];
        assert_eq!(length, 2);
        let entry = |item: u32| {
            let at = (entries_at + item * 2) as usize;
            let [cx, cy, cz, blend] = words[at].map(f32::from_bits);
            let [ex, ey, ez, _] = words[at + 1].map(f32::from_bits);
            (
                Vec3::new(cx, cy, cz),
                blend,
                Vec3::new(ex, ey, ez),
                words[at + 1][3],
            )
        };
        assert_eq!(entry(start), (small.center, 1.0, small.extent, 2));
        assert_eq!(entry(start + 1), (big.center, 1.0, big.extent, 1));
    }
}
