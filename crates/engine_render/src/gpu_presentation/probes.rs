//! Reflection probe captures: each cube face's cull and
//! compaction over the poses this frame's culling wrote, after the frame's
//! own drawing, and the still frames of every slot, which captures wait
//! for.

use engine_core::PoseRows;

use super::groups::PROBE;
use super::{GROUP, GpuPresentationPass, compact_group, entry, storage};
use crate::batches::DrawBatch;
use crate::gpu_data::{FLAG_PROBE, PresentationFrame, STILL_FRAMES, frustum_uniform};
use crate::mesh_arena::Ids;

// Draw group storage generation, pose storage generation and pose buffer
// index, and still frame storage generation, that a set of face groups
// binds.
type FacesKey = (u64, u64, usize, u64);

/// One cube face's cull and compaction groups.
struct FaceGroups {
    cull: wgpu::BindGroup,
    compact: wgpu::BindGroup,
}

/// Each slot's pose fingerprint and still frames while probes need them.
pub(super) struct Stillness {
    pub(super) frames: wgpu::Buffer,
    // Slots the frames cover (0 without probes), whether this frame counts
    // them (only while captures wait), and the frames counted in a row, up
    // to STILL_FRAMES.
    slots: u64,
    counting: bool,
    counted: u32,
    // Advances when `frames` is replaced.
    generation: u64,
}

impl Stillness {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        Self {
            frames: storage(
                device,
                "engine/presentation-still-frames",
                0,
                wgpu::BufferUsages::empty(),
            ),
            slots: 0,
            counting: false,
            counted: 0,
            generation: 0,
        }
    }

    /// Keeps still frames over every slot of `poses` while `stillness` is
    /// some, counting them this frame when it holds true (captures wait):
    /// storage replaced or released, and counting resumed, start the count
    /// over.
    pub(super) fn ensure(
        &mut self,
        device: &wgpu::Device,
        stillness: Option<bool>,
        poses: &wgpu::Buffer,
    ) {
        let slots = if stillness.is_some() {
            poses.size() / size_of::<PoseRows>() as u64
        } else {
            0
        };
        if slots > self.slots || (slots == 0 && self.slots != 0) {
            self.frames = storage(
                device,
                "engine/presentation-still-frames",
                slots * 8,
                wgpu::BufferUsages::empty(),
            );
            self.slots = slots;
            self.counted = 0;
            self.generation += 1;
        }
        let counting = stillness == Some(true);
        if counting && !self.counting {
            self.counted = 0;
        }
        self.counting = counting;
        if counting {
            self.counted = (self.counted + 1).min(STILL_FRAMES);
        }
    }

    /// Whether still frames exist, for slots while probes need them.
    pub(super) fn kept(&self) -> bool {
        self.slots != 0
    }

    /// Whether this frame counts still frames.
    pub(super) fn counting(&self) -> bool {
        self.counting
    }

    /// Advances when the still frames' storage is replaced.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether frames were counted long enough for captures to find every
    /// still slot.
    pub(super) fn settled(&self) -> bool {
        self.counting && self.counted >= STILL_FRAMES
    }
}

/// Every cube face's uniform, and its groups per pose buffer (temporally
/// antialiased frames alternate two), the latest of which this frame binds.
pub(super) struct ProbeFaces {
    uniforms: [wgpu::Buffer; 6],
    groups: Vec<(FacesKey, Vec<FaceGroups>)>,
    current: usize,
}

impl ProbeFaces {
    pub(super) fn new(uniforms: [wgpu::Buffer; 6]) -> Self {
        Self {
            uniforms,
            groups: Vec::new(),
            current: 0,
        }
    }
}

impl GpuPresentationPass {
    /// Records the cull and compaction of reflection probe cube face `face`,
    /// `size` texels across, seen through `view_projection`,
    /// over `poses` (this frame's culling wrote them), after the frame's own
    /// drawing; returns whether [`Self::probe_draw_source`] can draw it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_probe_face(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        face: usize,
        size: u32,
        view_projection: glam::Mat4,
        poses: &wgpu::Buffer,
    ) -> bool {
        // Prepared frames wrote a uniform and bound their groups.
        let Some(mut uniform) = self.frame_uniform else {
            return false;
        };
        if self.groups.count() == 0 {
            return false;
        }
        let (planes, absolute) = frustum_uniform(view_projection);
        uniform.planes = planes;
        uniform.absolute_view_projection = absolute;
        uniform.view_projection = view_projection.to_cols_array_2d();
        uniform.culling = 1;
        uniform.mask = FLAG_PROBE;
        uniform.occlusion = 0;
        // Candidates covering no texel center draw nothing.
        uniform.subpixel = 1;
        uniform.viewport = [size as f32, size as f32, 0.0, 0.0];
        uniform.motion = 0;
        uniform.budget = u32::MAX;
        // Probe faces draw every candidate at full detail.
        uniform.lod_error = 0.0;
        queue.write_buffer(
            &self.probes.uniforms[face],
            0,
            bytemuck::bytes_of::<PresentationFrame>(&uniform),
        );
        let (index, generation) = self.poses;
        let key = (
            self.groups.generation(),
            generation,
            index,
            self.stillness.generation(),
        );
        if let Some(index) = self
            .probes
            .groups
            .iter()
            .position(|(built, _)| *built == key)
        {
            self.probes.current = index;
        } else {
            // Stale generations never come back.
            self.probes.groups.retain(|((groups, poses, _, still), _)| {
                (*groups, *poses, *still) == (key.0, key.1, key.3)
            });
            let groups = (0..6)
                .map(|face| self.probe_face_groups(device, face, poses))
                .collect();
            self.probes.groups.push((key, groups));
            self.probes.current = self.probes.groups.len() - 1;
        }
        let groups = &self.probes.groups[self.probes.current].1[face];
        let candidates = self.groups.count();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/presentation-probe"),
            timestamp_writes: None,
        });
        pass.set_pipeline(self.shaders.probe_cull(device));
        pass.set_bind_group(0, &groups.cull, &[]);
        pass.dispatch_workgroups(candidates.div_ceil(GROUP), 1, 1);
        self.dispatch_compaction(&mut pass, &groups.compact);
        true
    }

    fn probe_face_groups(
        &self,
        device: &wgpu::Device,
        face: usize,
        poses: &wgpu::Buffer,
    ) -> FaceGroups {
        let frame = &self.probes.uniforms[face];
        let groups = &self.groups;
        FaceGroups {
            cull: self.shaders.group(
                device,
                &self.shaders.families.tests,
                "engine/presentation-probe-group",
                &[
                    entry("frame", frame),
                    entry("batches", &groups.batches),
                    entry("poses", poses),
                    entry("scan", &groups.scan),
                    entry("slot_candidates", &groups.slot_candidates),
                    entry("still_frames", &self.stillness.frames),
                ],
            ),
            // Probe faces count no triangles and draw no impostors.
            compact: compact_group(
                device,
                &self.shaders,
                groups,
                &self.occlusion_counts,
                "engine/presentation-probe-compact-group",
                frame,
                PROBE,
            ),
        }
    }

    /// Visible IDs and indirect arguments of plan batch `index` in the
    /// reflection probe face last recorded.
    pub(crate) fn probe_draw_source<'a>(
        &'a self,
        batch: &DrawBatch,
        index: usize,
    ) -> (Ids<'a>, &'a wgpu::Buffer, u64) {
        self.groups.draw_source(batch, index, PROBE, 0)
    }

    /// Each slot's pose fingerprint and still frames, and the
    /// generation of their storage.
    pub(crate) fn still_frames(&self) -> (&wgpu::Buffer, u64) {
        (&self.stillness.frames, self.stillness.generation())
    }

    /// Whether culling has counted still frames long enough for reflection
    /// probe captures to find every still slot.
    pub(crate) fn stillness_settled(&self) -> bool {
        self.stillness.settled()
    }
}
