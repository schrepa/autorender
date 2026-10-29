use engine_core::PoseRows;

use super::groups::PROBE_FACES;
use super::{GROUP, GpuPresentationPass, compact_group, entry, storage};
use crate::batches::DrawBatch;
use crate::gpu_data::{FLAG_PROBE, PresentationFrame, STILL_FRAMES, frustum_uniform};
use crate::mesh_arena::Ids;

type FacesKey = (u64, u64, usize, u64, usize);

struct FaceGroups {
    cull: wgpu::BindGroup,
    compact: wgpu::BindGroup,
}

pub(super) struct Stillness {
    pub(super) frames: wgpu::Buffer,
    slots: u64,
    counting: bool,
    counted: u32,
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
            crate::memory::release_buffer(&self.frames);
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

    pub(super) fn kept(&self) -> bool {
        self.slots != 0
    }

    pub(super) fn counting(&self) -> bool {
        self.counting
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn settled(&self) -> bool {
        self.counting && self.counted >= STILL_FRAMES
    }
}

#[derive(Default)]
pub(super) struct ViewFaces {
    uniforms: Vec<wgpu::Buffer>,
    groups: Vec<(FacesKey, Vec<FaceGroups>)>,
    current: usize,
}

impl GpuPresentationPass {
    #[expect(clippy::too_many_arguments, reason = "a pass's disjoint borrows")]
    pub(crate) fn encode_view(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        encoder: &mut wgpu::CommandEncoder,
        face: usize,
        size: u32,
        view_projection: glam::Mat4,
        poses: &wgpu::Buffer,
    ) -> bool {
        let Some(mut uniform) = self.frame_uniform else {
            return false;
        };
        if self.groups.count() == 0 {
            return false;
        }
        while self.probes.uniforms.len() <= face {
            self.probes.uniforms.push(crate::buffers::uniform(
                device,
                "engine/presentation-frame-probe",
                size_of::<PresentationFrame>() as u64,
            ));
        }
        let (planes, absolute) = frustum_uniform(view_projection);
        uniform.planes = planes;
        uniform.absolute_view_projection = absolute;
        uniform.view_projection = view_projection.to_cols_array_2d();
        uniform.culling = 1;
        uniform.mask = FLAG_PROBE;
        uniform.occlusion = 0;
        uniform.subpixel = 1;
        uniform.viewport = [size as f32, size as f32, 0.0, 0.0];
        uniform.motion = 0;
        uniform.budget = u32::MAX;
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
            self.probes.uniforms.len(),
        );
        if let Some(index) = self
            .probes
            .groups
            .iter()
            .position(|(built, _)| *built == key)
        {
            self.probes.current = index;
        } else {
            self.probes
                .groups
                .retain(|((groups, poses, _, still, views), _)| {
                    (*groups, *poses, *still, *views) == (key.0, key.1, key.3, key.4)
                });
            let groups = (0..self.probes.uniforms.len())
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
            compact: compact_group(
                device,
                &self.shaders,
                groups,
                &self.occlusion_counts,
                "engine/presentation-probe-compact-group",
                frame,
                PROBE_FACES,
            ),
        }
    }

    pub(crate) fn view_draw_source<'a>(
        &'a self,
        batch: &DrawBatch,
        index: usize,
    ) -> (Ids<'a>, &'a wgpu::Buffer, u64) {
        self.groups.draw_source(batch, index, PROBE_FACES, 0)
    }

    pub fn still_frames(&self) -> (&wgpu::Buffer, u64) {
        (&self.stillness.frames, self.stillness.generation())
    }

    pub fn stillness_settled(&self) -> bool {
        self.stillness.settled()
    }
}
