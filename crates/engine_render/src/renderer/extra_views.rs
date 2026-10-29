use super::meshes::BatchBinds;
use super::*;

pub struct CasterDraw {
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub lods: Vec<(u32, u32, f32)>,
}

impl Core {
    pub fn view_group(&self, view: usize, without: &[&str]) -> &wgpu::BindGroup {
        self.camera.view_group(view, without)
    }

    pub fn views_settled(&self) -> bool {
        self.gpu_presentation.stillness_settled()
    }

    pub fn view_still_frames(&self) -> (&wgpu::Buffer, u64) {
        self.gpu_presentation.still_frames()
    }

    pub fn cull_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: usize,
        (view_projection, size): (glam::Mat4, u32),
        parity: usize,
    ) -> bool {
        if parity == 1 {
            self.instances.prepare_pose_group(&self.device, 1);
        }
        let Self {
            device,
            queue,
            instances,
            gpu_presentation,
            ..
        } = self;
        let poses = instances.frame_poses(parity);
        gpu_presentation.encode_view(
            device,
            queue,
            encoder,
            view,
            size,
            view_projection,
            poses.current(),
        )
    }

    pub fn draw_view(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        view: usize,
        parity: usize,
        lights: Option<&wgpu::BindGroup>,
    ) -> u32 {
        if self.mesh_entries.is_none() {
            return 0;
        }
        pass.set_bind_group(0, self.view_group(view, &["refraction"]), &[]);
        pass.set_bind_group(2, self.instances.pose_group(parity), &[]);
        let lights = lights.unwrap_or_else(|| self.shared_group("lights", &[]));
        pass.set_bind_group(3, lights, &[]);
        let kind = PassKind::view();
        let mut binds = BatchBinds::default();
        let mut draws = 0;
        for &index in &self.draw_order.order {
            let index = index as usize;
            let batch = &self.plan.batches[index];
            self.bind_batch(pass, &mut binds, (kind, true), batch);
            let (ids, buffer, offset) = self.gpu_presentation.view_draw_source(batch, index);
            binds.bindings.ids(pass, 1, ids);
            pass.draw_indexed_indirect(buffer, offset);
            draws += 1;
        }
        draws
    }

    pub fn caster_draw(&self, mesh: MeshHandle, levels: bool) -> CasterDraw {
        self.meshes.get(&mesh).map_or(
            CasterDraw {
                index_count: 0,
                first_index: 0,
                base_vertex: 0,
                lods: Vec::new(),
            },
            |mesh| {
                let draw = mesh.draw(&self.mesh_arena);
                let lods = if !levels {
                    Vec::new()
                } else {
                    mesh.lods
                        .iter()
                        .enumerate()
                        .map(|(level, &(offset, count, error))| {
                            let (offset, count) = mesh.caster_range(level + 1, (offset, count));
                            (draw.first_index + offset, count, error)
                        })
                        .collect()
                };
                let (offset, count) = mesh.caster_range(0, (0, draw.index_count));
                CasterDraw {
                    index_count: count,
                    first_index: draw.first_index + offset,
                    base_vertex: draw.base_vertex,
                    lods,
                }
            },
        )
    }
}
