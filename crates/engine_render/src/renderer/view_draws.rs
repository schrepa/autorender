use super::*;
use crate::entries::PassKind;

impl Core {
    pub fn views_settled(&self) -> bool {
        self.gpu_presentation.stillness_settled()
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
        let poses = self.instances.frame_poses(parity);
        self.gpu_presentation.encode_probe_face(
            &self.device,
            &self.queue,
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
    ) {
        pass.set_bind_group(0, self.view_group(view, &["refraction"]), &[]);
        pass.set_bind_group(2, self.instances.pose_group(parity), &[]);
        let lights = lights.unwrap_or_else(|| self.shared_group("lights", &[]));
        pass.set_bind_group(3, lights, &[]);
        let kind = PassKind::view();
        let mut binds = BatchBinds::default();
        for &index in &self.draw_order.order {
            let index = index as usize;
            let batch = &self.plan.batches[index];
            self.bind_batch(pass, &mut binds, (kind, true), batch);
            let (ids, draws, offset) = self.gpu_presentation.probe_draw_source(batch, index);
            binds.bindings.ids(pass, 1, ids);
            pass.draw_indexed_indirect(draws, offset);
        }
    }
}
