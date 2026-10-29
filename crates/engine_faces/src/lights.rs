use engine_render::gpu_calls::Queue;
use glam::Mat4;

pub trait LightListFace {
    fn probe_group(&self, face: usize) -> Option<wgpu::BindGroup>;

    fn encode_probe_face(
        &self,
        gpu: (&wgpu::Device, &Queue),
        encoder: &mut wgpu::CommandEncoder,
        view: (usize, Mat4, u32),
    );
}
