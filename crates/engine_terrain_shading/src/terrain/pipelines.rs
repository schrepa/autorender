use engine_render::wgpu;

pub(super) fn surface_primitive(front_face: wgpu::FrontFace) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face,
        cull_mode: Some(wgpu::Face::Back),
        ..Default::default()
    }
}
