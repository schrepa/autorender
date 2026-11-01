use engine_render::wgpu;

pub(super) const CASTER_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState {
    constant: 2,
    slope_scale: 2.5,
    clamp: 0.0,
};

pub(super) fn surface_primitive(front_face: wgpu::FrontFace) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        front_face,
        cull_mode: Some(wgpu::Face::Back),
        ..Default::default()
    }
}
