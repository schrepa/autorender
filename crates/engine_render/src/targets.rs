pub fn texture_2d(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    layers: u32,
    levels: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    crate::gpu_calls::create_texture(device, &wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

pub use crate::graph::Clear as TargetClear;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedTarget {
    pub name: engine_core::TargetName,
    pub format: wgpu::TextureFormat,
    pub samples: u32,
    pub clear: TargetClear,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
}

impl FrameSize {
    pub fn pixels(self) -> (u32, u32) {
        (self.width, self.height)
    }
}
