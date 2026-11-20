use engine_render::{Allocation, GpuPart, Install, Lifetime, Prepare, Texture, wgpu};

const TEXEL: wgpu::Extent3d = wgpu::Extent3d {
    width: 1,
    height: 1,
    depth_or_array_layers: 1,
};

pub struct StandIn<P> {
    texture: Texture<P>,
    texel: [u8; 4],
}

impl<P: GpuPart> StandIn<P> {
    pub fn install(cx: &mut Install<'_, P>, name: &'static str, texel: [u8; 4]) -> Self {
        let texture = cx.texture(name, wgpu::TextureFormat::Rgba8Unorm, Lifetime::Kept);
        Self { texture, texel }
    }

    pub fn written(&self, cx: &mut Prepare<'_, P>) -> bool {
        if cx.allocate(self.texture, TEXEL) != Allocation::Fresh {
            return false;
        }
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        };
        cx.write_texture(self.texture, wgpu::Origin3d::ZERO, &self.texel, layout, TEXEL);
        true
    }

    pub fn texture(&self) -> Texture<P> {
        self.texture
    }
}
