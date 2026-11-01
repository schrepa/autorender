pub fn texture_2d(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    layers: u32,
    levels: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    crate::memory::texture(
        device,
        &wgpu::TextureDescriptor {
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
        },
    )
}

pub fn texel(
    device: &wgpu::Device,
    queue: &crate::gpu_calls::Queue,
    label: &str,
    format: wgpu::TextureFormat,
    texel: [u8; 4],
) -> wgpu::Texture {
    let texture = texture_2d(
        device,
        label,
        (1, 1),
        1,
        1,
        format,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    queue.write_texture(
        texture.as_image_copy(),
        &texel,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameRegion {
    pub display: FrameSize,
    pub render: FrameSize,
}

impl FrameRegion {
    pub fn new(display: FrameSize, render: FrameSize) -> Self {
        let within = |size: u32, whole: u32| size.clamp(1, whole.max(1));
        Self {
            display,
            render: FrameSize {
                width: within(render.width, display.width),
                height: within(render.height, display.height),
            },
        }
    }

    pub fn whole(display: FrameSize) -> Self {
        Self::new(display, display)
    }

    pub fn upscales(self) -> bool {
        self.render != self.display
    }

    pub fn share(self) -> [f32; 2] {
        [
            self.render.width as f32 / self.display.width.max(1) as f32,
            self.render.height as f32 / self.display.height.max(1) as f32,
        ]
    }

    pub fn mip_bias(self) -> f32 {
        self.share()[0].log2()
    }

    pub fn uniform(self) -> [f32; 4] {
        let [x, y] = self.share();
        [self.render.width as f32, self.render.height as f32, x, y]
    }

    pub fn level_region(self, level: u32) -> (u32, u32) {
        let FrameRegion { display, render } = self;
        let part = |drawn: u32, whole: u32| {
            let whole = (whole >> (level + 1)).max(1);
            ((drawn >> (level + 1)).max(1) + 1).min(whole)
        };
        (
            part(render.width, display.width),
            part(render.height, display.height),
        )
    }
}

pub fn draw_within(pass: &mut wgpu::RenderPass<'_>, (width, height): (u32, u32)) {
    pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    pass.set_scissor_rect(0, 0, width, height);
}
