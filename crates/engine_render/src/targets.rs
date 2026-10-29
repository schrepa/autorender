use crate::cached::Versioned;
use crate::debug_view::{COUNT_FORMAT as DEBUG_COUNT_FORMAT, IDS_FORMAT as DEBUG_IDS_FORMAT};
use crate::post::{GLOW_FORMAT, SCENE_FORMAT, SURFACE_FORMAT};
pub(crate) use crate::render_graph::Targets;
use crate::temporal::MOTION_FORMAT;

impl Targets {
    pub(crate) fn formats(self, canvas: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
        match self {
            Self::Canvas => vec![canvas],
            Self::Post => vec![SCENE_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
            Self::Temporal => vec![SCENE_FORMAT, MOTION_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
            Self::Debug => vec![DEBUG_IDS_FORMAT, DEBUG_COUNT_FORMAT],
        }
    }

    pub(crate) fn color_targets(
        self,
        canvas: wgpu::TextureFormat,
    ) -> Vec<Option<wgpu::ColorTargetState>> {
        if self == Self::Debug {
            return crate::debug_view::color_targets();
        }
        self.formats(canvas)
            .into_iter()
            .map(|format| Some(format.into()))
            .collect()
    }
}

pub(crate) const RENDER_TARGET: wgpu::TextureUsages =
    wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);

pub(crate) fn texture_2d(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    layers: u32,
    levels: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    crate::memory::texture(device, &wgpu::TextureDescriptor {
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

pub(crate) fn texel(
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
pub(crate) struct FrameSize {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl FrameSize {
    pub(crate) fn pixels(self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameRegion {
    pub(crate) display: FrameSize,
    pub(crate) render: FrameSize,
}

impl FrameRegion {
    pub(crate) fn new(display: FrameSize, render: FrameSize) -> Self {
        let within = |size: u32, whole: u32| size.clamp(1, whole.max(1));
        Self {
            display,
            render: FrameSize {
                width: within(render.width, display.width),
                height: within(render.height, display.height),
            },
        }
    }

    pub(crate) fn whole(display: FrameSize) -> Self {
        Self::new(display, display)
    }

    pub(crate) fn upscales(self) -> bool {
        self.render != self.display
    }

    pub(crate) fn share(self) -> [f32; 2] {
        [
            self.render.width as f32 / self.display.width.max(1) as f32,
            self.render.height as f32 / self.display.height.max(1) as f32,
        ]
    }

    pub(crate) fn mip_bias(self) -> f32 {
        self.share()[0].log2()
    }

    pub(crate) fn uniform(self) -> [f32; 4] {
        let [x, y] = self.share();
        [self.render.width as f32, self.render.height as f32, x, y]
    }
}

pub(crate) fn draw_within(pass: &mut wgpu::RenderPass<'_>, (width, height): (u32, u32)) {
    pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    pass.set_scissor_rect(0, 0, width, height);
}

pub(crate) struct ScreenTarget<T> {
    made: Option<(FrameSize, Versioned<T>)>,
    mark: u64,
    created: Vec<u64>,
}

impl<T> Default for ScreenTarget<T> {
    fn default() -> Self {
        Self {
            made: None,
            mark: 0,
            created: Vec::new(),
        }
    }
}

impl<T> ScreenTarget<T> {
    pub(crate) fn resizes(&mut self, size: FrameSize) -> bool {
        if self.size() == Some(size) {
            return false;
        }
        self.release();
        self.mark = crate::memory::created_count();
        true
    }

    pub(crate) fn set(&mut self, size: FrameSize, targets: T) {
        self.created = crate::memory::created_since(self.mark);
        self.made = Some((size, Versioned::new(targets)));
    }

    pub(crate) fn get(&self) -> Option<&Versioned<T>> {
        self.made.as_ref().map(|(_, targets)| targets)
    }

    pub(crate) fn size(&self) -> Option<FrameSize> {
        self.made.as_ref().map(|(size, _)| *size)
    }

    pub(crate) fn release(&mut self) {
        crate::memory::release_ids(&std::mem::take(&mut self.created));
        self.made = None;
    }
}
