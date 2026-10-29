use crate::cached::Versioned;
use crate::debug_targets::{COUNT_FORMAT, IDS_FORMAT, color_targets as debug_color_targets};
pub(crate) use crate::graph::Targets;
use crate::post::{GLOW_FORMAT, SCENE_FORMAT, SURFACE_FORMAT};
use crate::renderer::MOTION_FORMAT;

impl Targets {
    pub fn formats(self, canvas: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
        match self {
            Self::Canvas => vec![canvas],
            Self::Post => vec![SCENE_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
            Self::Temporal => vec![SCENE_FORMAT, MOTION_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
            Self::Debug => vec![IDS_FORMAT, COUNT_FORMAT],
        }
    }

    pub fn color_targets(self, canvas: wgpu::TextureFormat) -> Vec<Option<wgpu::ColorTargetState>> {
        if self == Self::Debug {
            return debug_color_targets();
        }
        self.formats(canvas)
            .into_iter()
            .map(|format| Some(format.into()))
            .collect()
    }
}

pub const RENDER_TARGET: wgpu::TextureUsages =
    wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);

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

pub fn fullscreen_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    (view, drawn): (&wgpu::TextureView, (u32, u32)),
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    draw_within(&mut pass, drawn);
    pass
}

pub struct ScreenTarget<T> {
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
    pub fn resizes(&mut self, size: FrameSize) -> bool {
        if self.size() == Some(size) {
            return false;
        }
        self.release();
        self.mark = crate::memory::created_count();
        true
    }

    pub fn set(&mut self, size: FrameSize, targets: T) {
        self.created = crate::memory::created_since(self.mark);
        self.made = Some((size, Versioned::new(targets)));
    }

    pub fn get(&self) -> Option<&Versioned<T>> {
        self.made.as_ref().map(|(_, targets)| targets)
    }

    pub fn size(&self) -> Option<FrameSize> {
        self.made.as_ref().map(|(size, _)| *size)
    }

    pub fn release(&mut self) {
        crate::memory::release_ids(&std::mem::take(&mut self.created));
        self.made = None;
    }
}
