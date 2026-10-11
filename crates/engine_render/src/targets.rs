//! The color targets of the frame's main passes (the render graph's
//! [`Targets`]): the formats every pipeline drawing into them declares, in
//! pipeline order; the 2D textures passes draw into and sample; and the
//! frame's size, which screen-sized targets are made at.

use crate::cached::Versioned;
use crate::post::{GLOW_FORMAT, SURFACE_FORMAT};
use crate::radiance::RADIANCE_FORMAT;
pub(crate) use crate::render_graph::Targets;
use crate::temporal::MOTION_FORMAT;

impl Targets {
    /// Main-pass formats: radiance, then the optional motion, glow and
    /// surface targets. The later Grade pass chooses the canvas format.
    pub(crate) fn formats(self, _canvas: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
        match self {
            Self::Canvas => vec![RADIANCE_FORMAT],
            Self::Post => vec![RADIANCE_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
            Self::Temporal => vec![RADIANCE_FORMAT, MOTION_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
        }
    }

    /// Opaque, unblended color target states of [`Self::formats`].
    pub(crate) fn color_targets(
        self,
        canvas: wgpu::TextureFormat,
    ) -> Vec<Option<wgpu::ColorTargetState>> {
        self.formats(canvas)
            .into_iter()
            .map(|format| Some(format.into()))
            .collect()
    }
}

/// Usage of targets that later passes sample.
pub(crate) const RENDER_TARGET: wgpu::TextureUsages =
    wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);

/// A single-sample 2D texture of `size` texels, `layers` array layers deep,
/// with `levels` mip levels.
pub(crate) fn texture_2d(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    layers: u32,
    levels: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
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

/// A 1x1 sampled texture of `format` holding `texel`, standing in for a
/// map or target that is absent.
pub(crate) fn texel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
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

/// The frame's size in physical pixels: the display's, which every
/// screen-sized target is made at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameSize {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl FrameSize {
    /// Width and height, as textures are sized.
    pub(crate) fn pixels(self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Targets made at the frame's size: made when a frame first wants them
/// and again, once, at each new size, each set a new generation, so the
/// groups binding them rebuild. Those of the old size are dropped before
/// the new ones are made (their textures destroyed with them).
pub(crate) struct ScreenTarget<T> {
    made: Option<(FrameSize, Versioned<T>)>,
}

impl<T> Default for ScreenTarget<T> {
    fn default() -> Self {
        Self { made: None }
    }
}

impl<T> ScreenTarget<T> {
    /// Whether targets must be made at `size`: none are held, or those
    /// held, which this drops, were made at another size.
    pub(crate) fn resizes(&mut self, size: FrameSize) -> bool {
        if self.size() == Some(size) {
            return false;
        }
        self.made = None;
        true
    }

    /// Keeps `targets`, made at `size`, as a new generation.
    pub(crate) fn set(&mut self, size: FrameSize, targets: T) {
        self.made = Some((size, Versioned::new(targets)));
    }

    /// The targets, once made.
    pub(crate) fn get(&self) -> Option<&Versioned<T>> {
        self.made.as_ref().map(|(_, targets)| targets)
    }

    /// The size the targets were made at.
    pub(crate) fn size(&self) -> Option<FrameSize> {
        self.made.as_ref().map(|(size, _)| *size)
    }

    /// Drops the targets.
    pub(crate) fn release(&mut self) {
        self.made = None;
    }
}
