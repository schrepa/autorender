//! The color targets of the frame's main passes (the render graph's
//! [`Targets`]): the formats every pipeline drawing into them declares, in
//! pipeline order; and the 2D textures passes draw into and sample.

use crate::post::{GLOW_FORMAT, SCENE_FORMAT, SURFACE_FORMAT};
pub(crate) use crate::render_graph::Targets;
use crate::temporal::MOTION_FORMAT;

impl Targets {
    /// The color target formats in pipeline order: the canvas's `canvas`
    /// format; the screen effects' scene, glow and surface targets;
    /// temporal frames' motion target after the scene.
    pub(crate) fn formats(self, canvas: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
        match self {
            Self::Canvas => vec![canvas],
            Self::Post => vec![SCENE_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
            Self::Temporal => vec![SCENE_FORMAT, MOTION_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
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
