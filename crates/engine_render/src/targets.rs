//! The color targets of the frame's main passes (the render graph's
//! [`Targets`]): the formats every pipeline drawing into them declares, in
//! pipeline order; the 2D textures passes draw into and sample; and the
//! frame's sizes: the display's, which screen-sized targets are made at,
//! and the render size frames draw at within them.

use crate::cached::Versioned;
use crate::debug_view::{COUNT_FORMAT as DEBUG_COUNT_FORMAT, IDS_FORMAT as DEBUG_IDS_FORMAT};
use crate::post::{GLOW_FORMAT, SCENE_FORMAT, SURFACE_FORMAT};
pub(crate) use crate::render_graph::Targets;
use crate::temporal::MOTION_FORMAT;

impl Targets {
    /// The color target formats in pipeline order: the canvas's `canvas`
    /// format; the screen effects' scene, glow and surface targets;
    /// temporal frames' motion target after the scene; the debug views'
    /// targets.
    pub(crate) fn formats(self, canvas: wgpu::TextureFormat) -> Vec<wgpu::TextureFormat> {
        match self {
            Self::Canvas => vec![canvas],
            Self::Post => vec![SCENE_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
            Self::Temporal => vec![SCENE_FORMAT, MOTION_FORMAT, GLOW_FORMAT, SURFACE_FORMAT],
            Self::Debug => vec![DEBUG_IDS_FORMAT, DEBUG_COUNT_FORMAT],
        }
    }

    /// Color target states of [`Self::formats`]: opaque and unblended, but
    /// for the debug views' fragments, which add up.
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

/// A size in physical pixels: the display's, which every screen-sized
/// target is made at, or the frame's render size within it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

/// A frame's two sizes: the display's, which the surface and every
/// screen-sized target are made at, and the render size the frame draws at
/// from their top-left corner, which its last pass upscales to the display.
/// Passes drawing screen-sized targets draw that region alone (viewport and
/// scissor), and those reading them read within it; a render size changing
/// leaves every target as it is. The default, of no pixels, stands in until
/// a frame plans its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameRegion {
    pub(crate) display: FrameSize,
    pub(crate) render: FrameSize,
}

impl FrameRegion {
    /// `render` within `display`, a pixel at least each way.
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

    /// The display's whole size.
    pub(crate) fn whole(display: FrameSize) -> Self {
        Self::new(display, display)
    }

    /// Whether the frame draws smaller than the display, which its last
    /// pass upscales.
    pub(crate) fn upscales(self) -> bool {
        self.render != self.display
    }

    /// The share of each axis of the display the frame draws.
    pub(crate) fn share(self) -> [f32; 2] {
        [
            self.render.width as f32 / self.display.width.max(1) as f32,
            self.render.height as f32 / self.display.height.max(1) as f32,
        ]
    }

    /// The mip bias material textures sample with: log2 of the render
    /// width's share of the display's, 0 at full size and negative below it,
    /// so frames drawing smaller keep their textures as sharp as the
    /// upscaled display shows them.
    pub(crate) fn mip_bias(self) -> f32 {
        self.share()[0].log2()
    }

    /// The region as shaders read it (`Camera.region` of `scene.wgsl`):
    /// the render size in pixels, and its share of the display on each axis.
    pub(crate) fn uniform(self) -> [f32; 4] {
        let [x, y] = self.share();
        [self.render.width as f32, self.render.height as f32, x, y]
    }
}

/// Sets `pass`'s viewport and scissor to its attachments' `width` ×
/// `height` texels from their top-left corner: the part of them a frame
/// drawing at its render size draws.
pub(crate) fn draw_within(pass: &mut wgpu::RenderPass<'_>, (width, height): (u32, u32)) {
    pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
    pass.set_scissor_rect(0, 0, width, height);
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
