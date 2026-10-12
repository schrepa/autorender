//! What the renderer presents to, a canvas or a window: its surface,
//! configured at the target's size within the device's limits, suspended at
//! size zero and reconfigured at the render boundary after a resize. An
//! offscreen target has a texture of that size in place of a surface.

use super::*;

pub(super) struct Display {
    instance: wgpu::Instance,
    // None for an offscreen target, which draws into `offscreen`.
    surface: Option<wgpu::Surface<'static>>,
    offscreen: Option<wgpu::Texture>,
    target: RenderTarget,
    configuration: wgpu::SurfaceConfiguration,
    pub(super) max_dimension: u32,
    // Layers a texture array holds, read once.
    pub(super) max_layers: u32,
    pub(super) suspended: bool,
    reconfigure_next_frame: bool,
}

impl Display {
    /// The display of `surface` on `target`, sized as the target within
    /// `max_dimension` (suspended when it has no area), not yet configured.
    pub(super) fn new(
        instance: wgpu::Instance,
        surface: Option<wgpu::Surface<'static>>,
        target: RenderTarget,
        adapter: &wgpu::Adapter,
        (max_dimension, max_layers): (u32, u32),
    ) -> Result<Self, RenderError> {
        let (width, height) = target.size();
        let extent = clamped_extent(width, height, max_dimension);
        let (width, height) = extent.unwrap_or((1, 1));
        let mut configuration = match &surface {
            Some(surface) => surface
                .get_default_config(adapter, width, height)
                .ok_or_else(|| {
                    RenderError("This adapter cannot present to the supplied canvas.".into())
                })?,
            // Read back as it is stored: RGBA.
            None => wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                format: wgpu::TextureFormat::Rgba8Unorm,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: Vec::new(),
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        };
        configuration.present_mode = target.present_mode();
        // Shaders encode their output for a UNORM target, as canvases are.
        configuration.format = configuration.format.remove_srgb_suffix();
        configuration.desired_maximum_frame_latency = 2;
        Ok(Self {
            instance,
            surface,
            offscreen: None,
            target,
            configuration,
            max_dimension,
            max_layers,
            suspended: extent.is_none(),
            reconfigure_next_frame: false,
        })
    }

    /// Sizes the target and configures the surface (or makes the offscreen
    /// texture), unless suspended.
    pub(super) fn configure(&mut self, device: &wgpu::Device) {
        if !self.suspended {
            self.target
                .set_size((self.configuration.width, self.configuration.height));
            self.configure_surface(device);
        }
    }

    fn configure_surface(&mut self, device: &wgpu::Device) {
        match &self.surface {
            Some(surface) => surface.configure(device, &self.configuration),
            None => {
                self.offscreen = Some(crate::targets::texture_2d(
                    device,
                    "engine/offscreen",
                    self.size().pixels(),
                    1,
                    1,
                    self.configuration.format,
                    self.configuration.usage,
                ));
            }
        }
    }

    /// The surface's size in physical pixels: the frame's, once acquired.
    pub(super) fn size(&self) -> FrameSize {
        FrameSize {
            width: self.configuration.width,
            height: self.configuration.height,
        }
    }

    pub(super) fn format(&self) -> wgpu::TextureFormat {
        self.configuration.format
    }

    /// See [`Renderer::resize`].
    pub(super) fn resize(&mut self, width: u32, height: u32) {
        let Some((width, height)) = clamped_extent(width, height, self.max_dimension) else {
            self.suspended = true;
            return;
        };
        if !self.suspended
            && self.configuration.width == width
            && self.configuration.height == height
        {
            return;
        }
        self.suspended = false;
        self.configuration.width = width;
        self.configuration.height = height;
        self.target.set_size((width, height));
        self.reconfigure_next_frame = true;
        // Reconfiguration happens at the next render boundary after checking
        // device loss, so resize never touches a known-lost GPU device.
    }

    /// The texture to draw, reconfiguring the surface (and replacing
    /// `depth`) first when resized; `None` skips the frame (a lost, outdated
    /// or busy surface).
    pub(super) fn acquire(
        &mut self,
        device: &wgpu::Device,
        depth: &mut DepthTarget,
    ) -> Result<Option<DisplayFrame>, RenderError> {
        let _profile = engine_profile::span("render", "surface_acquisition");
        if self.reconfigure_next_frame {
            self.configure_surface(device);
            depth.resize(device, self.size());
            self.reconfigure_next_frame = false;
        }
        let Some(surface) = &self.surface else {
            return Ok(self.offscreen.clone().map(DisplayFrame::Offscreen));
        };
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.reconfigure_next_frame = true;
                frame
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.reconfigure_next_frame = true;
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .target
                    .surface(&self.instance)
                    .map_err(|error| render_error("Could not recover the GPU surface", error))?;
                self.reconfigure_next_frame = true;
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(RenderError(
                    "WebGPU surface validation failed. Reload to recreate the engine.".into(),
                ));
            }
        };
        Ok(Some(DisplayFrame::Surface(frame)))
    }

    /// The offscreen texture frames draw into, for read-backs; none for a
    /// canvas or window.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn offscreen(&self) -> Option<&wgpu::Texture> {
        self.offscreen.as_ref()
    }
}

/// A frame's texture: a surface's, presented after submission, or the
/// offscreen texture, which keeps it.
pub(super) enum DisplayFrame {
    Surface(wgpu::SurfaceTexture),
    Offscreen(wgpu::Texture),
}

impl DisplayFrame {
    pub(super) fn texture(&self) -> &wgpu::Texture {
        match self {
            Self::Surface(frame) => &frame.texture,
            Self::Offscreen(texture) => texture,
        }
    }

    /// Shows the frame: a surface's presents, the offscreen texture keeps it.
    pub(super) fn present(self, queue: &wgpu::Queue) {
        if let Self::Surface(frame) = self {
            queue.present(frame);
        }
    }
}

/// The depth buffer and its view, whose generation advances when it is
/// replaced (groups that sample it rebind).
pub(super) struct DepthTarget {
    // Water copies it before drawing over it.
    pub(super) texture: wgpu::Texture,
    pub(super) view: Versioned<wgpu::TextureView>,
}

impl DepthTarget {
    pub(super) fn new(device: &wgpu::Device, size: FrameSize) -> Self {
        let texture = crate::targets::texture_2d(
            device,
            "engine/depth",
            size.pixels(),
            1,
            1,
            DEPTH_FORMAT,
            // Ambient occlusion samples it after the opaque
            // pass, and water copies it.
            crate::targets::RENDER_TARGET | wgpu::TextureUsages::COPY_SRC,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view: Versioned::new(view),
        }
    }

    /// Replaces the buffer with one of `size`, its view of a new generation.
    fn resize(&mut self, device: &wgpu::Device, size: FrameSize) {
        *self = Self::new(device, size);
    }
}
