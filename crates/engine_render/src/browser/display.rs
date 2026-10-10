//! What the renderer presents to, a canvas or a window: its surface,
//! configured at the target's size within the device's limits, suspended at
//! size zero and reconfigured at the render boundary after a resize.

use super::*;

pub(super) struct Display {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
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
        surface: wgpu::Surface<'static>,
        target: RenderTarget,
        adapter: &wgpu::Adapter,
        (max_dimension, max_layers): (u32, u32),
    ) -> Result<Self, RenderError> {
        let (width, height) = target.size();
        let extent = clamped_extent(width, height, max_dimension);
        let (width, height) = extent.unwrap_or((1, 1));
        let mut configuration = surface
            .get_default_config(adapter, width, height)
            .ok_or_else(|| {
                RenderError("This adapter cannot present to the supplied canvas.".into())
            })?;
        configuration.present_mode = wgpu::PresentMode::Fifo;
        // Shaders encode their output for a UNORM target, as canvases are.
        configuration.format = configuration.format.remove_srgb_suffix();
        configuration.desired_maximum_frame_latency = 2;
        Ok(Self {
            instance,
            surface,
            target,
            configuration,
            max_dimension,
            max_layers,
            suspended: extent.is_none(),
            reconfigure_next_frame: false,
        })
    }

    /// Sizes the target and configures the surface, unless suspended.
    pub(super) fn configure(&mut self, device: &wgpu::Device) {
        if !self.suspended {
            self.target
                .set_size((self.configuration.width, self.configuration.height));
            self.surface.configure(device, &self.configuration);
        }
    }

    /// The surface's size in physical pixels.
    pub(super) fn size(&self) -> (u32, u32) {
        (self.configuration.width, self.configuration.height)
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

    /// The canvas texture to draw, reconfiguring the surface (and replacing
    /// `depth`) first when resized; `None` skips the frame (a lost, outdated
    /// or busy surface).
    pub(super) fn acquire(
        &mut self,
        device: &wgpu::Device,
        depth: &mut DepthTarget,
    ) -> Result<Option<wgpu::SurfaceTexture>, RenderError> {
        let _profile = engine_profile::span("render", "surface_acquisition");
        if self.reconfigure_next_frame {
            self.surface.configure(device, &self.configuration);
            depth.resize(device, self.size());
            self.reconfigure_next_frame = false;
        }
        let frame = match self.surface.get_current_texture() {
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
        Ok(Some(frame))
    }
}

/// The depth buffer, and a generation that advances when it is replaced
/// (passes that sample it rebind).
pub(super) struct DepthTarget {
    // Water copies it before drawing over it.
    pub(super) texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
    pub(super) generation: u64,
}

impl DepthTarget {
    pub(super) fn new(device: &wgpu::Device, (width, height): (u32, u32)) -> Self {
        let texture = crate::targets::texture_2d(
            device,
            "engine/depth",
            (width, height),
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
            view,
            generation: 0,
        }
    }

    /// Replaces the buffer with one of `size`, advancing the generation.
    fn resize(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        let generation = self.generation + 1;
        *self = Self::new(device, size);
        self.generation = generation;
    }
}
