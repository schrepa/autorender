//! The canvas the renderer presents to: its WebGPU surface, configured at
//! the canvas's size within the device's limits, suspended at size zero and
//! reconfigured at the render boundary after a resize.

use super::*;

pub(super) struct Display {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    canvas: HtmlCanvasElement,
    configuration: wgpu::SurfaceConfiguration,
    pub(super) max_dimension: u32,
    pub(super) suspended: bool,
    reconfigure_next_frame: bool,
}

impl Display {
    /// The display of `surface` on `canvas`, sized as the canvas within
    /// `max_dimension` (suspended when it has no area), not yet configured.
    pub(super) fn new(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        canvas: HtmlCanvasElement,
        adapter: &wgpu::Adapter,
        max_dimension: u32,
    ) -> Result<Self, JsValue> {
        let extent = clamped_extent(canvas.width(), canvas.height(), max_dimension);
        let (width, height) = extent.unwrap_or((1, 1));
        let mut configuration = surface
            .get_default_config(adapter, width, height)
            .ok_or_else(|| {
                JsValue::from_str("This adapter cannot present to the supplied canvas.")
            })?;
        configuration.present_mode = wgpu::PresentMode::Fifo;
        configuration.desired_maximum_frame_latency = 2;
        Ok(Self {
            instance,
            surface,
            canvas,
            configuration,
            max_dimension,
            suspended: extent.is_none(),
            reconfigure_next_frame: false,
        })
    }

    /// Sizes the canvas and configures the surface, unless suspended.
    pub(super) fn configure(&self, device: &wgpu::Device) {
        if !self.suspended {
            self.canvas.set_width(self.configuration.width);
            self.canvas.set_height(self.configuration.height);
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
        self.canvas.set_width(width);
        self.canvas.set_height(height);
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
    ) -> Result<Option<wgpu::SurfaceTexture>, JsValue> {
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
                    .instance
                    .create_surface(wgpu::SurfaceTarget::Canvas(self.canvas.clone()))
                    .map_err(|error| js_error("Could not recover the WebGPU surface", error))?;
                self.reconfigure_next_frame = true;
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(JsValue::from_str(
                    "WebGPU surface validation failed. Reload to recreate the engine.",
                ));
            }
        };
        Ok(Some(frame))
    }
}

/// The depth buffer, and a generation that advances when it is replaced
/// (passes that sample it rebind).
pub(super) struct DepthTarget {
    // Keep texture ownership explicit alongside its reusable view.
    _texture: wgpu::Texture,
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
            // Ambient occlusion (decision 021) samples it after the opaque pass.
            crate::targets::RENDER_TARGET,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
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
