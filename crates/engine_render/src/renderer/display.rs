//! What the renderer presents to, a canvas or a window: its surface,
//! configured at the target's whole size within the device's limits,
//! suspended at size zero and reconfigured at the render boundary after a
//! resize; the size frames draw at within it; and the GPU time of the frames
//! it shows. An offscreen target has a texture of that size in place of a
//! surface.

use std::cell::Cell;
use std::rc::Rc;

use super::*;
use crate::buffers::ReadbackRing;
use crate::targets::FrameRegion;

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
    // The render size the host asks display frames to draw at, and the
    // frame's sizes.
    asked: Option<(u32, u32)>,
    pub(super) region: FrameRegion,
    // The GPU time of the frames shown, where the device times work.
    timer: Option<Rc<FrameTimer>>,
    // The pixel probe of development sessions, which reads back what the
    // display showed.
    #[cfg(feature = "dev")]
    pub(super) probe: super::probe::Probe,
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
            // As windows and pages draw on most devices: BGRA, which no
            // target of the renderer's own takes, so pipeline lists cooked
            // offscreen tell the canvas's targets apart.
            None => wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                format: wgpu::TextureFormat::Bgra8Unorm,
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
            asked: None,
            region: FrameRegion::whole(FrameSize { width, height }),
            timer: None,
            #[cfg(feature = "dev")]
            probe: super::probe::Probe::default(),
        })
    }

    /// Sizes the target and configures the surface (or makes the offscreen
    /// texture), unless suspended; times frames where `device` times work.
    pub(super) fn configure(&mut self, device: &wgpu::Device) {
        self.timer = FrameTimer::new(device).map(Rc::new);
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

    /// See [`Renderer::set_render_size`].
    pub(super) fn ask_render_size(&mut self, width: u32, height: u32) {
        self.asked = Some((width, height));
    }

    /// Plans this frame's sizes: the render size asked for within the
    /// display where the frame is `drawn` smaller (a display frame through
    /// the screen effects' targets, which its last pass upscales from), the
    /// display's whole size otherwise and for the pixel probe's frames, whose
    /// copies read the pixel asked for.
    pub(super) fn plan_region(&mut self, drawn: bool) -> FrameRegion {
        let display = self.size();
        #[cfg(feature = "dev")]
        let drawn = drawn && self.probe.forces().is_none();
        self.region = match self.asked.filter(|_| drawn) {
            Some((width, height)) => FrameRegion::new(display, FrameSize { width, height }),
            None => FrameRegion::whole(display),
        };
        self.region
    }

    /// Whether the device times the GPU work of frames.
    pub(super) fn times_frames(&self) -> bool {
        self.timer.is_some()
    }

    /// The GPU time in milliseconds of the last frame read back since the
    /// last call, and the render size it drew at.
    pub(super) fn take_frame_time(&self) -> Option<(f64, (u32, u32))> {
        self.timer.as_ref()?.read.take()
    }

    /// Writes the frame's first timestamp into `encoder`, before its work.
    pub(super) fn time_begin(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(timer) = &self.timer {
            timer.begin(encoder);
        }
    }

    /// Writes the frame's last timestamp into `encoder`, after its work, and
    /// records both's read-back.
    pub(super) fn time_end(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if let Some(timer) = &self.timer {
            timer.end(device, encoder, self.region.render.pixels());
        }
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
        let timer = self.timer.clone();
        let Some(surface) = &self.surface else {
            return Ok(self.offscreen.clone().map(|texture| DisplayFrame {
                shown: Shown::Offscreen(texture),
                timer,
            }));
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
        Ok(Some(DisplayFrame {
            shown: Shown::Surface(frame),
            timer,
        }))
    }

    /// The offscreen texture frames draw into, for read-backs; none for a
    /// canvas or window.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn offscreen(&self) -> Option<&wgpu::Texture> {
        self.offscreen.as_ref()
    }
}

/// A frame's texture, and the display's frame timer, which reads back the
/// frame's GPU time once it is submitted.
pub(super) struct DisplayFrame {
    shown: Shown,
    timer: Option<Rc<FrameTimer>>,
}

/// A frame's texture: a surface's, presented after submission, or the
/// offscreen texture, which keeps it.
enum Shown {
    Surface(wgpu::SurfaceTexture),
    Offscreen(wgpu::Texture),
}

impl DisplayFrame {
    pub(super) fn texture(&self) -> &wgpu::Texture {
        match &self.shown {
            Shown::Surface(frame) => &frame.texture,
            Shown::Offscreen(texture) => texture,
        }
    }

    /// Shows the frame once submitted: a surface's presents, the offscreen
    /// texture keeps it; its GPU time's read-back maps.
    pub(super) fn present(self, queue: &wgpu::Queue) {
        if let Some(timer) = &self.timer {
            timer.after_submit(queue.get_timestamp_period());
        }
        if let Shown::Surface(frame) = self.shown {
            queue.present(frame);
        }
    }
}

/// The GPU time of the frames a display shows, where the device times work:
/// two timestamps a frame, from empty passes before and after its work,
/// resolved and read back a few frames late with the render size the frame
/// drew at.
pub(super) struct FrameTimer {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readbacks: ReadbackRing<(u32, u32)>,
    // The last frame read back, its milliseconds and render size, until
    // taken.
    read: Rc<Cell<Option<(f64, (u32, u32))>>>,
}

/// Bytes of the two timestamps a frame resolves.
const FRAME_STAMP_BYTES: u64 = 16;

impl FrameTimer {
    /// The timer of `device`; none where it has no timestamp queries.
    fn new(device: &wgpu::Device) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        Some(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("engine/frame-timer"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolved: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("engine/frame-timer-resolve"),
                size: FRAME_STAMP_BYTES,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readbacks: ReadbackRing::new("engine/frame-timer-readback"),
            read: Rc::default(),
        })
    }

    /// Writes the frame's first timestamp into `encoder`, from an empty
    /// pass.
    fn begin(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/frame-timer"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: None,
            }),
        });
    }

    /// Writes the frame's last timestamp into `encoder`, from an empty pass,
    /// resolves both and records their read-back, tagged with the render
    /// size `size` the frame drew at.
    fn end(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, size: (u32, u32)) {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/frame-timer"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(1),
            }),
        });
        encoder.resolve_query_set(&self.queries, 0..2, &self.resolved, 0);
        self.readbacks
            .record(device, encoder, &self.resolved, FRAME_STAMP_BYTES, size);
    }

    /// Requests the map of the read-back the frame just submitted recorded,
    /// its timestamps `period` nanoseconds a tick apart.
    fn after_submit(&self, period: f32) {
        let read = Rc::clone(&self.read);
        let period = f64::from(period);
        self.readbacks.after_submit(move |words, size| {
            let stamp = |word: usize| {
                u64::from(words.get(word)) | (u64::from(words.get(word + 1)) << 32)
            };
            let (begin, end) = (stamp(0), stamp(2));
            if end > begin {
                read.set(Some(((end - begin) as f64 * period / 1.0e6, size)));
            }
        });
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
