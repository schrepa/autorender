use std::cell::Cell;
use std::rc::Rc;

use super::*;
use crate::buffers::ReadbackRing;
use crate::targets::FrameRegion;

pub struct Display {
    instance: wgpu::Instance,
    surface: Option<wgpu::Surface<'static>>,
    offscreen: Option<wgpu::Texture>,
    target: RenderTarget,
    configuration: wgpu::SurfaceConfiguration,
    pub(crate) max_dimension: u32,
    pub max_layers: u32,
    pub(crate) suspended: bool,
    reconfigure_next_frame: bool,
    asked: Option<(u32, u32)>,
    pub region: FrameRegion,
    timer: Option<Rc<FrameTimer>>,
    #[cfg(feature = "dev")]
    pub(crate) probe: super::probe::Probe,
}

impl Display {
    pub(crate) fn new(
        instance: wgpu::Instance,
        surface: Option<wgpu::Surface<'static>>,
        target: RenderTarget,
        adapter: &wgpu::Adapter,
        (max_dimension, max_layers): (u32, u32),
    ) -> Result<Self, Error> {
        let (width, height) = target.size();
        let extent = clamped_extent(width, height, max_dimension);
        let (width, height) = extent.unwrap_or((1, 1));
        let mut configuration = match &surface {
            Some(surface) => surface
                .get_default_config(adapter, width, height)
                .ok_or_else(|| failed("$.surface").expected(Expected::Supported))?,
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

    pub(crate) fn configure(&mut self, device: &wgpu::Device) {
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
                if let Some(offscreen) = &self.offscreen {
                    crate::memory::release_any(offscreen);
                }
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

    pub fn size(&self) -> FrameSize {
        FrameSize {
            width: self.configuration.width,
            height: self.configuration.height,
        }
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.configuration.format
    }

    pub(crate) fn ask_render_size(&mut self, width: u32, height: u32) {
        self.asked = Some((width, height));
    }

    pub(crate) fn plan_region(&mut self, drawn: bool) -> FrameRegion {
        let display = self.size();
        #[cfg(feature = "dev")]
        let drawn = drawn && self.probe.forces().is_none();
        self.region = match self.asked.filter(|_| drawn) {
            Some((width, height)) => FrameRegion::new(display, FrameSize { width, height }),
            None => FrameRegion::whole(display),
        };
        self.region
    }

    pub(crate) fn times_frames(&self) -> bool {
        self.timer.is_some()
    }

    pub(crate) fn take_frame_time(&self) -> Option<(f64, (u32, u32))> {
        self.timer.as_ref()?.read.take()
    }

    pub(crate) fn time_begin(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(timer) = &self.timer {
            timer.begin(encoder);
        }
    }

    pub(crate) fn time_end(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if let Some(timer) = &self.timer {
            timer.end(device, encoder, self.region.render.pixels());
        }
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
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
    }

    pub(crate) fn acquire(
        &mut self,
        device: &wgpu::Device,
        depth: &mut DepthTarget,
    ) -> Result<Option<DisplayFrame>, Error> {
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
                    .map_err(|error| failed_with("$.surface", error))?;
                self.reconfigure_next_frame = true;
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(failed("$.surface").expected(Expected::Live));
            }
        };
        Ok(Some(DisplayFrame {
            shown: Shown::Surface(frame),
            timer,
        }))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn offscreen(&self) -> Option<&wgpu::Texture> {
        self.offscreen.as_ref()
    }
}

pub(crate) struct DisplayFrame {
    shown: Shown,
    timer: Option<Rc<FrameTimer>>,
}

enum Shown {
    Surface(wgpu::SurfaceTexture),
    Offscreen(wgpu::Texture),
}

impl DisplayFrame {
    pub(crate) fn texture(&self) -> &wgpu::Texture {
        match &self.shown {
            Shown::Surface(frame) => &frame.texture,
            Shown::Offscreen(texture) => texture,
        }
    }

    pub(crate) fn present(self, queue: &crate::gpu_calls::Queue) {
        if let Some(timer) = &self.timer {
            timer.after_submit(queue.get_timestamp_period());
        }
        if let Shown::Surface(frame) = self.shown {
            queue.present(frame);
        }
    }
}

pub(crate) struct FrameTimer {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readbacks: ReadbackRing<(u32, u32)>,
    read: Rc<Cell<Option<ReadFrame>>>,
}

type ReadFrame = (f64, (u32, u32));

const FRAME_STAMP_BYTES: u64 = 16;

impl FrameTimer {
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
            resolved: crate::memory::buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("engine/frame-timer-resolve"),
                    size: FRAME_STAMP_BYTES,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            ),
            readbacks: ReadbackRing::new("engine/frame-timer-readback"),
            read: Rc::default(),
        })
    }

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

    fn after_submit(&self, period: f32) {
        let read = Rc::clone(&self.read);
        let period = f64::from(period);
        self.readbacks.after_submit(move |words, size| {
            let stamp =
                |word: usize| u64::from(words.get(word)) | (u64::from(words.get(word + 1)) << 32);
            let (begin, end) = (stamp(0), stamp(2));
            if end > begin {
                read.set(Some(((end - begin) as f64 * period / 1.0e6, size)));
            }
        });
    }
}

pub struct DepthTarget {
    pub texture: wgpu::Texture,
    pub view: Versioned<wgpu::TextureView>,
}

impl DepthTarget {
    pub(crate) fn new(device: &wgpu::Device, size: FrameSize) -> Self {
        let texture = crate::targets::texture_2d(
            device,
            "engine/depth",
            size.pixels(),
            1,
            1,
            DEPTH_FORMAT,
            crate::targets::RENDER_TARGET | wgpu::TextureUsages::COPY_SRC,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view: Versioned::new(view),
        }
    }

    fn resize(&mut self, device: &wgpu::Device, size: FrameSize) {
        crate::memory::release_any(&self.texture);
        *self = Self::new(device, size);
    }
}
