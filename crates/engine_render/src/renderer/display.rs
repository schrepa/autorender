use std::cell::Cell;

use super::*;
use crate::gpu_data::clamped_extent;
use crate::readback::{Arrival, Keep, Readbacks, Reader};
use crate::targets::FrameSize;
use engine_core::{Expected, FrameId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SurfaceState {
    Ready,
    Stale,
    Suspended,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SurfaceChange {
    Hidden,
    Sized,
    Configured,
    Outdated,
}

impl SurfaceState {
    fn after(self, change: SurfaceChange) -> Self {
        match (self, change) {
            (_, SurfaceChange::Hidden) => Self::Suspended,
            (_, SurfaceChange::Sized) | (Self::Ready | Self::Stale, SurfaceChange::Outdated) => {
                Self::Stale
            }
            (Self::Stale, SurfaceChange::Configured) => Self::Ready,
            (state @ (Self::Ready | Self::Suspended), SurfaceChange::Configured)
            | (state @ Self::Suspended, SurfaceChange::Outdated) => state,
        }
    }
}

pub struct Display {
    instance: wgpu::Instance,
    surface: Option<wgpu::Surface<'static>>,
    offscreen: Option<wgpu::Texture>,
    target: RenderTarget,
    configuration: wgpu::SurfaceConfiguration,
    pub(crate) max_dimension: u32,
    state: SurfaceState,
    timer: Option<FrameTimer>,
}

impl Display {
    pub(crate) fn new(
        instance: wgpu::Instance,
        surface: Option<wgpu::Surface<'static>>,
        target: RenderTarget,
        adapter: &wgpu::Adapter,
        max_dimension: u32,
    ) -> Result<Self, Error> {
        let (width, height) = target.size();
        let extent = clamped_extent(width, height, max_dimension);
        let (width, height) = extent.unwrap_or((1, 1));
        let mut configuration = match &surface {
            Some(surface) => surface
                .get_default_config(adapter, width, height)
                .ok_or_else(|| failed("$.surface").expected(Expected::Supported))?,
            None => wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
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
        if let Some(surface) = &surface {
            configuration.usage |= surface.get_capabilities(adapter).usages
                & (wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST);
        }
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
            state: if extent.is_some() {
                SurfaceState::Stale
            } else {
                SurfaceState::Suspended
            },
            timer: None,
        })
    }

    pub(crate) fn configure(&mut self, device: &wgpu::Device, period: f32) {
        self.timer = FrameTimer::new(device, period);
        if !self.suspended() {
            self.target
                .set_size((self.configuration.width, self.configuration.height));
            self.configure_surface(device);
        }
    }

    pub(crate) fn suspended(&self) -> bool {
        self.state == SurfaceState::Suspended
    }

    pub(crate) fn instance(&self) -> wgpu::Instance {
        self.instance.clone()
    }

    pub(crate) fn power_preference(&self) -> wgpu::PowerPreference {
        self.target.power_preference()
    }

    pub(crate) fn supports(&self, adapter: &wgpu::Adapter) -> bool {
        self.surface
            .as_ref()
            .is_none_or(|surface| adapter.is_surface_supported(surface))
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
        self.state = self.state.after(SurfaceChange::Configured);
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

    pub(crate) fn take_frame_time(&self) -> Option<(f64, (u32, u32))> {
        self.timer.as_ref()?.read.take()
    }

    pub(crate) fn time_begin(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(timer) = &self.timer {
            timer.begin(encoder);
        }
    }

    pub(crate) fn time_end(
        &mut self,
        recording: (&wgpu::Device, &mut wgpu::CommandEncoder),
        (readbacks, written, slots): (&mut Readbacks, FrameId, usize),
    ) {
        let size = self.size().pixels();
        if let Some(timer) = &mut self.timer {
            timer.end(recording, (readbacks, written, slots), size);
        }
    }

    pub(crate) fn timed(&mut self, written: FrameId, arrival: Arrival<'_>) {
        if let Some(timer) = &mut self.timer {
            timer.arrived(written, arrival);
        }
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        let Some((width, height)) = clamped_extent(width, height, self.max_dimension) else {
            self.state = self.state.after(SurfaceChange::Hidden);
            return;
        };
        if !self.suspended()
            && self.configuration.width == width
            && self.configuration.height == height
        {
            return;
        }
        self.configuration.width = width;
        self.configuration.height = height;
        self.target.set_size((width, height));
        self.state = self.state.after(SurfaceChange::Sized);
    }

    pub(crate) fn acquire(&mut self, device: &wgpu::Device) -> Result<Option<DisplayFrame>, Error> {
        let _profile = engine_profile::span("render", "surface_acquisition");
        if self.state == SurfaceState::Stale {
            self.configure_surface(device);
        }
        let Some(surface) = &self.surface else {
            return Ok(self.offscreen.clone().map(|texture| DisplayFrame {
                shown: Shown::Offscreen(texture),
            }));
        };
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                self.state = self.state.after(SurfaceChange::Outdated);
                frame
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.state = self.state.after(SurfaceChange::Outdated);
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                self.surface = self
                    .target
                    .surface(&self.instance)
                    .map_err(|error| failed_with("$.surface", error))?;
                self.state = self.state.after(SurfaceChange::Outdated);
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err(failed("$.surface").expected(Expected::Live));
            }
        };
        Ok(Some(DisplayFrame {
            shown: Shown::Surface(frame),
        }))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn offscreen(&self) -> Option<&wgpu::Texture> {
        self.offscreen.as_ref()
    }
}

pub(crate) struct DisplayFrame {
    shown: Shown,
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
        if let Shown::Surface(frame) = self.shown {
            queue.present(frame);
        }
    }
}

pub(crate) struct FrameTimer {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    period: f64,
    sizes: Vec<(FrameId, (u32, u32))>,
    read: Cell<Option<ReadFrame>>,
}

type ReadFrame = (f64, (u32, u32));

const FRAME_STAMP_BYTES: u64 = 16;

impl FrameTimer {
    fn new(device: &wgpu::Device, period: f32) -> Option<Self> {
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
            period: f64::from(period),
            sizes: Vec::new(),
            read: Cell::new(None),
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

    fn end(
        &mut self,
        (device, encoder): (&wgpu::Device, &mut wgpu::CommandEncoder),
        (readbacks, written, slots): (&mut Readbacks, FrameId, usize),
        size: (u32, u32),
    ) {
        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/frame-timer"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.queries,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(1),
            }),
        });
        encoder.resolve_query_set(&self.queries, 0..2, &self.resolved, 0);
        let recorded = readbacks.record(
            (device, encoder),
            (&self.resolved, 0..FRAME_STAMP_BYTES),
            (Reader::FrameTimer, written, None),
            Keep::Ring(slots),
        );
        if recorded {
            self.sizes.push((written, size));
        }
    }

    fn arrived(&mut self, written: FrameId, arrival: Arrival<'_>) {
        let size = self
            .sizes
            .iter()
            .position(|(known, _)| *known == written)
            .map(|at| self.sizes.remove(at).1);
        self.sizes.retain(|(known, _)| *known >= written);
        let (Some(size), Arrival::Read(bytes)) = (size, arrival) else {
            return;
        };
        let stamp = |at: usize| {
            bytes
                .get(at..at + 8)
                .and_then(|word| word.try_into().ok())
                .map_or(0, u64::from_le_bytes)
        };
        let (begin, end) = (stamp(0), stamp(8));
        if end > begin {
            self.read
                .set(Some(((end - begin) as f64 * self.period / 1.0e6, size)));
        }
    }
}
