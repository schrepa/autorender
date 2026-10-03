//! Pipeline creation (decision 049). Every render and compute pipeline the
//! renderer builds goes through these functions, which count creations so
//! frames report when compiling ends: a frame that creates none compiles
//! nothing, and hosts reveal the canvas once frames stop creating them.

use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

thread_local! {
    static CREATED: Cell<u32> = const { Cell::new(0) };
}

fn count() {
    CREATED.with(|created| created.set(created.get().saturating_add(1)));
}

/// Creates a render pipeline, counting it.
pub(crate) fn render(
    device: &wgpu::Device,
    descriptor: &wgpu::RenderPipelineDescriptor<'_>,
) -> wgpu::RenderPipeline {
    count();
    device.create_render_pipeline(descriptor)
}

/// Creates a compute pipeline, counting it.
pub(crate) fn compute(
    device: &wgpu::Device,
    descriptor: &wgpu::ComputePipelineDescriptor<'_>,
) -> wgpu::ComputePipeline {
    count();
    device.create_compute_pipeline(descriptor)
}

/// The pipelines created since the last call.
fn take_created() -> u32 {
    CREATED.with(|created| created.replace(0))
}

/// The pipelines a renderer created, and whether their compiles may still
/// run: they precede the work submitted after them, so once the GPU
/// completes the last frame that created some, they are done.
#[derive(Default)]
pub(crate) struct PipelineCounts {
    created: u32,
    // Frames that created some, and the latest of those the GPU completed.
    frames: u64,
    compiled: Arc<AtomicU64>,
}

impl PipelineCounts {
    /// Counts the pipelines created since the last call, after the frame
    /// creating them was submitted to `queue`; returns how many.
    pub(crate) fn frame_submitted(&mut self, queue: &wgpu::Queue) -> u32 {
        let created = take_created();
        self.created = self.created.saturating_add(created);
        if created > 0 {
            self.frames += 1;
            let (compiled, frame) = (Arc::clone(&self.compiled), self.frames);
            queue.on_submitted_work_done(move || {
                compiled.fetch_max(frame, Ordering::Relaxed);
            });
        }
        created
    }

    /// Pipelines created so far.
    pub(crate) fn created(&self) -> u32 {
        self.created
    }

    /// Whether the GPU has not yet completed the last frame that created
    /// pipelines.
    pub(crate) fn compiling(&self) -> bool {
        self.compiled.load(Ordering::Relaxed) < self.frames
    }
}

/// What one binding of a shared layout holds (decision 049).
///
/// Each distinct pipeline layout costs the browser real work: Firefox's
/// Direct3D 12 backend builds a root signature and three indirect command
/// signatures for it, which an integrated Intel GPU's driver compiles in
/// about 150 ms. Passes of one family therefore share a layout listing every
/// binding any of them reads, and their bind groups fill the bindings a pass
/// leaves alone with [`Fillers`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Slot {
    Uniform,
    /// A storage buffer shaders only read.
    Read,
    /// A storage buffer shaders write.
    Write,
    /// A 2D float texture shaders load from.
    Texture,
    /// A 2D float texture shaders sample with filtering.
    Filtered,
    /// A 2D depth texture.
    Depth,
    /// A filtering sampler.
    Sampler,
}

impl Slot {
    fn binding_type(self) -> wgpu::BindingType {
        let buffer = |ty| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let texture = |sample_type| wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        match self {
            Self::Uniform => buffer(wgpu::BufferBindingType::Uniform),
            Self::Read => buffer(wgpu::BufferBindingType::Storage { read_only: true }),
            Self::Write => buffer(wgpu::BufferBindingType::Storage { read_only: false }),
            Self::Texture => texture(wgpu::TextureSampleType::Float { filterable: false }),
            Self::Filtered => texture(wgpu::TextureSampleType::Float { filterable: true }),
            Self::Depth => texture(wgpu::TextureSampleType::Depth),
            Self::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        }
    }
}

/// A bind group layout of `slots`, each visible to `visibility`.
pub(crate) fn group_layout(
    device: &wgpu::Device,
    label: &str,
    visibility: wgpu::ShaderStages,
    slots: &[(u32, Slot)],
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = slots
        .iter()
        .map(|&(binding, slot)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: slot.binding_type(),
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

/// Written storage fillers per group: each written binding a group fills
/// gets its own range, since written bindings of a dispatch or draw may not
/// overlap.
const WRITE_FILLERS: u64 = 32;
/// Bytes of each filler buffer range (the storage offset alignment).
const FILLER_BYTES: u64 = 256;

/// Resources standing in for the bindings of a shared layout a pass does not
/// read: the smallest buffers and textures of each [`Slot`]. Read and written
/// storage have separate buffers, since one buffer may not be both in a
/// dispatch or draw.
pub(crate) struct Fillers {
    uniform: wgpu::Buffer,
    read: wgpu::Buffer,
    write: wgpu::Buffer,
    texture: wgpu::TextureView,
    depth: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

impl Fillers {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let texture = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self {
            uniform: buffer(
                "engine/filler-uniform",
                FILLER_BYTES,
                wgpu::BufferUsages::UNIFORM,
            ),
            read: buffer(
                "engine/filler-read",
                FILLER_BYTES,
                wgpu::BufferUsages::STORAGE,
            ),
            write: buffer(
                "engine/filler-write",
                WRITE_FILLERS * FILLER_BYTES,
                wgpu::BufferUsages::STORAGE,
            ),
            texture: texture("engine/filler-texture", wgpu::TextureFormat::Rgba8Unorm),
            depth: texture("engine/filler-depth", wgpu::TextureFormat::Depth32Float),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/filler-sampler"),
                ..Default::default()
            }),
        }
    }

    // The filler of `slot`, the `written`th written storage binding a group
    // fills.
    fn resource(&self, slot: Slot, written: u64) -> wgpu::BindingResource<'_> {
        match slot {
            Slot::Uniform => self.uniform.as_entire_binding(),
            Slot::Read => self.read.as_entire_binding(),
            Slot::Write => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.write,
                offset: written % WRITE_FILLERS * FILLER_BYTES,
                size: wgpu::BufferSize::new(FILLER_BYTES),
            }),
            Slot::Texture | Slot::Filtered => wgpu::BindingResource::TextureView(&self.texture),
            Slot::Depth => wgpu::BindingResource::TextureView(&self.depth),
            Slot::Sampler => wgpu::BindingResource::Sampler(&self.sampler),
        }
    }
}

/// A bind group of `layout`, made from `slots`, holding `entries` and
/// [`Fillers`] for the bindings they leave out.
pub(crate) fn filled_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    slots: &[(u32, Slot)],
    entries: &[wgpu::BindGroupEntry<'_>],
    fillers: &Fillers,
) -> wgpu::BindGroup {
    let mut written = 0;
    let entries: Vec<wgpu::BindGroupEntry<'_>> = slots
        .iter()
        .map(|&(binding, slot)| {
            entries
                .iter()
                .find(|entry| entry.binding == binding)
                .cloned()
                .unwrap_or_else(|| {
                    let resource = fillers.resource(slot, written);
                    written += u64::from(slot == Slot::Write);
                    wgpu::BindGroupEntry { binding, resource }
                })
        })
        .collect();
    debug_assert!(written <= WRITE_FILLERS, "more written fillers than ranges");
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// The compute pipeline of `module`'s `entry` over `layout`, labeled with
/// the entry point.
pub(crate) fn compute_entry(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    compute(
        device,
        &wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        },
    )
}

/// A pass drawing one triangle over its whole target: no vertex buffers,
/// entry points `vertex` and `fragment` of `module` with pipeline
/// `constants`, color `targets` and an optional depth test.
pub(crate) struct Fullscreen<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub module: &'a wgpu::ShaderModule,
    pub vertex: &'a str,
    pub fragment: &'a str,
    pub constants: &'a [(&'a str, f64)],
    pub targets: &'a [Option<wgpu::ColorTargetState>],
    pub depth: Option<wgpu::DepthStencilState>,
}

/// The pipeline of a fullscreen pass.
pub(crate) fn fullscreen(device: &wgpu::Device, pass: &Fullscreen<'_>) -> wgpu::RenderPipeline {
    render(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some(pass.label),
            layout: Some(pass.layout),
            vertex: wgpu::VertexState {
                module: pass.module,
                entry_point: Some(pass.vertex),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: pass.depth.clone(),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: pass.module,
                entry_point: Some(pass.fragment),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: pass.constants,
                    ..Default::default()
                },
                targets: pass.targets,
            }),
            multiview_mask: None,
            cache: None,
        },
    )
}
