//! Pipeline creation. Every render and compute pipeline the
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

use crate::bindings::{Format, Kind, Sample, Sampling, Stages, Table, View};

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

/// The `wgpu` binding type of `kind`.
fn binding_type(kind: Kind) -> wgpu::BindingType {
    let buffer = |ty, min_size: u64, dynamic| wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset: dynamic,
        min_binding_size: wgpu::BufferSize::new(min_size),
    };
    match kind {
        Kind::Uniform { min_size, dynamic } => {
            buffer(wgpu::BufferBindingType::Uniform, min_size, dynamic)
        }
        Kind::Read { min_size } => buffer(
            wgpu::BufferBindingType::Storage { read_only: true },
            min_size,
            false,
        ),
        Kind::Write => buffer(
            wgpu::BufferBindingType::Storage { read_only: false },
            0,
            false,
        ),
        Kind::Texture { sample, view } => wgpu::BindingType::Texture {
            sample_type: match sample {
                Sample::Float => wgpu::TextureSampleType::Float { filterable: true },
                Sample::Unfilterable => wgpu::TextureSampleType::Float { filterable: false },
                Sample::Depth => wgpu::TextureSampleType::Depth,
                Sample::Uint => wgpu::TextureSampleType::Uint,
            },
            view_dimension: match view {
                View::D2 => wgpu::TextureViewDimension::D2,
                View::D2Array => wgpu::TextureViewDimension::D2Array,
                View::D3 => wgpu::TextureViewDimension::D3,
                View::Cube => wgpu::TextureViewDimension::Cube,
                View::CubeArray => wgpu::TextureViewDimension::CubeArray,
            },
            multisampled: false,
        },
        Kind::Storage(format) => wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: texture_format(format),
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        Kind::Storage3d(format) => wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: texture_format(format),
            view_dimension: wgpu::TextureViewDimension::D3,
        },
        Kind::Sampler(sampling) => wgpu::BindingType::Sampler(match sampling {
            Sampling::Filtering => wgpu::SamplerBindingType::Filtering,
            Sampling::NonFiltering => wgpu::SamplerBindingType::NonFiltering,
            Sampling::Comparison => wgpu::SamplerBindingType::Comparison,
        }),
    }
}

/// The `wgpu` format of `format`.
fn texture_format(format: Format) -> wgpu::TextureFormat {
    match format {
        Format::R32Float => wgpu::TextureFormat::R32Float,
        Format::R32Uint => wgpu::TextureFormat::R32Uint,
        Format::Rgba8Snorm => wgpu::TextureFormat::Rgba8Snorm,
        Format::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        Format::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
    }
}

/// The `wgpu` stages of `stages`.
fn shader_stages(stages: Stages) -> wgpu::ShaderStages {
    let mut shader = wgpu::ShaderStages::NONE;
    if stages.vertex() {
        shader |= wgpu::ShaderStages::VERTEX;
    }
    if stages.fragment() {
        shader |= wgpu::ShaderStages::FRAGMENT;
    }
    if stages.compute() {
        shader |= wgpu::ShaderStages::COMPUTE;
    }
    shader
}

/// The bind group layout of `table` (`crate::bindings`).
///
/// Each distinct pipeline layout costs the browser real work: Firefox's
/// Direct3D 12 backend builds a root signature and three indirect command
/// signatures for it, which an integrated Intel GPU's driver compiles in
/// about 150 ms. Passes of one family therefore share a table listing every
/// binding any of them reads, and their bind groups fill the bindings a pass
/// leaves alone with [`Fillers`].
pub(crate) fn layout(device: &wgpu::Device, table: &Table) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = table
        .bindings
        .iter()
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding: binding.binding,
            visibility: shader_stages(binding.stages),
            ty: binding_type(binding.kind),
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(table.label),
        entries: &entries,
    })
}

/// A shared layout: its table (`crate::bindings`), group
/// layout, and the pipeline layout of passes binding it first, then `more`.
pub(crate) struct Family {
    table: &'static Table,
    group: wgpu::BindGroupLayout,
    pub(crate) pipeline: wgpu::PipelineLayout,
}

impl Family {
    pub(crate) fn new(
        device: &wgpu::Device,
        table: &'static Table,
        more: &[&wgpu::BindGroupLayout],
    ) -> Self {
        let group = layout(device, table);
        let groups: Vec<Option<&wgpu::BindGroupLayout>> = std::iter::once(&group)
            .chain(more.iter().copied())
            .map(Some)
            .collect();
        let pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(table.label),
            bind_group_layouts: &groups,
            immediate_size: 0,
        });
        Self {
            table,
            group,
            pipeline,
        }
    }

    /// A group of this layout binding `resources` by name, filled where
    /// they leave bindings out.
    pub(crate) fn group(
        &self,
        device: &wgpu::Device,
        label: &str,
        resources: &[(&str, wgpu::BindingResource<'_>)],
        fillers: &Fillers,
    ) -> wgpu::BindGroup {
        filled_group(device, label, self.table, &self.group, resources, fillers)
    }
}

/// A texture view as a binding's resource.
pub(crate) fn view(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

/// A sampler as a binding's resource.
pub(crate) fn sampler(sampler: &wgpu::Sampler) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Sampler(sampler)
}

/// Written storage fillers per group: each written binding a group fills
/// gets its own range, since written bindings of a dispatch or draw may not
/// overlap.
const WRITE_FILLERS: u64 = 32;
/// Bytes of each filler buffer range (the storage offset alignment).
const FILLER_BYTES: u64 = 256;

/// Resources standing in for the bindings of a shared layout a pass does not
/// read: the smallest buffers, 2D textures and sampler. Read and written
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
            crate::targets::texture_2d(
                device,
                label,
                (1, 1),
                1,
                1,
                format,
                wgpu::TextureUsages::TEXTURE_BINDING,
            )
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

    // The filler of a binding of `kind`, the `written`th written storage
    // binding a group fills.
    fn resource(&self, kind: Kind, written: u64) -> wgpu::BindingResource<'_> {
        match kind {
            Kind::Uniform { .. } => self.uniform.as_entire_binding(),
            Kind::Read { .. } => self.read.as_entire_binding(),
            Kind::Write => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.write,
                offset: written % WRITE_FILLERS * FILLER_BYTES,
                size: wgpu::BufferSize::new(FILLER_BYTES),
            }),
            Kind::Texture {
                sample: Sample::Depth,
                view: View::D2,
            } => wgpu::BindingResource::TextureView(&self.depth),
            Kind::Texture {
                sample: Sample::Float | Sample::Unfilterable,
                view: View::D2,
            } => wgpu::BindingResource::TextureView(&self.texture),
            Kind::Sampler(Sampling::Filtering) => wgpu::BindingResource::Sampler(&self.sampler),
            kind @ (Kind::Texture { .. }
            | Kind::Storage(_)
            | Kind::Storage3d(_)
            | Kind::Sampler(_)) => {
                panic!("no filler for {kind:?}")
            }
        }
    }
}

/// A bind group of `layout`, built from `table`, binding `resources` by
/// their WGSL names; every binding of the table needs one.
pub(crate) fn group(
    device: &wgpu::Device,
    label: &str,
    table: &Table,
    layout: &wgpu::BindGroupLayout,
    resources: &[(&str, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    debug_assert_eq!(
        resources.len(),
        table.bindings.len(),
        "{label} binds every binding of {}",
        table.label
    );
    let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
        .iter()
        .map(|(name, resource)| wgpu::BindGroupEntry {
            binding: table.binding(name).binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// [`group`] with [`Fillers`] for the bindings `resources` leave out.
pub(crate) fn filled_group(
    device: &wgpu::Device,
    label: &str,
    table: &Table,
    layout: &wgpu::BindGroupLayout,
    resources: &[(&str, wgpu::BindingResource<'_>)],
    fillers: &Fillers,
) -> wgpu::BindGroup {
    for (name, _) in resources {
        table.binding(name);
    }
    let mut written = 0;
    let entries: Vec<wgpu::BindGroupEntry<'_>> = table
        .bindings
        .iter()
        .map(|binding| {
            let resource = resources
                .iter()
                .find(|(name, _)| *name == binding.name)
                .map(|(_, resource)| resource.clone())
                .unwrap_or_else(|| {
                    let filler = fillers.resource(binding.kind, written);
                    written += u64::from(binding.kind == Kind::Write);
                    filler
                });
            wgpu::BindGroupEntry {
                binding: binding.binding,
                resource,
            }
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

/// A single-sample pipeline drawing geometry: `vertex` of `module` reading
/// `buffers`, rasterized by `primitive`, with an optional depth test and an
/// optional fragment stage of the same module.
pub(crate) struct Geometry<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub module: &'a wgpu::ShaderModule,
    pub vertex: &'a str,
    pub vertex_constants: &'a [(&'a str, f64)],
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub primitive: wgpu::PrimitiveState,
    pub depth: Option<wgpu::DepthStencilState>,
    pub fragment: Option<Fragment<'a>>,
}

/// A fragment stage: its entry point, pipeline constants and color targets.
pub(crate) struct Fragment<'a> {
    pub entry: &'a str,
    pub constants: &'a [(&'a str, f64)],
    pub targets: &'a [Option<wgpu::ColorTargetState>],
}

/// The pipeline of `geometry`, counted like every other.
pub(crate) fn geometry(device: &wgpu::Device, geometry: &Geometry<'_>) -> wgpu::RenderPipeline {
    render(
        device,
        &wgpu::RenderPipelineDescriptor {
            label: Some(geometry.label),
            layout: Some(geometry.layout),
            vertex: wgpu::VertexState {
                module: geometry.module,
                entry_point: Some(geometry.vertex),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: geometry.vertex_constants,
                    ..Default::default()
                },
                buffers: geometry.buffers,
            },
            primitive: geometry.primitive,
            depth_stencil: geometry.depth.clone(),
            multisample: wgpu::MultisampleState::default(),
            fragment: geometry
                .fragment
                .as_ref()
                .map(|fragment| wgpu::FragmentState {
                    module: geometry.module,
                    entry_point: Some(fragment.entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: fragment.constants,
                        ..Default::default()
                    },
                    targets: fragment.targets,
                }),
            multiview_mask: None,
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

/// The vertex attributes of upload struct `$ty`'s fields
/// (`crate::gpu_layouts`): `Type { location => field as Format, .. }`, each
/// at its field's offset. A format whose size differs from its field's fails
/// to compile.
macro_rules! vertex_attributes {
    ($ty:ident { $($location:literal => $field:ident as $format:ident),+ $(,)? }) => {
        [$({
            const FIELD: &$crate::gpu_layouts::Field = $ty::LAYOUT.field(stringify!($field));
            const _: () = assert!(wgpu::VertexFormat::$format.size() as usize == FIELD.size);
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::$format,
                offset: FIELD.offset as u64,
                shader_location: $location,
            }
        }),+]
    };
}
pub(crate) use vertex_attributes;

/// The pipeline of a fullscreen pass.
pub(crate) fn fullscreen(device: &wgpu::Device, pass: &Fullscreen<'_>) -> wgpu::RenderPipeline {
    geometry(
        device,
        &Geometry {
            label: pass.label,
            layout: pass.layout,
            module: pass.module,
            vertex_constants: &[],
            vertex: pass.vertex,
            buffers: &[],
            primitive: wgpu::PrimitiveState::default(),
            depth: pass.depth.clone(),
            fragment: Some(Fragment {
                entry: pass.fragment,
                constants: pass.constants,
                targets: pass.targets,
            }),
        },
    )
}
