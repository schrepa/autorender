//! Pipeline creation. Every render and compute pipeline the renderer
//! builds goes through these functions, which key it by what makes it: its
//! shader module and entry points by name ([`crate::shaders`]), its
//! pipeline layout by the declaration it was made from
//! ([`crate::pipeline_layouts`]: a device makes one layout for each) and
//! every state it sets. A device creates each key once, whoever asks:
//! those asking after take the same pipeline. Creations are counted, so
//! frames report when compiling ends: a frame that creates none compiles
//! nothing, and hosts reveal the canvas once frames stop creating them.
//!
//! The keys a device created are its scene's pipeline list
//! ([`PipelineCounts::list`]), a record hosts keep and the cook ships. A
//! later session creates every pipeline on it up front while it prepares
//! ([`PipelineCounts::create_listed`]): natively on several threads, in
//! browsers all at once, so they compile while the scene loads; and counts
//! those it then creates that the list lacks. A list names the format of
//! the canvas it drew to, and its pipelines drawing there draw to the
//! device's own. Lists are made on a GPU, which creates only what its
//! limits and formats allow: they are cooked once, never compared across
//! GPUs.
#![expect(
    clippy::expect_used,
    reason = "keys serialize, and layouts come from pipeline_layout"
)]

use crate::shaders::Entry;
use engine_core::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::bindings::{Format, Kind, Sample, Sampling, Stages, Table, View};
use crate::pipeline_layouts::{PIPELINE_LAYOUTS, PipelineLayout, StageLimits};

thread_local! {
    static CREATED: Cell<u32> = const { Cell::new(0) };
    static REGISTRIES: RefCell<Vec<Registry>> = const { RefCell::new(Vec::new()) };
}

/// Counts `created` pipelines created.
fn count(created: u32) {
    CREATED.with(|counted| counted.set(counted.get().saturating_add(created)));
}

/// The pipelines created since the last call.
fn take_created() -> u32 {
    CREATED.with(|created| created.replace(0))
}

/// What makes a pipeline: everything its descriptor sets, its objects by
/// name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Key {
    label: String,
    /// Its pipeline layout's label (`crate::pipeline_layouts`).
    layout: String,
    /// Its shader module's name (`crate::shaders::MODULES`).
    module: String,
    stages: KeyStages,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum KeyStages {
    Compute { entry: String },
    Render(Box<RenderKey>),
}

/// A render pipeline's stages and states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RenderKey {
    vertex: String,
    vertex_constants: Vec<(String, f64)>,
    buffers: Vec<Option<BufferKey>>,
    primitive: wgpu::PrimitiveState,
    depth: Option<wgpu::DepthStencilState>,
    samples: u32,
    fragment: Option<FragmentKey>,
}

/// A vertex buffer's layout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct BufferKey {
    stride: u64,
    step: wgpu::VertexStepMode,
    attributes: Vec<wgpu::VertexAttribute>,
}

/// A fragment stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct FragmentKey {
    entry: String,
    constants: Vec<(String, f64)>,
    targets: Vec<Option<wgpu::ColorTargetState>>,
}

/// Pipeline constants as keys hold them.
fn constants(constants: &[(&str, f64)]) -> Vec<(String, f64)> {
    constants
        .iter()
        .map(|&(name, value)| (name.to_owned(), value))
        .collect()
}

/// Pipeline constants as descriptors take them.
fn named(constants: &[(String, f64)]) -> Vec<(&str, f64)> {
    constants
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect()
}

impl Key {
    /// Its text, which a device's pipelines and lists are kept by.
    fn text(&self) -> String {
        serde_json::to_string(self).expect("pipeline keys serialize")
    }

    /// The pipeline of this key over `layout` and `module`, created and
    /// counted.
    fn create(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
    ) -> Created {
        count(1);
        let render = match &self.stages {
            KeyStages::Compute { entry } => {
                return Created::Compute(create_compute(
                    device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some(self.label.as_str()),
                        layout: Some(layout),
                        module,
                        entry_point: Some(entry.as_str()),
                        compilation_options: Default::default(),
                        cache: None,
                    },
                ));
            }
            KeyStages::Render(render) => render,
        };
        let vertex_constants = named(&render.vertex_constants);
        let buffers: Vec<Option<wgpu::VertexBufferLayout<'_>>> = render
            .buffers
            .iter()
            .map(|buffer| {
                buffer.as_ref().map(|buffer| wgpu::VertexBufferLayout {
                    array_stride: buffer.stride,
                    step_mode: buffer.step,
                    attributes: &buffer.attributes,
                })
            })
            .collect();
        let fragment_constants = render
            .fragment
            .as_ref()
            .map(|fragment| named(&fragment.constants))
            .unwrap_or_default();
        Created::Render(create_render(
            device,
            &wgpu::RenderPipelineDescriptor {
                label: Some(self.label.as_str()),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(render.vertex.as_str()),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &vertex_constants,
                        ..Default::default()
                    },
                    buffers: &buffers,
                },
                primitive: render.primitive,
                depth_stencil: render.depth.clone(),
                multisample: wgpu::MultisampleState {
                    count: render.samples,
                    ..Default::default()
                },
                fragment: render
                    .fragment
                    .as_ref()
                    .map(|fragment| wgpu::FragmentState {
                        module,
                        entry_point: Some(fragment.entry.as_str()),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &fragment_constants,
                            ..Default::default()
                        },
                        targets: &fragment.targets,
                    }),
                multiview_mask: None,
                cache: None,
            },
        ))
    }

    /// Whether `device` can create it: its layout keeps within the
    /// device's limits, and the formats it draws to render (and blend, and
    /// take its samples) there.
    fn creatable(&self, device: &wgpu::Device) -> bool {
        let Some(declared) = declared(&self.layout) else {
            return false;
        };
        if !fits(&declared.limits, &device.limits()) {
            return false;
        }
        let KeyStages::Render(render) = &self.stages else {
            return true;
        };
        let features = device.features();
        let renders = |format: wgpu::TextureFormat, blends: bool| {
            let supported = format.guaranteed_format_features(features);
            supported
                .allowed_usages
                .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
                && (!blends
                    || supported
                        .flags
                        .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE))
                && (render.samples <= 1 || supported.flags.sample_count_supported(render.samples))
        };
        render
            .fragment
            .iter()
            .flat_map(|fragment| fragment.targets.iter().flatten())
            .all(|target| renders(target.format, target.blend.is_some()))
            && render
                .depth
                .as_ref()
                .is_none_or(|depth| renders(depth.format, false))
    }

    /// The key drawing to `canvas` where it drew to `listed`, the canvas
    /// format its list names.
    fn retargeted(mut self, listed: wgpu::TextureFormat, canvas: wgpu::TextureFormat) -> Self {
        if let KeyStages::Render(render) = &mut self.stages
            && let Some(fragment) = &mut render.fragment
        {
            for target in fragment.targets.iter_mut().flatten() {
                if target.format == listed {
                    target.format = canvas;
                }
            }
        }
        self
    }
}

/// A pipeline as created.
enum Created {
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
}

/// The layout labeled `label`, as `crate::pipeline_layouts` declares it.
fn declared(label: &str) -> Option<&'static PipelineLayout> {
    PIPELINE_LAYOUTS
        .iter()
        .find(|(_, declared)| declared.label == label)
        .map(|&(_, declared)| declared)
}

/// Whether `limits` keep within what `device` grants.
fn fits(limits: &StageLimits, device: &wgpu::Limits) -> bool {
    limits.bind_groups <= device.max_bind_groups
        && limits.uniform_buffers <= device.max_uniform_buffers_per_shader_stage
        && limits.storage_buffers <= device.max_storage_buffers_per_shader_stage
        && limits.sampled_textures <= device.max_sampled_textures_per_shader_stage
        && limits.samplers <= device.max_samplers_per_shader_stage
        && limits.storage_textures <= device.max_storage_textures_per_shader_stage
        && limits.dynamic_uniform_buffers <= device.max_dynamic_uniform_buffers_per_pipeline_layout
        && limits.dynamic_storage_buffers <= device.max_dynamic_storage_buffers_per_pipeline_layout
}

/// A device's pipelines by key, its pipeline layouts by declaration and
/// their labels, the shader modules its lists' pipelines were created
/// with, and what its scene's list held: the keys it named, the pipelines
/// created up front and those created since that it lacked.
struct Registry {
    device: wgpu::Device,
    canvas: Option<wgpu::TextureFormat>,
    layouts: BTreeMap<&'static str, wgpu::PipelineLayout>,
    labels: HashMap<wgpu::PipelineLayout, &'static str>,
    modules: BTreeMap<&'static str, wgpu::ShaderModule>,
    render: BTreeMap<String, wgpu::RenderPipeline>,
    compute: BTreeMap<String, wgpu::ComputePipeline>,
    listed: Option<BTreeSet<String>>,
    up_front: u32,
    unlisted: u32,
}

impl Registry {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            device: device.clone(),
            canvas: None,
            layouts: BTreeMap::new(),
            labels: HashMap::default(),
            modules: BTreeMap::new(),
            render: BTreeMap::new(),
            compute: BTreeMap::new(),
            listed: None,
            up_front: 0,
            unlisted: 0,
        }
    }

    /// The label `crate::pipeline_layouts` declares `layout` by.
    fn label(&self, layout: &wgpu::PipelineLayout) -> &'static str {
        self.labels
            .get(layout)
            .copied()
            .expect("pipeline layouts come from pipelines::pipeline_layout")
    }

    /// Keeps `layout` as the device's of `label`.
    fn keep_layout(&mut self, label: &'static str, layout: wgpu::PipelineLayout) {
        self.labels.insert(layout.clone(), label);
        self.layouts.insert(label, layout);
    }

    /// The device's layout labeled `label`, made from its declaration's
    /// tables unless made.
    fn declared_layout(&mut self, label: &str) -> Option<wgpu::PipelineLayout> {
        let declared = declared(label)?;
        if let Some(layout) = self.layouts.get(declared.label) {
            return Some(layout.clone());
        }
        let groups: Vec<GroupLayout> = declared
            .groups
            .iter()
            .map(|&table| layout(&self.device, table))
            .collect();
        let layouts: Vec<Option<&wgpu::BindGroupLayout>> =
            groups.iter().map(|group| Some(&group.layout)).collect();
        let layout = create_layout(&self.device, declared.label, &layouts);
        self.keep_layout(declared.label, layout.clone());
        Some(layout)
    }

    /// The module named `name` (`crate::shaders::MODULES`), created unless
    /// it was.
    fn module(&mut self, name: &str) -> Option<wgpu::ShaderModule> {
        let &(name, source) = crate::shaders::MODULES
            .iter()
            .find(|(module, _)| *module == name)?;
        let device = &self.device;
        Some(
            self.modules
                .entry(name)
                .or_insert_with(|| {
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    })
                })
                .clone(),
        )
    }

    /// A pipeline of key text `text` is being created: one the list
    /// lacked, once there is one.
    fn creating(&mut self, text: &str) {
        if self
            .listed
            .as_ref()
            .is_some_and(|listed| !listed.contains(text))
        {
            self.unlisted += 1;
        }
    }

    /// Its pipelines' key texts, in order.
    fn keys(&self) -> BTreeSet<&String> {
        self.render.keys().chain(self.compute.keys()).collect()
    }
}

/// `act` on `device`'s registry, made with its first use.
fn with_registry<R>(device: &wgpu::Device, act: impl FnOnce(&mut Registry) -> R) -> R {
    REGISTRIES.with_borrow_mut(|registries| {
        let index = match registries
            .iter()
            .position(|registry| registry.device == *device)
        {
            Some(index) => index,
            None => {
                registries.push(Registry::new(device));
                registries.len() - 1
            }
        };
        act(&mut registries[index])
    })
}

/// The tag of a pipeline list's record.
const LIST: &[u8; 4] = b"PIPL";

/// The canvas format and keys of the pipeline list `record` holds.
fn read_list(record: &[u8]) -> Result<(Option<wgpu::TextureFormat>, Vec<Key>), String> {
    let mut rest = record
        .strip_prefix(LIST)
        .ok_or("not a pipeline list's record")?;
    let mut texts = Vec::new();
    while let Some((length, after)) = rest.split_first_chunk::<4>() {
        let length = u32::from_le_bytes(*length) as usize;
        let text = after.get(..length).ok_or("the pipeline list ends early")?;
        texts.push(std::str::from_utf8(text).map_err(|error| error.to_string())?);
        rest = &after[length..];
    }
    let Some((canvas, keys)) = texts.split_first().filter(|_| rest.is_empty()) else {
        return Err("the pipeline list ends early".into());
    };
    let canvas = match *canvas {
        "" => None,
        canvas => Some(serde_json::from_str(canvas).map_err(|error| error.to_string())?),
    };
    let keys = keys
        .iter()
        .map(|text| serde_json::from_str(text).map_err(|error| error.to_string()))
        .collect::<Result<_, String>>()?;
    Ok((canvas, keys))
}

/// A pipeline to create up front: its key text, key, layout and module.
type Wanted = (String, Key, wgpu::PipelineLayout, wgpu::ShaderModule);

/// `wanted`'s pipelines, created on as many threads as the processors
/// allow, each in its own validation scope: those failing validation are
/// left out, to be created as frames first draw with them.
#[cfg(not(target_arch = "wasm32"))]
fn create_all(device: &wgpu::Device, wanted: &[Wanted]) -> Vec<(String, Created)> {
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = wanted.len().div_ceil(threads.clamp(1, 8)).max(1);
    let parts = std::thread::scope(|scope| {
        let spawned: Vec<_> = wanted
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let made: Vec<(String, Created)> = part
                        .iter()
                        .filter_map(|(text, key, layout, module)| {
                            let errors = crate::gpu_calls::validation_scope(device);
                            let created = key.create(device, layout, module);
                            let failed = crate::platform::ready(errors.pop()).flatten();
                            failed.is_none().then(|| (text.clone(), created))
                        })
                        .collect();
                    // Counted on this thread: the frame's count is the
                    // renderer's thread's.
                    (made, take_created())
                })
            })
            .collect();
        spawned
            .into_iter()
            .map(|part| part.join().expect("creating pipelines does not panic"))
            .collect::<Vec<_>>()
    });
    let mut made = Vec::new();
    for (part, created) in parts {
        count(created);
        made.extend(part);
    }
    made
}

/// `wanted`'s pipelines, created at once in one validation scope, so the
/// browser compiles them side by side: what fails validation the scope
/// reports, and the pipeline's first draw then fails as it would have.
#[cfg(target_arch = "wasm32")]
fn create_all(device: &wgpu::Device, wanted: &[Wanted]) -> Vec<(String, Created)> {
    let errors = crate::gpu_calls::validation_scope(device);
    let made = wanted
        .iter()
        .map(|(text, key, layout, module)| (text.clone(), key.create(device, layout, module)))
        .collect();
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(error) = errors.pop().await {
            web_sys::console::warn_1(&format!("A listed pipeline failed: {error}").into());
        }
    });
    made
}

/// Creates a render pipeline: the one way the renderer does.
#[expect(clippy::disallowed_methods, reason = "the one way to create pipelines")]
fn create_render(
    device: &wgpu::Device,
    descriptor: &wgpu::RenderPipelineDescriptor<'_>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(descriptor)
}

/// Creates a compute pipeline: the one way the renderer does.
#[expect(clippy::disallowed_methods, reason = "the one way to create pipelines")]
fn create_compute(
    device: &wgpu::Device,
    descriptor: &wgpu::ComputePipelineDescriptor<'_>,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(descriptor)
}

/// Makes a pipeline layout: the one way the renderer does.
#[expect(
    clippy::disallowed_methods,
    reason = "the one way to make pipeline layouts"
)]
fn create_layout(
    device: &wgpu::Device,
    label: &str,
    groups: &[Option<&wgpu::BindGroupLayout>],
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: groups,
        immediate_size: 0,
    })
}

/// The pipelines a renderer created, and whether their compiles may still
/// run: they precede the work submitted after them, so once the GPU
/// completes the last frame that created some, they are done. While it
/// lives, its device keeps its pipelines by key, which go with it.
pub(crate) struct PipelineCounts {
    device: wgpu::Device,
    created: u32,
    // Frames that created some, and the latest of those the GPU completed.
    frames: u64,
    compiled: Arc<AtomicU64>,
}

impl Drop for PipelineCounts {
    fn drop(&mut self) {
        // A thread ending drops its registries itself.
        let _ = REGISTRIES.try_with(|registries| {
            registries
                .borrow_mut()
                .retain(|registry| registry.device != self.device);
        });
    }
}

impl PipelineCounts {
    /// The counts of the renderer of `device`, which draws to a canvas of
    /// `canvas`'s format.
    pub(crate) fn new(device: &wgpu::Device, canvas: wgpu::TextureFormat) -> Self {
        with_registry(device, |registry| registry.canvas = Some(canvas));
        Self {
            device: device.clone(),
            created: 0,
            frames: 0,
            compiled: Arc::default(),
        }
    }

    /// The scene's pipeline list: a record of every pipeline the device
    /// created, by key, after the format of the canvas they drew to.
    pub(crate) fn list(&self) -> Vec<u8> {
        with_registry(&self.device, |registry| {
            let canvas = registry
                .canvas
                .map(|canvas| serde_json::to_string(&canvas).expect("formats serialize"))
                .unwrap_or_default();
            let mut record = LIST.to_vec();
            for text in std::iter::once(&canvas).chain(registry.keys()) {
                record.extend((text.len() as u32).to_le_bytes());
                record.extend(text.as_bytes());
            }
            record
        })
    }

    /// Creates every pipeline the pipeline list `record` names that the
    /// device can create and has yet to (natively on several threads),
    /// drawing to the device's canvas where the list's drew to its own;
    /// from then on, pipelines created that the list lacks are counted
    /// ([`Self::listed`]). Returns how many it created.
    pub(crate) fn create_listed(&self, record: &[u8]) -> Result<u32, String> {
        let (listed_canvas, keys) = read_list(record)?;
        let device = &self.device;
        with_registry(device, |registry| {
            let keys: Vec<(String, Key)> = keys
                .into_iter()
                .map(|key| match (listed_canvas, registry.canvas) {
                    (Some(listed), Some(canvas)) => key.retargeted(listed, canvas),
                    _ => key,
                })
                .map(|key| (key.text(), key))
                .collect();
            registry.listed = Some(keys.iter().map(|(text, _)| text.clone()).collect());
            registry.unlisted = 0;
            let mut wanted = Vec::new();
            for (text, key) in keys {
                if registry.render.contains_key(&text)
                    || registry.compute.contains_key(&text)
                    || !key.creatable(device)
                {
                    continue;
                }
                let (Some(layout), Some(module)) = (
                    registry.declared_layout(&key.layout),
                    registry.module(&key.module),
                ) else {
                    continue;
                };
                wanted.push((text, key, layout, module));
            }
            let mut created = 0;
            for (text, made) in create_all(device, &wanted) {
                created += 1;
                match made {
                    Created::Render(pipeline) => {
                        registry.render.insert(text, pipeline);
                    }
                    Created::Compute(pipeline) => {
                        registry.compute.insert(text, pipeline);
                    }
                }
            }
            registry.up_front += created;
            Ok(created)
        })
    }

    /// What the scene's pipeline list did: the pipelines it created up
    /// front, and those the session created that it lacked.
    pub(crate) fn listed(&self) -> (u32, u32) {
        with_registry(&self.device, |registry| {
            (registry.up_front, registry.unlisted)
        })
    }

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
        Kind::StorageArray(format) => wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: texture_format(format),
            view_dimension: wgpu::TextureViewDimension::D2Array,
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
pub(crate) fn layout(device: &wgpu::Device, table: &'static Table) -> GroupLayout {
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
    GroupLayout {
        table,
        layout: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(table.label),
            entries: &entries,
        }),
    }
}

/// A bind group layout and the table (`crate::bindings`) it was made from,
/// which [`pipeline_layout`] checks against the tables a pipeline layout
/// declares.
#[derive(Clone, Debug)]
pub(crate) struct GroupLayout {
    table: &'static Table,
    layout: wgpu::BindGroupLayout,
}

impl std::ops::Deref for GroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
}

/// The pipeline layout `declared` names (`crate::pipeline_layouts`), over
/// `groups`, layouts of its tables in order: the one way pipeline layouts
/// are made, so each is one `cargo xtask lint` counted, and the device
/// makes each once, whoever asks (layouts of one table are equivalent, so
/// groups made with any bind with it). Groups of other tables than it
/// declares are a programming error.
pub(crate) fn pipeline_layout(
    device: &wgpu::Device,
    declared: &PipelineLayout,
    groups: &[&GroupLayout],
) -> wgpu::PipelineLayout {
    assert!(
        groups.len() == declared.groups.len()
            && groups
                .iter()
                .zip(declared.groups)
                .all(|(group, table)| group.table.label == table.label),
        "{} binds other groups than it declares (crate::pipeline_layouts)",
        declared.label
    );
    with_registry(device, |registry| {
        if let Some(layout) = registry.layouts.get(declared.label) {
            return layout.clone();
        }
        let layouts: Vec<Option<&wgpu::BindGroupLayout>> =
            groups.iter().map(|group| Some(&group.layout)).collect();
        let layout = create_layout(device, declared.label, &layouts);
        registry.keep_layout(declared.label, layout.clone());
        layout
    })
}

/// A shared layout: its table (`crate::bindings`), group
/// layout, and the pipeline layout of passes binding it first, then `more`.
pub(crate) struct Family {
    group: GroupLayout,
    pub(crate) pipeline: wgpu::PipelineLayout,
}

impl Family {
    /// The family of pipelines over `declared`: a group of its first table,
    /// then the layouts of the rest, `more`.
    pub(crate) fn new(
        device: &wgpu::Device,
        declared: &'static crate::pipeline_layouts::PipelineLayout,
        more: &[&GroupLayout],
    ) -> Self {
        let group = layout(device, declared.groups[0]);
        let groups: Vec<&GroupLayout> = std::iter::once(&group)
            .chain(more.iter().copied())
            .collect();
        let pipeline = pipeline_layout(device, declared, &groups);
        Self { group, pipeline }
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
        filled_group(
            device,
            label,
            self.group.table,
            &self.group,
            resources,
            fillers,
        )
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

/// Bytes of the uniform filler: as large as the largest uniform a shared
/// layout's filled bindings may declare.
const UNIFORM_FILLER_BYTES: u64 = 4096;

/// Resources standing in for the bindings of a shared layout a pass does not
/// read: the smallest buffers, 2D textures (and a depth array) and samplers.
/// Read and written storage have separate buffers, since one buffer may not
/// be both in a dispatch or draw.
pub(crate) struct Fillers {
    uniform: wgpu::Buffer,
    read: wgpu::Buffer,
    write: wgpu::Buffer,
    texture: wgpu::TextureView,
    depth: wgpu::TextureView,
    depth_array: wgpu::TextureView,
    sampler: wgpu::Sampler,
    comparison: wgpu::Sampler,
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
        };
        let depth = texture("engine/filler-depth", wgpu::TextureFormat::Depth32Float);
        Self {
            uniform: buffer(
                "engine/filler-uniform",
                UNIFORM_FILLER_BYTES,
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
            texture: texture("engine/filler-texture", wgpu::TextureFormat::Rgba8Unorm)
                .create_view(&Default::default()),
            depth_array: depth.create_view(&wgpu::TextureViewDescriptor {
                label: Some("engine/filler-depth-array"),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            }),
            depth: depth.create_view(&Default::default()),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/filler-sampler"),
                ..Default::default()
            }),
            comparison: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("engine/filler-comparison"),
                compare: Some(wgpu::CompareFunction::LessEqual),
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
                sample: Sample::Depth,
                view: View::D2Array,
            } => wgpu::BindingResource::TextureView(&self.depth_array),
            Kind::Texture {
                sample: Sample::Float | Sample::Unfilterable,
                view: View::D2,
            } => wgpu::BindingResource::TextureView(&self.texture),
            Kind::Sampler(Sampling::Filtering) => wgpu::BindingResource::Sampler(&self.sampler),
            Kind::Sampler(Sampling::Comparison) => wgpu::BindingResource::Sampler(&self.comparison),
            kind @ (Kind::Texture { .. }
            | Kind::Storage(_)
            | Kind::StorageArray(_)
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
    layout: &GroupLayout,
    resources: &[(&str, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    debug_assert_eq!(layout.table.label, table.label, "{label}'s layout");
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
    bind_group(device, label, layout, &entries)
}

/// [`group`] with [`Fillers`] for the bindings `resources` leave out.
pub(crate) fn filled_group(
    device: &wgpu::Device,
    label: &str,
    table: &Table,
    layout: &GroupLayout,
    resources: &[(&str, wgpu::BindingResource<'_>)],
    fillers: &Fillers,
) -> wgpu::BindGroup {
    debug_assert_eq!(layout.table.label, table.label, "{label}'s layout");
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
    bind_group(device, label, layout, &entries)
}

/// The bind group of `layout` over `entries`: the one way bind groups are
/// made, by [`group`] and [`filled_group`] from their tables.
#[expect(clippy::disallowed_methods, reason = "the one way to make bind groups")]
fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &GroupLayout,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries,
    })
}

/// The compute pipeline of `module`'s `entry` over `layout`, labeled with
/// the entry point: the device's of its key, created unless it was.
pub(crate) fn compute_entry(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: Entry,
) -> wgpu::ComputePipeline {
    with_registry(device, |registry| {
        let key = Key {
            label: entry.name.to_owned(),
            layout: registry.label(layout).to_owned(),
            module: entry.module.to_owned(),
            stages: KeyStages::Compute {
                entry: entry.name.to_owned(),
            },
        };
        let text = key.text();
        if let Some(pipeline) = registry.compute.get(&text) {
            return pipeline.clone();
        }
        registry.creating(&text);
        let Created::Compute(pipeline) = key.create(device, layout, module) else {
            unreachable!("a compute key creates a compute pipeline");
        };
        registry.compute.insert(text, pipeline.clone());
        pipeline
    })
}

/// A pipeline drawing geometry: `vertex` of `module` reading `buffers`,
/// rasterized by `primitive`, with an optional depth test and an optional
/// fragment stage of the same module.
pub(crate) struct Geometry<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub module: &'a wgpu::ShaderModule,
    pub vertex: Entry,
    pub vertex_constants: &'a [(&'a str, f64)],
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub primitive: wgpu::PrimitiveState,
    pub depth: Option<wgpu::DepthStencilState>,
    pub fragment: Option<Fragment<'a>>,
}

/// A fragment stage: its entry point, pipeline constants and color targets.
pub(crate) struct Fragment<'a> {
    pub entry: Entry,
    pub constants: &'a [(&'a str, f64)],
    pub targets: &'a [Option<wgpu::ColorTargetState>],
}

/// The pipeline of `geometry`, one sample a pixel.
pub(crate) fn geometry(device: &wgpu::Device, geometry: &Geometry<'_>) -> wgpu::RenderPipeline {
    sampled(device, geometry, 1)
}

/// The pipeline of `geometry` taking `samples` samples a pixel: the
/// device's of its key, created unless it was.
pub(crate) fn sampled(
    device: &wgpu::Device,
    geometry: &Geometry<'_>,
    samples: u32,
) -> wgpu::RenderPipeline {
    debug_assert!(
        geometry
            .fragment
            .as_ref()
            .is_none_or(|fragment| fragment.entry.module == geometry.vertex.module),
        "{}'s stages come from one module",
        geometry.label
    );
    with_registry(device, |registry| {
        let key = Key {
            label: geometry.label.to_owned(),
            layout: registry.label(geometry.layout).to_owned(),
            module: geometry.vertex.module.to_owned(),
            stages: KeyStages::Render(Box::new(RenderKey {
                vertex: geometry.vertex.name.to_owned(),
                vertex_constants: constants(geometry.vertex_constants),
                buffers: geometry
                    .buffers
                    .iter()
                    .map(|buffer| {
                        buffer.as_ref().map(|buffer| BufferKey {
                            stride: buffer.array_stride,
                            step: buffer.step_mode,
                            attributes: buffer.attributes.to_vec(),
                        })
                    })
                    .collect(),
                primitive: geometry.primitive,
                depth: geometry.depth.clone(),
                samples,
                fragment: geometry.fragment.as_ref().map(|fragment| FragmentKey {
                    entry: fragment.entry.name.to_owned(),
                    constants: constants(fragment.constants),
                    targets: fragment.targets.to_vec(),
                }),
            })),
        };
        let text = key.text();
        if let Some(pipeline) = registry.render.get(&text) {
            return pipeline.clone();
        }
        registry.creating(&text);
        let Created::Render(pipeline) = key.create(device, geometry.layout, geometry.module) else {
            unreachable!("a render key creates a render pipeline");
        };
        registry.render.insert(text, pipeline.clone());
        pipeline
    })
}

/// A pass drawing one triangle over its whole target: no vertex buffers,
/// entry points `vertex` and `fragment` of `module` with pipeline
/// `constants`, color `targets` and an optional depth test.
pub(crate) struct Fullscreen<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub module: &'a wgpu::ShaderModule,
    pub vertex: Entry,
    pub fragment: Entry,
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
