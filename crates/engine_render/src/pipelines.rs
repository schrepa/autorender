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

fn count(created: u32) {
    CREATED.with(|counted| counted.set(counted.get().saturating_add(created)));
}

fn take_created() -> u32 {
    CREATED.with(|created| created.replace(0))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Key {
    label: String,
    layout: String,
    module: String,
    stages: KeyStages,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum KeyStages {
    Compute { entry: String },
    Render(Box<RenderKey>),
}

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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct BufferKey {
    stride: u64,
    step: wgpu::VertexStepMode,
    attributes: Vec<wgpu::VertexAttribute>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct FragmentKey {
    entry: String,
    constants: Vec<(String, f64)>,
    targets: Vec<Option<wgpu::ColorTargetState>>,
}

fn constants(constants: &[(&str, f64)]) -> Vec<(String, f64)> {
    constants
        .iter()
        .map(|&(name, value)| (name.to_owned(), value))
        .collect()
}

fn named(constants: &[(String, f64)]) -> Vec<(&str, f64)> {
    constants
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect()
}

impl Key {
    fn text(&self) -> String {
        serde_json::to_string(self).expect("pipeline keys serialize")
    }

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

enum Created {
    Render(wgpu::RenderPipeline),
    Compute(wgpu::ComputePipeline),
}

fn declared(label: &str) -> Option<&'static PipelineLayout> {
    PIPELINE_LAYOUTS
        .iter()
        .find(|(_, declared)| declared.label == label)
        .map(|&(_, declared)| declared)
}

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

    fn label(&self, layout: &wgpu::PipelineLayout) -> &'static str {
        self.labels
            .get(layout)
            .copied()
            .expect("pipeline layouts come from pipelines::pipeline_layout")
    }

    fn keep_layout(&mut self, label: &'static str, layout: wgpu::PipelineLayout) {
        self.labels.insert(layout.clone(), label);
        self.layouts.insert(label, layout);
    }

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

    fn creating(&mut self, text: &str) {
        if self
            .listed
            .as_ref()
            .is_some_and(|listed| !listed.contains(text))
        {
            self.unlisted += 1;
        }
    }

    fn keys(&self) -> BTreeSet<&String> {
        self.render.keys().chain(self.compute.keys()).collect()
    }
}

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

const LIST: &[u8; 4] = b"PIPL";

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

type Wanted = (String, Key, wgpu::PipelineLayout, wgpu::ShaderModule);

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

#[expect(clippy::disallowed_methods, reason = "the one way to create pipelines")]
fn create_render(
    device: &wgpu::Device,
    descriptor: &wgpu::RenderPipelineDescriptor<'_>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(descriptor)
}

#[expect(clippy::disallowed_methods, reason = "the one way to create pipelines")]
fn create_compute(
    device: &wgpu::Device,
    descriptor: &wgpu::ComputePipelineDescriptor<'_>,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(descriptor)
}

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

pub(crate) struct PipelineCounts {
    device: wgpu::Device,
    created: u32,
    frames: u64,
    compiled: Arc<AtomicU64>,
}

impl Drop for PipelineCounts {
    fn drop(&mut self) {
        let _ = REGISTRIES.try_with(|registries| {
            registries
                .borrow_mut()
                .retain(|registry| registry.device != self.device);
        });
    }
}

impl PipelineCounts {
    pub(crate) fn new(device: &wgpu::Device, canvas: wgpu::TextureFormat) -> Self {
        with_registry(device, |registry| registry.canvas = Some(canvas));
        Self {
            device: device.clone(),
            created: 0,
            frames: 0,
            compiled: Arc::default(),
        }
    }

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

    pub(crate) fn listed(&self) -> (u32, u32) {
        with_registry(&self.device, |registry| {
            (registry.up_front, registry.unlisted)
        })
    }

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

    pub(crate) fn created(&self) -> u32 {
        self.created
    }

    pub(crate) fn compiling(&self) -> bool {
        self.compiled.load(Ordering::Relaxed) < self.frames
    }
}

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

fn texture_format(format: Format) -> wgpu::TextureFormat {
    match format {
        Format::R32Float => wgpu::TextureFormat::R32Float,
        Format::R32Uint => wgpu::TextureFormat::R32Uint,
        Format::Rgba8Snorm => wgpu::TextureFormat::Rgba8Snorm,
        Format::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        Format::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
    }
}

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

pub(crate) struct Family {
    group: GroupLayout,
    pub(crate) pipeline: wgpu::PipelineLayout,
}

impl Family {
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

pub(crate) fn view(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

pub(crate) fn sampler(sampler: &wgpu::Sampler) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Sampler(sampler)
}

const WRITE_FILLERS: u64 = 32;
const FILLER_BYTES: u64 = 256;

const UNIFORM_FILLER_BYTES: u64 = 4096;

pub(crate) struct Fillers {
    uniform: wgpu::Buffer,
    read: wgpu::Buffer,
    write: wgpu::Buffer,
    texture: wgpu::TextureView,
    unsigned: wgpu::TextureView,
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
            unsigned: texture("engine/filler-unsigned", wgpu::TextureFormat::Rgba16Uint)
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
                sample: Sample::Uint,
                view: View::D2,
            } => wgpu::BindingResource::TextureView(&self.unsigned),
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

pub(crate) struct Fragment<'a> {
    pub entry: Entry,
    pub constants: &'a [(&'a str, f64)],
    pub targets: &'a [Option<wgpu::ColorTargetState>],
}

pub(crate) fn geometry(device: &wgpu::Device, geometry: &Geometry<'_>) -> wgpu::RenderPipeline {
    sampled(device, geometry, 1)
}

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
