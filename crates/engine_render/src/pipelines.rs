#![expect(
    clippy::expect_used,
    reason = "keys serialize, and layouts come from pipeline_layout"
)]

use crate::platform::failed;
use crate::shaders::Entry;
use engine_core::collections::HashMap;
use engine_core::{Error, Expected};
use serde::{Deserialize, Serialize};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use crate::bindings::{
    Format, GroupRef, Kind, ResolvedBinding, ResolvedLayout, Sample, Sampling, SharedGroups,
    Stages, Table, View,
};
use crate::linker::{Composed, Linker};
use crate::pipeline_layouts::{PIPELINE_LAYOUTS, PipelineLayout, StageLimits};

pub type Layouts = &'static [(&'static str, &'static PipelineLayout)];

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
    stages: KeyStages,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum KeyStages {
    Compute { module: String, entry: String },
    Render(Box<RenderKey>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RenderKey {
    vertex_module: String,
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
    module: String,
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

#[derive(Clone, Debug)]
struct Linked {
    module: wgpu::ShaderModule,
    constants: Vec<(String, f64)>,
}

#[derive(Clone, Debug)]
struct Modules {
    first: Linked,
    fragment: Option<Linked>,
}

impl Key {
    fn text(&self) -> String {
        serde_json::to_string(self).expect("pipeline_key_serializes")
    }

    fn create(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        modules: &Modules,
    ) -> Created {
        count(1);
        let render = match &self.stages {
            KeyStages::Compute { entry, .. } => {
                return Created::Compute(create_compute(
                    device,
                    &wgpu::ComputePipelineDescriptor {
                        label: Some(self.label.as_str()),
                        layout: Some(layout),
                        module: &modules.first.module,
                        entry_point: Some(entry.as_str()),
                        compilation_options: Default::default(),
                        cache: None,
                    },
                ));
            }
            KeyStages::Render(render) => render,
        };
        let vertex_constants = named(&modules.first.constants);
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
        let fragment_constants = modules
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
                    module: &modules.first.module,
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
                fragment: render.fragment.as_ref().zip(modules.fragment.as_ref()).map(
                    |(fragment, linked)| wgpu::FragmentState {
                        module: &linked.module,
                        entry_point: Some(fragment.entry.as_str()),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &fragment_constants,
                            ..Default::default()
                        },
                        targets: &fragment.targets,
                    },
                ),
                multiview_mask: None,
                cache: None,
            },
        ))
    }

    fn creatable(&self, device: &wgpu::Device, declared: &PipelineLayout) -> bool {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum GroupKey {
    Own(&'static str),
    Shared(&'static str, &'static [&'static str]),
}

impl GroupKey {
    fn of(group: GroupRef) -> Self {
        match group {
            GroupRef::Own(table) => GroupKey::Own(table.label),
            GroupRef::Shared { group, without } => GroupKey::Shared(group, without),
        }
    }
}

struct Registry {
    device: wgpu::Device,
    canvas: Option<wgpu::TextureFormat>,
    linker: Linker,
    shared: SharedGroups,
    layouts_listed: Vec<Layouts>,
    declared: BTreeMap<&'static str, PipelineLayout>,
    resolved: BTreeMap<&'static str, Result<Arc<ResolvedLayout>, Error>>,
    groups: BTreeMap<GroupKey, GroupLayout>,
    layouts: BTreeMap<&'static str, wgpu::PipelineLayout>,
    labels: HashMap<wgpu::PipelineLayout, &'static str>,
    modules: BTreeMap<u64, wgpu::ShaderModule>,
    unlinked: Option<wgpu::ShaderModule>,
    render: BTreeMap<String, wgpu::RenderPipeline>,
    compute: BTreeMap<String, wgpu::ComputePipeline>,
    listed: Option<BTreeSet<String>>,
    up_front: u32,
    unlisted: u32,
    errors: Vec<Error>,
}

impl Registry {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            device: device.clone(),
            canvas: None,
            linker: Linker::new(crate::shaders::MODULES),
            shared: SharedGroups::default(),
            layouts_listed: vec![PIPELINE_LAYOUTS],
            declared: BTreeMap::new(),
            resolved: BTreeMap::new(),
            groups: BTreeMap::new(),
            layouts: BTreeMap::new(),
            labels: HashMap::default(),
            modules: BTreeMap::new(),
            unlinked: None,
            render: BTreeMap::new(),
            compute: BTreeMap::new(),
            listed: None,
            up_front: 0,
            unlisted: 0,
            errors: Vec::new(),
        }
    }

    fn label(&self, layout: &wgpu::PipelineLayout) -> &'static str {
        self.labels
            .get(layout)
            .copied()
            .expect("layout_from_pipeline_layout")
    }

    fn declared(&self, label: &str) -> Option<PipelineLayout> {
        self.declared.get(label).copied().or_else(|| {
            self.layouts_listed
                .iter()
                .flat_map(|layouts| layouts.iter())
                .find(|(_, declared)| declared.label == label)
                .map(|&(_, declared)| *declared)
        })
    }

    fn resolved(&mut self, label: &str) -> Result<Arc<ResolvedLayout>, Error> {
        if let Some(resolved) = self.resolved.get(label) {
            return resolved.clone();
        }
        let Some(declared) = self.declared(label) else {
            return Err(failed("$.layouts").expected(Expected::Present).got(label));
        };
        let resolved = ResolvedLayout::resolve(declared.label, declared.groups, &self.shared)
            .and_then(|resolved| resolved.check(&declared.limits).map(|()| resolved))
            .map(Arc::new);
        if let Err(error) = &resolved {
            self.errors.push(error.clone());
        }
        self.resolved.insert(declared.label, resolved.clone());
        resolved
    }

    fn group_layout(&mut self, group: GroupRef) -> GroupLayout {
        let key = GroupKey::of(group);
        if let Some(found) = self.groups.get(&key) {
            return found.clone();
        }
        let made = match group {
            GroupRef::Own(table) => layout(&self.device, table),
            GroupRef::Shared { group, without } => {
                let bindings = self
                    .shared
                    .view(group, without)
                    .map(|binding| ResolvedBinding {
                        group: 0,
                        number: binding.number,
                        name: binding.decl.name,
                        kind: binding.decl.kind,
                        stages: binding.decl.stages,
                        owner: binding.brick,
                    })
                    .collect();
                group_layout(
                    &self.device,
                    key,
                    &format!("engine/{group}-group"),
                    bindings,
                )
            }
        };
        self.groups.insert(key, made.clone());
        made
    }

    fn layout_of(&mut self, declared: &PipelineLayout) -> wgpu::PipelineLayout {
        self.declared.insert(declared.label, *declared);
        if let Some(layout) = self.layouts.get(declared.label) {
            return layout.clone();
        }
        let _ = self.resolved(declared.label);
        let groups: Vec<GroupLayout> = declared
            .groups
            .iter()
            .map(|&group| self.group_layout(group))
            .collect();
        let layouts: Vec<Option<&wgpu::BindGroupLayout>> =
            groups.iter().map(|group| Some(&group.layout)).collect();
        let layout = create_layout(&self.device, declared.label, &layouts);
        self.labels.insert(layout.clone(), declared.label);
        self.layouts.insert(declared.label, layout.clone());
        layout
    }

    fn shader_module(&mut self, composed: &Composed) -> wgpu::ShaderModule {
        let device = &self.device;
        self.modules
            .entry(composed.hash)
            .or_insert_with(|| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("{}::{}", composed.key.module, composed.key.entry)),
                    source: wgpu::ShaderSource::Wgsl(composed.text.as_str().into()),
                })
            })
            .clone()
    }

    fn unlinked(&mut self) -> wgpu::ShaderModule {
        let device = &self.device;
        self.unlinked
            .get_or_insert_with(|| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("engine/unlinked"),
                    source: wgpu::ShaderSource::Wgsl("".into()),
                })
            })
            .clone()
    }

    fn try_linked(
        &mut self,
        layout: &str,
        module: &str,
        entry: &str,
        constants: &[(String, f64)],
    ) -> Result<Linked, Error> {
        let resolved = self.resolved(layout)?;
        let composed = self.linker.compose_named(module, entry, &resolved)?;
        let values = named(constants);
        let constants = composed
            .constants(&values)
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .collect();
        Ok(Linked {
            module: self.shader_module(&composed),
            constants,
        })
    }

    fn try_modules(&mut self, key: &Key) -> Result<Modules, Error> {
        match &key.stages {
            KeyStages::Compute { module, entry } => Ok(Modules {
                first: self.try_linked(&key.layout, module, entry, &[])?,
                fragment: None,
            }),
            KeyStages::Render(render) => {
                let first = self.try_linked(
                    &key.layout,
                    &render.vertex_module,
                    &render.vertex,
                    &render.vertex_constants,
                )?;
                let fragment = match &render.fragment {
                    Some(fragment) => Some(self.try_linked(
                        &key.layout,
                        &fragment.module,
                        &fragment.entry,
                        &fragment.constants,
                    )?),
                    None => None,
                };
                Ok(Modules { first, fragment })
            }
        }
    }

    fn modules_for(&mut self, key: &Key) -> Modules {
        match self.try_modules(key) {
            Ok(modules) => modules,
            Err(error) => {
                if !self.errors.contains(&error) {
                    self.errors.push(error);
                }
                let unlinked = Linked {
                    module: self.unlinked(),
                    constants: Vec::new(),
                };
                let fragment = match &key.stages {
                    KeyStages::Render(render) if render.fragment.is_some() => {
                        Some(unlinked.clone())
                    }
                    KeyStages::Render(_) | KeyStages::Compute { .. } => None,
                };
                Modules {
                    first: unlinked,
                    fragment,
                }
            }
        }
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

    fn forget_pipelines(&mut self) {
        self.render.clear();
        self.compute.clear();
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

pub fn install(
    device: &wgpu::Device,
    shared: &SharedGroups,
    layouts: impl IntoIterator<Item = Layouts>,
) {
    with_registry(device, |registry| {
        let mut listed = vec![PIPELINE_LAYOUTS];
        for layouts in layouts {
            if !listed.iter().any(|known| std::ptr::eq(*known, layouts)) {
                listed.push(layouts);
            }
        }
        registry.layouts_listed = listed;
        if registry.shared != *shared {
            registry.shared = shared.clone();
            registry.resolved.clear();
            registry.groups.clear();
            registry.layouts.clear();
            registry.forget_pipelines();
        }
    });
}

pub fn link(device: &wgpu::Device, linker: Linker) {
    with_registry(device, |registry| {
        if registry.linker.sources_hash() != linker.sources_hash() {
            registry.linker = linker;
            registry.forget_pipelines();
        }
    });
}

pub fn add_generated(device: &wgpu::Device, path: String, brick: &'static str, source: String) {
    with_registry(device, |registry| {
        let named = format!("\"{path}\"");
        let imported = registry.linker.imports(&path);
        if registry.linker.add_generated(path, brick, source) {
            if imported {
                registry.forget_pipelines();
            } else {
                registry.render.retain(|text, _| !text.contains(&named));
                registry.compute.retain(|text, _| !text.contains(&named));
            }
        }
    });
}

pub fn layout_of(device: &wgpu::Device, declared: &PipelineLayout) -> wgpu::PipelineLayout {
    with_registry(device, |registry| registry.layout_of(declared))
}

pub fn shared_layout(
    device: &wgpu::Device,
    group: &'static str,
    without: &'static [&'static str],
) -> GroupLayout {
    with_registry(device, |registry| {
        registry.group_layout(GroupRef::Shared { group, without })
    })
}

const LIST: &[u8; 4] = b"PIPL";

type Listed = (Option<wgpu::TextureFormat>, Vec<Key>, Vec<Composed>);

fn read_list(record: &[u8]) -> Result<Listed, Error> {
    let broken = |expected: Expected| failed("$.pipelines").expected(expected);
    let rest = record
        .strip_prefix(LIST)
        .ok_or_else(|| broken(Expected::Matching))?;
    let (count, mut rest) = rest
        .split_first_chunk::<4>()
        .ok_or_else(|| broken(Expected::Present))?;
    let count = u32::from_le_bytes(*count) as usize;
    let mut texts = Vec::new();
    while let Some((length, after)) = rest.split_first_chunk::<4>() {
        let length = u32::from_le_bytes(*length) as usize;
        let text = after
            .get(..length)
            .ok_or_else(|| broken(Expected::Present))?;
        texts.push(std::str::from_utf8(text).map_err(|_| broken(Expected::Utf8))?);
        rest = &after[length..];
    }
    let Some((canvas, entries)) = texts
        .split_first()
        .filter(|(_, entries)| rest.is_empty() && entries.len() >= count)
    else {
        return Err(broken(Expected::Present));
    };
    let (keys, composed) = entries.split_at(count);
    let unreadable = |text: &str| broken(Expected::Json).got(text);
    let canvas = match *canvas {
        "" => None,
        canvas => Some(serde_json::from_str(canvas).map_err(|_| unreadable(canvas))?),
    };
    let keys = keys
        .iter()
        .map(|text| serde_json::from_str(text).map_err(|_| unreadable(text)))
        .collect::<Result<_, Error>>()?;
    let composed = composed
        .iter()
        .map(|text| serde_json::from_str(text).map_err(|_| unreadable(text)))
        .collect::<Result<_, Error>>()?;
    Ok((canvas, keys, composed))
}

type Wanted = (String, Key, wgpu::PipelineLayout, Modules);

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
                        .filter_map(|(text, key, layout, modules)| {
                            let errors = crate::gpu_calls::validation_scope(device);
                            let created = key.create(device, layout, modules);
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
            .map(|part| part.join().expect("pipeline_creation_unpanicked"))
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
        .map(|(text, key, layout, modules)| (text.clone(), key.create(device, layout, modules)))
        .collect();
    wasm_bindgen_futures::spawn_local(async move {
        if let Some(error) = errors.pop().await {
            let record = serde_json::json!({
                "event": "listed_pipeline_failed",
                "error": error.to_string(),
            });
            web_sys::console::warn_1(&record.to_string().into());
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
                .map(|canvas| serde_json::to_string(&canvas).expect("formats_serialize"))
                .unwrap_or_default();
            let keys = registry.keys();
            let composed: Vec<String> = registry
                .linker
                .composed()
                .map(|composed| serde_json::to_string(composed).expect("composed_serializes"))
                .collect();
            let mut record = LIST.to_vec();
            record.extend((keys.len() as u32).to_le_bytes());
            for text in std::iter::once(&canvas).chain(keys).chain(composed.iter()) {
                record.extend((text.len() as u32).to_le_bytes());
                record.extend(text.as_bytes());
            }
            record
        })
    }

    pub(crate) fn create_listed(&self, record: &[u8]) -> Result<u32, Error> {
        let (listed_canvas, keys, composed) = read_list(record)?;
        let device = &self.device;
        with_registry(device, |registry| {
            registry.linker.preload(composed);
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
                if registry.render.contains_key(&text) || registry.compute.contains_key(&text) {
                    continue;
                }
                let Some(declared) = registry.declared(&key.layout) else {
                    continue;
                };
                if !key.creatable(device, &declared) {
                    continue;
                }
                let layout = registry.layout_of(&declared);
                let Ok(modules) = registry.try_modules(&key) else {
                    continue;
                };
                wanted.push((text, key, layout, modules));
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

    pub(crate) fn take_errors(&self) -> Vec<Error> {
        with_registry(&self.device, |registry| {
            std::mem::take(&mut registry.errors)
        })
    }

    pub(crate) fn frame_submitted(&mut self, queue: &crate::gpu_calls::Queue) -> u32 {
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

fn group_layout(
    device: &wgpu::Device,
    key: GroupKey,
    label: &str,
    bindings: Vec<ResolvedBinding>,
) -> GroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
        .iter()
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding: binding.number,
            visibility: shader_stages(binding.stages),
            ty: binding_type(binding.kind),
            count: None,
        })
        .collect();
    GroupLayout {
        key,
        bindings,
        layout: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        }),
    }
}

pub fn layout(device: &wgpu::Device, table: &'static Table) -> GroupLayout {
    let bindings = table
        .bindings
        .iter()
        .zip(0..)
        .map(|(binding, number)| ResolvedBinding {
            group: 0,
            number,
            name: binding.name,
            kind: binding.kind,
            stages: binding.stages,
            owner: table.label,
        })
        .collect();
    group_layout(device, GroupKey::Own(table.label), table.label, bindings)
}

#[derive(Clone, Debug)]
pub struct GroupLayout {
    key: GroupKey,
    bindings: Vec<ResolvedBinding>,
    layout: wgpu::BindGroupLayout,
}

impl GroupLayout {
    fn number(&self, name: &str) -> u32 {
        self.bindings
            .iter()
            .find(|binding| binding.name == name)
            .map(|binding| binding.number)
            .unwrap_or_else(|| panic!("binding_unknown {:?} {name}", self.key))
    }
}

impl std::ops::Deref for GroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }
}

pub fn pipeline_layout(
    device: &wgpu::Device,
    declared: &PipelineLayout,
    groups: &[&GroupLayout],
) -> wgpu::PipelineLayout {
    assert!(
        groups.len() == declared.groups.len()
            && groups
                .iter()
                .zip(declared.groups)
                .all(|(group, &declared)| group.key == GroupKey::of(declared)),
        "groups_as_declared {}",
        declared.label
    );
    layout_of(device, declared)
}

pub struct Family {
    group: GroupLayout,
    pub pipeline: wgpu::PipelineLayout,
}

impl Family {
    pub fn new(device: &wgpu::Device, declared: &PipelineLayout, more: &[&GroupLayout]) -> Self {
        let group = match declared.groups[0] {
            GroupRef::Own(table) => layout(device, table),
            GroupRef::Shared { group, without } => shared_layout(device, group, without),
        };
        let groups: Vec<&GroupLayout> = std::iter::once(&group)
            .chain(more.iter().copied())
            .collect();
        let pipeline = pipeline_layout(device, declared, &groups);
        Self { group, pipeline }
    }

    pub fn group(
        &self,
        device: &wgpu::Device,
        label: &str,
        resources: &[(&str, wgpu::BindingResource<'_>)],
        fillers: &Fillers,
    ) -> wgpu::BindGroup {
        fill(device, label, &self.group, resources, fillers)
    }
}

pub fn view(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

pub fn sampler(sampler: &wgpu::Sampler) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Sampler(sampler)
}

const WRITE_FILLERS: u64 = 32;
const FILLER_BYTES: u64 = 256;

const UNIFORM_FILLER_BYTES: u64 = 4096;

pub struct Fillers {
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
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = |label, size, usage| {
            crate::memory::buffer(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                },
            )
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
                panic!("filler_missing {kind:?}")
            }
        }
    }
}

pub fn group(
    device: &wgpu::Device,
    label: &str,
    table: &Table,
    layout: &GroupLayout,
    resources: &[(&str, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    debug_assert_eq!(
        layout.key,
        GroupKey::Own(table.label),
        "layout_matches {label}"
    );
    debug_assert_eq!(
        resources.len(),
        table.bindings.len(),
        "binds_every_binding {label} {}",
        table.label
    );
    let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
        .iter()
        .map(|(name, resource)| wgpu::BindGroupEntry {
            binding: layout.number(name),
            resource: resource.clone(),
        })
        .collect();
    bind_group(device, label, layout, &entries)
}

pub fn filled_group(
    device: &wgpu::Device,
    label: &str,
    table: &Table,
    layout: &GroupLayout,
    resources: &[(&str, wgpu::BindingResource<'_>)],
    fillers: &Fillers,
) -> wgpu::BindGroup {
    debug_assert_eq!(
        layout.key,
        GroupKey::Own(table.label),
        "layout_matches {label}"
    );
    fill(device, label, layout, resources, fillers)
}

fn fill(
    device: &wgpu::Device,
    label: &str,
    layout: &GroupLayout,
    resources: &[(&str, wgpu::BindingResource<'_>)],
    fillers: &Fillers,
) -> wgpu::BindGroup {
    for (name, _) in resources {
        layout.number(name);
    }
    let mut written = 0;
    let entries: Vec<wgpu::BindGroupEntry<'_>> = layout
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
                binding: binding.number,
                resource,
            }
        })
        .collect();
    debug_assert!(written <= WRITE_FILLERS, "fillers_within_ranges");
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

pub fn compute_entry(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    entry: Entry,
) -> wgpu::ComputePipeline {
    with_registry(device, |registry| {
        let key = Key {
            label: entry.name.to_owned(),
            layout: registry.label(layout).to_owned(),
            stages: KeyStages::Compute {
                module: entry.module.to_owned(),
                entry: entry.name.to_owned(),
            },
        };
        let text = key.text();
        if let Some(pipeline) = registry.compute.get(&text) {
            return pipeline.clone();
        }
        registry.creating(&text);
        let modules = registry.modules_for(&key);
        let Created::Compute(pipeline) = key.create(device, layout, &modules) else {
            unreachable!("compute_key_compute_pipeline");
        };
        registry.compute.insert(text, pipeline.clone());
        pipeline
    })
}

pub struct Geometry<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub vertex: Entry,
    pub vertex_constants: &'a [(&'a str, f64)],
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub primitive: wgpu::PrimitiveState,
    pub depth: Option<wgpu::DepthStencilState>,
    pub fragment: Option<Fragment<'a>>,
}

pub struct Fragment<'a> {
    pub entry: Entry,
    pub constants: &'a [(&'a str, f64)],
    pub targets: &'a [Option<wgpu::ColorTargetState>],
}

pub fn geometry(device: &wgpu::Device, geometry: &Geometry<'_>) -> wgpu::RenderPipeline {
    sampled(device, geometry, 1)
}

pub fn sampled(
    device: &wgpu::Device,
    geometry: &Geometry<'_>,
    samples: u32,
) -> wgpu::RenderPipeline {
    with_registry(device, |registry| {
        let key = Key {
            label: geometry.label.to_owned(),
            layout: registry.label(geometry.layout).to_owned(),
            stages: KeyStages::Render(Box::new(RenderKey {
                vertex_module: geometry.vertex.module.to_owned(),
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
                    module: fragment.entry.module.to_owned(),
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
        let modules = registry.modules_for(&key);
        let Created::Render(pipeline) = key.create(device, geometry.layout, &modules) else {
            unreachable!("render_key_render_pipeline");
        };
        registry.render.insert(text, pipeline.clone());
        pipeline
    })
}

pub struct Fullscreen<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub vertex: Entry,
    pub fragment: Entry,
    pub constants: &'a [(&'a str, f64)],
    pub targets: &'a [Option<wgpu::ColorTargetState>],
    pub depth: Option<wgpu::DepthStencilState>,
}

#[macro_export]
macro_rules! vertex_attributes {
    ($ty:ident { $($location:literal => $field:ident as $format:ident),+ $(,)? }) => {
        [$({
            const FIELD: &$crate::gpu_layouts::Field = $ty::LAYOUT.field(stringify!($field));
            const _: () = assert!(::wgpu::VertexFormat::$format.size() as usize == FIELD.size);
            ::wgpu::VertexAttribute {
                format: ::wgpu::VertexFormat::$format,
                offset: FIELD.offset as u64,
                shader_location: $location,
            }
        }),+]
    };
}

pub fn fullscreen(device: &wgpu::Device, pass: &Fullscreen<'_>) -> wgpu::RenderPipeline {
    geometry(
        device,
        &Geometry {
            label: pass.label,
            layout: pass.layout,
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
