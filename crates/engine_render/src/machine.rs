use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use engine_core::{Entry, Error, ErrorCode, PassViews, TargetName};
use wgpu::ShaderStages;

use crate::graph::{
    Access as GraphAccess, BLACK, Clear, Declared as GraphPass, FAR, Life, PassEntry, PassId,
    ResourceEntry, ResourceId, StaticGraph, TRANSPARENT,
};
use crate::install::{Bound, Copied, Declared, PassDecl, PipelineDecl, Program, Scoped, Work};
use crate::linker::{Composed, Linker, Ty};
use crate::{Access, DepthUse, Lifetime, SharedTarget, VertexLayout};

mod composing;

pub(crate) use composing::{Composing, compose};

pub(crate) const CAMERA_STRIDE: u64 = 512;
const TARGET_KEY: u64 = 1 << 62;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    Shared(TargetName),
    Canvas,
}

pub(crate) fn final_target(name: TargetName) -> Target {
    if name == TargetName::CANVAS {
        Target::Canvas
    } else {
        Target::Shared(name)
    }
}

impl Declared {
    pub(crate) fn shared_target(&self, name: TargetName) -> Option<&SharedTarget> {
        self.targets.iter().find(|target| target.name == name)
    }

    pub(crate) fn target_format(&self, target: Target) -> Option<wgpu::TextureFormat> {
        match target {
            Target::Shared(name) => self.shared_target(name).map(|target| target.format),
            Target::Canvas => None,
        }
    }

    pub(crate) fn depth_target(&self) -> Option<Target> {
        self.targets
            .iter()
            .find(|target| target.format.is_depth_stencil_format())
            .map(|target| Target::Shared(target.name))
    }

    pub(crate) fn target_key(&self, target: Target) -> u64 {
        let (kind, index) = match target {
            Target::Shared(name) => (
                0,
                self.targets
                    .iter()
                    .position(|target| target.name == name)
                    .map_or(u64::from(u32::MAX), |at| at as u64),
            ),
            Target::Canvas => (2, 0),
        };
        TARGET_KEY | kind << 32 | index
    }

    pub(crate) fn pass_set(&self, pass: &PassDecl) -> Vec<SetFormat> {
        let Some(scene) = pass.scene.as_ref().filter(|scene| scene.core) else {
            return Vec::new();
        };
        scene
            .pass
            .writes
            .targets()
            .iter()
            .filter_map(|name| self.shared_target(*name))
            .filter(|target| !target.format.is_depth_stencil_format())
            .map(|target| {
                (
                    Target::Shared(target.name),
                    target.format,
                    Clear::Color(BLACK),
                )
            })
            .collect()
    }
}

fn unknown_targets(declared: &Declared) -> Vec<Error> {
    let mut problems = Vec::new();
    let named =
        |name: TargetName| name == TargetName::CANVAS || declared.shared_target(name).is_some();
    let pipelines = declared
        .pipelines
        .iter()
        .flat_map(|decl| decl.bound.iter().map(|(_, bound)| bound));
    let bound = declared.frame.iter().map(|frame| &frame.bound).chain(pipelines);
    for bound in bound {
        if let Bound::Target(name) = bound
            && declared.shared_target(*name).is_none()
        {
            let path = format!("$.targets.{}", name.name());
            problems.push(Error::new(ErrorCode::UnknownVariable, path).got(name.name()));
        }
    }
    for pass in &declared.passes {
        if let Work::Copy { from, to } = pass.work {
            for copied in [from, to] {
                if let Copied::Target(name) = copied
                    && !named(name)
                {
                    let path = format!("$.targets.{}", name.name());
                    problems.push(Error::new(ErrorCode::UnknownVariable, path).got(name.name()));
                }
            }
        }
        for name in pass.scene.iter().flat_map(|scene| scene.pass.writes.targets()) {
            if declared.shared_target(*name).is_none() {
                let path = format!("$.passes.{}.writes.{}", pass.name, name.name());
                problems.push(Error::new(ErrorCode::UnknownVariable, path).got(name.name()));
            }
        }
        for &(name, _) in &pass.targets {
            if !named(name) || declared.depth_target() == Some(Target::Shared(name)) {
                let path = format!("$.passes.{}.targets.{}", pass.name, name.name());
                problems.push(Error::new(ErrorCode::PassMismatch, path).got(name.name()));
            }
        }
    }
    problems
}

pub(crate) struct Ids {
    pub(crate) textures: Vec<ResourceId>,
    pub(crate) buffers: Vec<ResourceId>,
    pub(crate) targets: Vec<(Target, ResourceId)>,
}

impl Ids {
    pub(crate) fn target(&self, target: Target) -> Option<ResourceId> {
        self.targets
            .iter()
            .find(|(known, _)| *known == target)
            .map(|(_, id)| *id)
    }
}

pub(crate) struct Stage {
    pub(crate) entry: Entry,
    pub(crate) composed: Arc<Composed>,
}

impl Stage {
    pub(crate) fn text_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.composed.text.hash(&mut hasher);
        hasher.finish()
    }

    pub(crate) fn shader(
        &self,
        device: &wgpu::Device,
        shaders: &mut BTreeMap<u64, wgpu::ShaderModule>,
    ) -> wgpu::ShaderModule {
        shaders
            .entry(self.text_key())
            .or_insert_with(|| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(self.entry.name()),
                    source: wgpu::ShaderSource::Wgsl(self.composed.text.as_str().into()),
                })
            })
            .clone()
    }
}

pub(crate) struct OwnGroup {
    pub(crate) index: u32,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) entries: Vec<(u32, Bound, Ty)>,
    pub(crate) windows: usize,
}

pub(crate) struct Ready {
    pub(crate) layout: crate::encode::PassLayout,
    pub(crate) bits: Option<crate::surface::MaterialVariant>,
    pub(crate) made: wgpu::RenderPipeline,
    pub(crate) exact: bool,
}

pub(crate) struct BuiltPipeline {
    pub(crate) decl: PipelineDecl,
    pub(crate) stages: Vec<Stage>,
    pub(crate) layout: wgpu::PipelineLayout,
    pub(crate) groups: Vec<OwnGroup>,
    pub(crate) variant: u32,
    pub(crate) reads: Vec<Bound>,
    pub(crate) writes: Vec<Bound>,
    pub(crate) made: Vec<(u64, wgpu::RenderPipeline)>,
    pub(crate) computed: Vec<(u64, wgpu::ComputePipeline)>,
    pub(crate) ready: Vec<Ready>,
    pub(crate) compute_ready: Option<wgpu::ComputePipeline>,
    pub(crate) geometry: Vec<engine_core::GeometryStream>,
    pub(crate) holes: bool,
    pub(crate) material: bool,
    pub(crate) home: u32,
    pub(crate) layout_key: String,
    pub(crate) program_key: u64,
}

pub(crate) struct FrameVariant {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) entries: Vec<(u32, ShaderStages)>,
    pub(crate) camera: bool,
}

pub(crate) struct FrameSlot {
    pub(crate) ty: Ty,
    pub(crate) post: bool,
}

pub(crate) type SetFormat = (Target, wgpu::TextureFormat, Clear);

pub(crate) struct Machine {
    pub(crate) declared: Declared,
    pub(crate) graph: StaticGraph,
    pub(crate) ids: Ids,
    pub(crate) pipelines: Vec<BuiltPipeline>,
    pub(crate) shaders: BTreeMap<u64, wgpu::ShaderModule>,
    pub(crate) frame_slots: BTreeMap<u32, FrameSlot>,
    pub(crate) frame_bound: BTreeMap<u32, Bound>,
    pub(crate) variants: Vec<FrameVariant>,
    pub(crate) binding_bytes: BTreeMap<(Option<usize>, u32, u32), u64>,
    pub(crate) linker: Linker,
}

fn layout_entries(entries: &[(u32, Ty, ShaderStages)]) -> Vec<wgpu::BindGroupLayoutEntry> {
    entries
        .iter()
        .map(|&(binding, ty, visibility)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: ty.binding(true, false),
            count: None,
        })
        .collect()
}

fn stage_of(entry: Entry, linker: &Linker) -> ShaderStages {
    let (path, name) = crate::linker::path_of((entry.module(), entry.name()));
    let text = linker.text(path).unwrap_or_default();
    let Some(at) = text.find(&format!("fn {name}(")) else {
        return ShaderStages::FRAGMENT;
    };
    let before = &text[..at];
    let start = before.rfind(['}', ';']).map_or(0, |end| end + 1);
    let attributes = &before[start..];
    if attributes.contains("@compute") {
        ShaderStages::COMPUTE
    } else if attributes.contains("@vertex") {
        ShaderStages::VERTEX
    } else {
        ShaderStages::FRAGMENT
    }
}

pub(crate) fn writes_depth(depth: DepthUse) -> bool {
    match depth {
        DepthUse::None => false,
        DepthUse::Compare { write, .. } => write,
    }
}

pub(crate) fn shared(pass: &PassDecl) -> bool {
    pass.scene.is_some_and(|scene| scene.core)
}

fn written_module(declared: &Declared, decl: &PipelineDecl) -> Option<(String, String)> {
    declared
        .passes
        .iter()
        .enumerate()
        .any(|(index, pass)| decl_in_pass(decl, index as u32, pass) && lit_set(&declared.pass_set(pass)))
        .then(crate::shared::pass_module)
}

pub(crate) fn lit_set(set: &[SetFormat]) -> bool {
    !set.is_empty()
}

pub(crate) fn decl_in_pass(decl: &PipelineDecl, index: u32, pass: &PassDecl) -> bool {
    match decl.scope {
        Scoped::Pass(own) => own == index,
        Scoped::Scene(name) => scene_member(name, decl.part, pass),
        Scoped::Outside(_) => false,
    }
}

fn scene_member(name: &str, part: u32, pass: &PassDecl) -> bool {
    pass.scene.is_some_and(|scene| scene.name == name) && pass.members.contains(&part)
}

pub(crate) fn in_pass(pipeline: &BuiltPipeline, index: u32, pass: &PassDecl) -> bool {
    decl_in_pass(&pipeline.decl, index, pass)
}

fn claimed_member(decl: &PipelineDecl, pass: &PassDecl) -> bool {
    matches!(decl.scope, Scoped::Scene(_)) && !shared(pass) && decl.part != pass.part
}

fn owned_by(declared: &Declared, bound: &Bound, part: u32) -> bool {
    match bound {
        Bound::Texture(texture) | Bound::Slice { texture, .. } => declared
            .textures
            .get(*texture as usize)
            .is_some_and(|texture| texture.part == part),
        Bound::Buffer(buffer) | Bound::Window { buffer, .. } => declared
            .buffers
            .get(*buffer as usize)
            .is_some_and(|buffer| buffer.part == part),
        Bound::Sampler(_) | Bound::Target(_) | Bound::Asset => false,
    }
}

fn depth_writes_of(declared: &Declared) -> Vec<bool> {
    declared
        .passes
        .iter()
        .enumerate()
        .map(|(index, pass)| {
            declared.pipelines.iter().any(|decl| {
                decl_in_pass(decl, index as u32, pass)
                    && matches!(&decl.program, Program::Render(render) if writes_depth(render.depth))
            })
        })
        .collect()
}

fn depth_direct_of(declared: &Declared, depth_writes: &[bool]) -> Vec<bool> {
    let writes = |index: usize| depth_writes.get(index).copied().unwrap_or(true);
    let passes = &declared.passes;
    passes
        .iter()
        .enumerate()
        .map(|(index, pass)| {
            shared(pass)
                && !writes(index)
                && passes.iter().enumerate().any(|(other, writer)| {
                    other != index
                        && writes(other)
                        && shared(writer)
                        && (writer.views == PassViews::Every || writer.views == pass.views)
                })
        })
        .collect()
}

fn reads_depth_direct(direct: &[bool], pass: usize) -> bool {
    direct.get(pass).copied().unwrap_or(false)
}

const EVERY: [u32; 2] = [0, u32::MAX];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Handle {
    Texture {
        texture: u32,
        mips: [u32; 2],
        layers: [u32; 2],
    },
    Buffer(u32),
    Target(Target),
}

impl Handle {
    fn whole(texture: u32) -> Self {
        Self::Texture {
            texture,
            mips: EVERY,
            layers: EVERY,
        }
    }

    fn overlaps(self, other: Self) -> bool {
        let meet = |[start, end]: [u32; 2], [low, high]: [u32; 2]| start < high && low < end;
        match (self, other) {
            (
                Self::Texture {
                    texture,
                    mips,
                    layers,
                },
                Self::Texture {
                    texture: known,
                    mips: known_mips,
                    layers: known_layers,
                },
            ) => texture == known && meet(mips, known_mips) && meet(layers, known_layers),
            _ => self == other,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Usage {
    Attached,
    Stored,
}

fn handle_of(bound: &Bound) -> Option<Handle> {
    match bound {
        Bound::Texture(texture) => Some(Handle::whole(*texture)),
        Bound::Slice {
            texture,
            mips,
            layers,
        } => Some(Handle::Texture {
            texture: *texture,
            mips: [mips.start, mips.end],
            layers: [layers.start, layers.end],
        }),
        Bound::Buffer(buffer) | Bound::Window { buffer, .. } => Some(Handle::Buffer(*buffer)),
        Bound::Target(name) => Some(Handle::Target(final_target(*name))),
        Bound::Sampler(_) | Bound::Asset => None,
    }
}

type Reached = Vec<(String, crate::linker::Declaration, ShaderStages)>;

fn own_writes(decl: &PipelineDecl, reached: &Reached, linker: &Linker) -> Vec<Handle> {
    reached
        .iter()
        .filter(|(module, found, _)| {
            found.group > 0 && found.ty.writes() && !(found.group == 1 && linker.is_core(module))
        })
        .filter_map(|(module, found, _)| {
            decl.bound.iter().find_map(|(var, bound)| {
                let (known, name) = crate::linker::path_of((var.module(), var.name()));
                (known == module && name == found.name)
                    .then(|| handle_of(bound))
                    .flatten()
            })
        })
        .collect()
}

fn attachments(declared: &Declared, pass: &PassDecl) -> Vec<Handle> {
    let mut found = Vec::new();
    for &(texture, _) in pass.colors.iter().chain(&pass.depths) {
        found.push(Handle::whole(texture));
    }
    if shared(pass) {
        for (target, ..) in declared.pass_set(pass) {
            found.push(Handle::Target(target));
        }
        found.extend(declared.depth_target().map(Handle::Target));
    }
    for &(name, _) in &pass.targets {
        found.push(Handle::Target(final_target(name)));
    }
    found
}

impl Machine {
    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    #[expect(
        clippy::disallowed_methods,
        reason = "the machine lays out and keys every pipeline it builds"
    )]
    pub(crate) fn build(
        device: &wgpu::Device,
        composing: Composing,
        problems: &mut Vec<Error>,
    ) -> Option<Self> {
        let Composing {
            linker,
            declared,
            core_frame,
            numbers,
            frame_bound,
            material_entries,
            depth_writes,
            reaches,
            staged,
        } = composing;
        let mut unknown = unknown_targets(&declared);
        if !unknown.is_empty() {
            problems.append(&mut unknown);
            return None;
        }
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/material-group"),
            entries: &material_entries,
        });
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/empty-group"),
            entries: &[],
        });
        let owned: Vec<Vec<Handle>> = staged
            .iter()
            .map(|(decl, _, reached)| own_writes(decl, reached, &linker))
            .collect();
        let writes: Vec<Vec<(Handle, Usage)>> = staged
            .iter()
            .zip(&owned)
            .map(|((decl, _, _), own)| {
                if let Program::Compute(_) = decl.program {
                    return own.iter().map(|&handle| (handle, Usage::Stored)).collect();
                }
                let mut written = Vec::new();
                for (index, pass) in declared.passes.iter().enumerate() {
                    if !decl_in_pass(decl, index as u32, pass) {
                        continue;
                    }
                    written.extend(
                        attachments(&declared, pass)
                            .into_iter()
                            .map(|handle| (handle, Usage::Attached)),
                    );
                    for (_, own) in staged.iter().zip(&owned).filter(|((other, _, _), _)| {
                        matches!(other.program, Program::Render(_))
                            && decl_in_pass(other, index as u32, pass)
                    }) {
                        written.extend(own.iter().map(|&handle| (handle, Usage::Stored)));
                    }
                }
                written
            })
            .collect();
        let held: Vec<Option<(Handle, Ty)>> = (0..core_frame.len())
            .map(|at| {
                let ty = core_frame[at].2;
                numbers
                    .get(at)
                    .and_then(|number| frame_bound.get(number))
                    .and_then(handle_of)
                    .map(|handle| (handle, ty))
            })
            .collect();
        let conflict = |pipeline: u32, variable: u32| {
            let Some(Some((handle, ty))) = held.get(variable as usize) else {
                return false;
            };
            writes.get(pipeline as usize).is_some_and(|written| {
                written.iter().any(|(known, usage)| {
                    known.overlaps(*handle) && !(*usage == Usage::Stored && ty.writes())
                })
            })
        };
        let mut conflicts = Vec::new();
        let mut grouped: Vec<Vec<(u32, ShaderStages)>> = Vec::new();
        let of_pipeline: Vec<u32> = reaches
            .iter()
            .enumerate()
            .map(|(pipeline, (label, alone, reached))| {
                if let Some(&(variable, _)) = reached
                    .iter()
                    .find(|&&(variable, _)| conflict(pipeline as u32, variable))
                {
                    conflicts.push(Error::new(
                        ErrorCode::UsageConflict,
                        format!("{label}.{}", core_frame[variable as usize].1),
                    ));
                }
                if *alone {
                    return u32::MAX;
                }
                let same = |known: &Vec<(u32, ShaderStages)>| {
                    known.len() == reached.len()
                        && known.iter().zip(reached).all(|(known, entry)| known.0 == entry.0)
                };
                match grouped.iter().position(same) {
                    Some(at) => {
                        for (known, entry) in grouped[at].iter_mut().zip(reached) {
                            known.1 |= entry.1;
                        }
                        at as u32
                    }
                    None => {
                        grouped.push(reached.clone());
                        (grouped.len() - 1) as u32
                    }
                }
            })
            .collect();
        if !conflicts.is_empty() {
            problems.append(&mut conflicts);
            return None;
        }
        let (variants, frame_keys): (Vec<FrameVariant>, Vec<String>) = grouped
            .iter()
            .map(|variant| {
                let entries: Vec<(u32, Ty, ShaderStages)> = variant
                    .iter()
                    .map(|&(at, stages)| (numbers[at as usize], core_frame[at as usize].2, stages))
                    .collect();
                let mut layout = layout_entries(&entries);
                for entry in &mut layout {
                    if entry.binding == 0
                        && let wgpu::BindingType::Buffer {
                            has_dynamic_offset, ..
                        } = &mut entry.ty
                    {
                        *has_dynamic_offset = true;
                    }
                    if let wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable },
                        ..
                    } = &mut entry.ty
                    {
                        *filterable = filterable_bound(frame_bound.get(&entry.binding), &declared);
                    }
                    if entry.ty == wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                        && !filterable_bound(frame_bound.get(&entry.binding), &declared)
                    {
                        entry.ty =
                            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering);
                    }
                }
                let variant = FrameVariant {
                    layout: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("engine/frame-group"),
                        entries: &layout,
                    }),
                    entries: entries
                        .iter()
                        .map(|&(binding, _, stages)| (binding, stages))
                        .collect(),
                    camera: entries.iter().any(|&(binding, _, _)| binding == 0),
                };
                (variant, format!("{layout:?}"))
            })
            .unzip();
        let material_key = format!("{material_entries:?}");
        let mut pipelines = Vec::with_capacity(staged.len());
        for (index, (decl, stages, reached)) in staged.into_iter().enumerate() {
            let material = reached
                .iter()
                .any(|(module, found, _)| found.group == 1 && linker.is_core(module));
            let mut groups: Vec<OwnGroup> = Vec::new();
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            for (module, found, _) in &reached {
                if found.group == 0 {
                    if let Some(bound) = linker
                        .binding_of(module, &found.name)
                        .and_then(|number| frame_bound.get(&number))
                    {
                        reads.push(bound.clone());
                    }
                    continue;
                }
                let Some(bound) = decl.bound.iter().find_map(|(var, bound)| {
                    let (known, name) = crate::linker::path_of((var.module(), var.name()));
                    (known == module && name == found.name).then(|| bound.clone())
                }) else {
                    continue;
                };
                if found.ty.writes() {
                    writes.push(bound.clone());
                } else {
                    reads.push(bound.clone());
                }
                let entry = (found.binding, bound, found.ty);
                match groups.iter_mut().find(|group| group.index == found.group) {
                    Some(group) => group.entries.push(entry),
                    None => groups.push(OwnGroup {
                        index: found.group,
                        layout: empty.clone(),
                        entries: vec![entry],
                        windows: 0,
                    }),
                }
            }
            groups.sort_by_key(|group| group.index);
            let mut layout_key = String::new();
            for group in &mut groups {
                group.windows = group
                    .entries
                    .iter()
                    .filter(|(_, bound, _)| bound.window())
                    .count();
                let entries: Vec<wgpu::BindGroupLayoutEntry> = group
                    .entries
                    .iter()
                    .map(|(binding, bound, ty)| {
                        let visibility = reached
                            .iter()
                            .find(|(_, found, _)| {
                                found.group == group.index && found.binding == *binding
                            })
                            .map_or(
                                ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                                |(_, _, stages)| *stages,
                            );
                        wgpu::BindGroupLayoutEntry {
                            binding: *binding,
                            visibility,
                            ty: ty
                                .binding(filterable_bound(Some(bound), &declared), bound.window()),
                            count: None,
                        }
                    })
                    .collect();
                layout_key.push_str(&format!("{}:{entries:?};", group.index));
                group.layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("engine/part-group"),
                    entries: &entries,
                });
            }
            let variant = of_pipeline.get(index).copied().unwrap_or(0);
            let frame_key = frame_keys.get(variant as usize).map_or("", String::as_str);
            let material_shown = if material { material_key.as_str() } else { "" };
            layout_key.push_str(&format!(
                "frame {variant} {frame_key} material {material_shown}"
            ));
            let top = groups
                .last()
                .map_or(0, |group| group.index)
                .max(u32::from(material));
            let mut layouts: Vec<&wgpu::BindGroupLayout> = Vec::new();
            let frame_layout = variants
                .get(variant as usize)
                .map_or(&empty, |found| &found.layout);
            layouts.push(frame_layout);
            for at in 1..=top {
                let layout = match groups.iter().find(|group| group.index == at) {
                    Some(group) => &group.layout,
                    None if material && at == 1 => &material_layout,
                    None => &empty,
                };
                layouts.push(layout);
            }
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("engine/part-pipeline"),
                bind_group_layouts: &layouts
                    .iter()
                    .map(|layout| Some(*layout))
                    .collect::<Vec<_>>(),
                immediate_size: 0,
            });
            let mut hasher = DefaultHasher::new();
            format!("{:?} {:?}", decl.program, decl.constants).hash(&mut hasher);
            for stage in &stages {
                stage.entry.name().hash(&mut hasher);
                stage.composed.text.hash(&mut hasher);
            }
            layout_key.hash(&mut hasher);
            let program_key = hasher.finish();
            let geometry = match &decl.program {
                Program::Render(desc) => desc
                    .vertices
                    .iter()
                    .filter_map(|layout| match layout {
                        VertexLayout::Geometry(stream) => Some(*stream),
                        VertexLayout::Buffer(_) => None,
                    })
                    .collect(),
                Program::Compute(_) => Vec::new(),
            };
            let holes = decl.holes() > 0;
            let home = declared
                .passes
                .iter()
                .enumerate()
                .find(|(at, pass)| decl_in_pass(&decl, *at as u32, pass))
                .map_or(u32::MAX, |(at, _)| at as u32);
            let material = material && groups.iter().all(|group| group.index != 1);
            pipelines.push(BuiltPipeline {
                decl,
                stages,
                layout,
                groups,
                variant,
                reads,
                writes,
                made: Vec::new(),
                computed: Vec::new(),
                ready: Vec::new(),
                compute_ready: None,
                geometry,
                holes,
                material,
                home,
                layout_key,
                program_key,
            });
        }
        let direct_depth = depth_direct_of(&declared, &depth_writes);
        let (graph, ids) = match graph_of(&declared, &pipelines, &direct_depth) {
            Ok(built) => built,
            Err(error) => {
                problems.push(error);
                return None;
            }
        };
        let frame_slots = core_frame
            .iter()
            .filter_map(|(path, name, ty)| {
                let number = linker.binding_of(path, name)?;
                let post = linker.is_core(path) && name == crate::shared::POST;
                Some((number, FrameSlot { ty: *ty, post }))
            })
            .collect();
        let binding_bytes = binding_bytes_of(&pipelines);
        Some(Self {
            declared,
            graph,
            ids,
            pipelines,
            shaders: BTreeMap::new(),
            frame_slots,
            frame_bound,
            variants,
            binding_bytes,
            linker,
        })
    }
}

fn binding_bytes_of(pipelines: &[BuiltPipeline]) -> BTreeMap<(Option<usize>, u32, u32), u64> {
    let mut known = BTreeMap::new();
    let mut sized: engine_core::collections::HashMap<
        *const crate::linker::Composed,
        Vec<(u32, u32, u64)>,
    > = engine_core::collections::HashMap::default();
    for (at, pipeline) in pipelines.iter().enumerate() {
        for stage in &pipeline.stages {
            let sizes = sized
                .entry(Arc::as_ptr(&stage.composed))
                .or_insert_with(|| crate::sizes::binding_sizes(&stage.composed.text));
            for &(group, binding, bytes) in sizes.iter() {
                known.entry((Some(at), group, binding)).or_insert(bytes);
                if group == 0 {
                    known.entry((None, group, binding)).or_insert(bytes);
                }
            }
        }
    }
    known
}

fn filterable_bound(bound: Option<&Bound>, declared: &Declared) -> bool {
    if let Some(Bound::Sampler(sampler)) = bound {
        return declared.samplers.get(*sampler as usize).is_none_or(|desc| {
            desc.mag_filter == engine_core::Filter::Linear
                || desc.min_filter == engine_core::Filter::Linear
                || desc.mipmap_filter == Some(engine_core::Filter::Linear)
        });
    }
    let format = match bound {
        Some(Bound::Texture(texture) | Bound::Slice { texture, .. }) => declared
            .textures
            .get(*texture as usize)
            .map(|texture| texture.format),
        Some(Bound::Target(name)) => declared.target_format(final_target(*name)),
        Some(_) | None => None,
    };
    format.is_none_or(|format| {
        format
            .sample_type(None, None)
            .is_some_and(|sample| sample == wgpu::TextureSampleType::Float { filterable: true })
    })
}

fn life(lifetime: Lifetime) -> Life {
    lifetime.into()
}

fn graph_access(access: Access, clear: Clear) -> GraphAccess {
    match access {
        Access::Load => GraphAccess::Load,
        Access::Clear => GraphAccess::Clear(clear),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn graph_of(
    declared: &Declared,
    pipelines: &[BuiltPipeline],
    direct: &[bool],
) -> Result<(StaticGraph, Ids), Error> {
    let mut resources = Vec::new();
    let mut ids = Ids {
        textures: Vec::new(),
        buffers: Vec::new(),
        targets: Vec::new(),
    };
    for texture in &declared.textures {
        ids.textures.push(ResourceId::new(resources.len()));
        resources.push(ResourceEntry {
            label: texture.label,
            life: life(texture.lifetime),
            view: false,
        });
    }
    for buffer in &declared.buffers {
        ids.buffers.push(ResourceId::new(resources.len()));
        resources.push(ResourceEntry {
            label: buffer.label,
            life: life(buffer.lifetime),
            view: false,
        });
    }
    let mut target =
        |target: Target, label: &'static str, view: bool, life: Life, ids: &mut Ids| {
            ids.targets.push((target, ResourceId::new(resources.len())));
            resources.push(ResourceEntry { label, life, view });
        };
    for shared in &declared.targets {
        let name = shared.name;
        target(Target::Shared(name), name.name(), true, Life::Output, &mut ids);
    }
    target(Target::Canvas, "canvas", true, Life::Output, &mut ids);
    let depth_target = declared.depth_target();
    let mut passes = Vec::with_capacity(declared.passes.len());
    for (index, pass) in declared.passes.iter().enumerate() {
        let part = declared.parts.get(pass.part as usize);
        let entry = PassEntry {
            label: pass.name,
            brick: part.map_or("", |part| part.package),
            registered: index as u32,
            shared: shared(pass),
            group: pass.group,
        };
        let direct_depth = reads_depth_direct(direct, index);
        let set = declared.pass_set(pass);
        let attached = |target: Target| {
            entry.shared
                && (set.iter().any(|(known, ..)| *known == target)
                    || (Some(target) == depth_target && !direct_depth))
        };
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        let mut colors = Vec::new();
        let mut depth = None;
        let bound_id = |bound: &Bound| -> Option<ResourceId> {
            match bound {
                Bound::Texture(texture) | Bound::Slice { texture, .. } => {
                    ids.textures.get(*texture as usize).copied()
                }
                Bound::Buffer(buffer) | Bound::Window { buffer, .. } => {
                    ids.buffers.get(*buffer as usize).copied()
                }
                Bound::Target(name) => ids.target(final_target(*name)),
                Bound::Sampler(_) | Bound::Asset => None,
            }
        };
        let mut current = Vec::new();
        for pipeline in pipelines
            .iter()
            .filter(|pipeline| in_pass(pipeline, index as u32, pass))
        {
            for bound in &pipeline.reads {
                if let Bound::Target(name) = bound
                    && attached(final_target(*name))
                {
                    let path = format!("$.passes.{}.reads.{}", pass.name, name.name());
                    return Err(Error::new(ErrorCode::PassMismatch, path).got(name.name()));
                }
            }
            reads.extend(pipeline.reads.iter().filter_map(&bound_id));
            writes.extend(pipeline.writes.iter().filter_map(&bound_id));
            if claimed_member(&pipeline.decl, pass) {
                current.extend(
                    pipeline
                        .reads
                        .iter()
                        .filter(|bound| owned_by(declared, bound, pipeline.decl.part))
                        .filter_map(&bound_id),
                );
            }
        }
        for input in &declared.inputs {
            let (scoped, member) = match input.scope {
                Scoped::Pass(own) => (own == index as u32, false),
                Scoped::Outside(_) => (false, false),
                Scoped::Scene(name) => (
                    scene_member(name, input.part, pass),
                    !shared(pass) && input.part != pass.part,
                ),
            };
            if scoped && let Some(id) = ids.buffers.get(input.buffer as usize) {
                reads.push(*id);
                if member
                    && declared
                        .buffers
                        .get(input.buffer as usize)
                        .is_some_and(|buffer| buffer.part == input.part)
                {
                    current.push(*id);
                }
            }
        }
        match pass.work {
            Work::Copy { from, to } => {
                let id = |copied: Copied| match copied {
                    Copied::Buffer(buffer) => ids.buffers.get(buffer as usize).copied(),
                    Copied::Texture(texture) => ids.textures.get(texture as usize).copied(),
                    Copied::Target(name) => ids.target(final_target(name)),
                    Copied::Canvas => ids.target(Target::Canvas),
                };
                reads.extend(id(from));
                writes.extend(id(to));
            }
            Work::Clear(buffer) => writes.extend(ids.buffers.get(buffer as usize).copied()),
            Work::Render => {}
        }
        if entry.shared {
            for (target, _, clear) in set {
                colors.extend(ids.target(target).map(|id| (id, GraphAccess::Draw(clear))));
            }
            if let Some(id) = depth_target.and_then(|target| ids.target(target)) {
                let access = if direct_depth {
                    GraphAccess::ReadOnly
                } else {
                    GraphAccess::Draw(Clear::Depth(FAR))
                };
                depth = Some((id, access));
            }
        }
        for &(name, access) in &pass.targets {
            if let Some(id) = ids.target(final_target(name)) {
                colors.push((id, graph_access(access, Clear::Color(BLACK))));
            }
        }
        for &(texture, access) in &pass.colors {
            if let Some(id) = ids.textures.get(texture as usize) {
                let clear = pass.clear_of(texture).unwrap_or(Clear::Color(TRANSPARENT));
                colors.push((*id, graph_access(access, clear)));
            }
        }
        for &(texture, access) in &pass.depths {
            if let Some(id) = ids.textures.get(texture as usize) {
                let clear = pass.clear_of(texture).unwrap_or(Clear::Depth(FAR));
                match depth {
                    None => depth = Some((*id, graph_access(access, clear))),
                    Some(_) => writes.push(*id),
                }
            }
        }
        passes.push(GraphPass {
            entry,
            reads,
            writes,
            colors,
            depth,
            after: pass
                .after
                .iter()
                .map(|after| PassId::new(*after as usize))
                .collect(),
            current,
        });
    }
    let graph = StaticGraph::build(resources, passes)?;
    Ok((graph, ids))
}
