use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use engine_core::{AttachPoint, Entry, Error, ErrorCode, PassViews};
use wgpu::ShaderStages;

use crate::frame_group::{Class, FrameVariable, Reach};
use crate::graph::{
    Access as GraphAccess, BLACK, Clear, Declared as GraphPass, FAR, Life, PassEntry, PassId,
    ResourceEntry, ResourceId, StaticGraph, TRANSPARENT,
};
use crate::install::{
    Bound, Copied, Declared, PassDecl, PipelineDecl, Program, Scoped, ViewTarget, Work,
};
use crate::linker::{Composed, Linker, Ty};
use crate::{Access, DepthUse, Lifetime, VertexLayout};

mod composing;

pub(crate) use composing::{Composing, compose};

pub const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub(crate) const CAMERA_STRIDE: u64 = 512;

#[derive(Clone, Debug)]
pub(crate) struct OutputDecl {
    pub(crate) name: &'static str,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) layers: Vec<AttachPoint>,
    pub(crate) module: Option<String>,
    pub(crate) last_frame: crate::LastFrame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    Scene,
    Depth,
    Output(u32),
    Canvas,
    Copy(AttachPoint, ViewTarget),
}

pub(crate) fn final_target(view: ViewTarget) -> Target {
    match view {
        ViewTarget::Scene => Target::Scene,
        ViewTarget::Depth => Target::Depth,
        ViewTarget::Output(index) => Target::Output(index),
    }
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
    pub(crate) view_reads: Vec<Target>,
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
    pub(crate) skipped: bool,
}

pub(crate) fn layout_key(pipeline: &BuiltPipeline) -> String {
    pipeline.layout_key.clone()
}

pub(crate) struct FrameVariant {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) entries: Vec<(u32, ShaderStages)>,
    pub(crate) camera: bool,
}

pub(crate) enum CoreFrame {
    Post,
    Linear,
    Target(Target),
}

pub(crate) struct FrameSlot {
    pub(crate) ty: Ty,
    pub(crate) core: Option<CoreFrame>,
}

pub(crate) type SetFormat = (Target, wgpu::TextureFormat, Clear);

pub(crate) struct Machine {
    pub(crate) declared: Declared,
    pub(crate) outputs: Vec<OutputDecl>,
    pub(crate) graph: StaticGraph,
    pub(crate) ids: Ids,
    pub(crate) pipelines: Vec<BuiltPipeline>,
    pub(crate) shaders: BTreeMap<u64, wgpu::ShaderModule>,
    pub(crate) frame_slots: BTreeMap<u32, FrameSlot>,
    pub(crate) frame_bound: BTreeMap<u32, Bound>,
    pub(crate) variants: Vec<FrameVariant>,
    pub(crate) switches: Vec<(u32, engine_registry::GateId)>,
    pub(crate) view_copies: Vec<(AttachPoint, ViewTarget)>,
    pub(crate) doubled: Vec<bool>,
    pub(crate) direct_depth: Vec<bool>,
    pub(crate) binding_bytes: BTreeMap<(Option<usize>, u32, u32), u64>,
    pub(crate) linker: Linker,
    pub(crate) over_limit: Vec<(u32, Error)>,
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

fn class(ty: Ty, dynamic: bool) -> Class {
    match ty {
        Ty::Uniform => Class::Uniform { dynamic },
        Ty::Storage { .. } => Class::Storage,
        Ty::Texture { .. } => Class::Texture,
        Ty::StorageTexture { .. } => Class::StorageTexture,
        Ty::Sampler | Ty::Comparison => Class::Sampler,
    }
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
        DepthUse::None | DepthUse::Test => false,
        DepthUse::Write { .. } | DepthUse::Overwrite => true,
        DepthUse::Compare { write, .. } => write,
    }
}

pub(crate) fn pass_layer(pass: &PassDecl) -> Option<AttachPoint> {
    pass.layer.filter(|point| point.scene())
}

pub(crate) fn shared(pass: &PassDecl) -> bool {
    pass.scene.is_some_and(|scene| scene.core)
}

pub(crate) fn pass_set(outputs: &[OutputDecl], pass: &PassDecl) -> Vec<SetFormat> {
    let Some(layer) = pass_layer(pass) else {
        return Vec::new();
    };
    match pass.scene.as_ref().filter(|scene| scene.core) {
        Some(scene) if !scene.pass.writes.contains(engine_core::TargetName::COLOR) => scene
            .pass
            .writes
            .targets()
            .iter()
            .filter_map(|name| outputs.iter().position(|output| output.name == name.name()))
            .map(|index| {
                (
                    Target::Output(index as u32),
                    outputs[index].format,
                    Clear::Color(TRANSPARENT),
                )
            })
            .collect(),
        Some(_) => set_formats(outputs, pass, layer),
        None if !pass.colors.is_empty() => Vec::new(),
        None => set_formats(outputs, pass, layer),
    }
}

fn written_module(
    declared: &Declared,
    decl: &PipelineDecl,
    (outputs, linker): (&[OutputDecl], &Linker),
    (path, problems): (&str, &mut Vec<Error>),
) -> Option<Option<(String, String)>> {
    let mut found: Option<(bool, Vec<u32>)> = None;
    for (index, pass) in declared.passes.iter().enumerate() {
        if !decl_in_pass(decl, index as u32, pass) {
            continue;
        }
        let set = pass_set(outputs, pass);
        let written = if lit_set(&set) {
            let written: Vec<u32> = set
                .iter()
                .filter_map(|(target, ..)| match target {
                    Target::Output(output) => Some(*output),
                    Target::Scene | Target::Depth | Target::Canvas | Target::Copy(..) => None,
                })
                .collect();
            (true, written)
        } else if pass.written.is_empty() {
            continue;
        } else if pass.written.len() == pass.colors.len() && pass.targets.is_empty() {
            (false, pass.written.clone())
        } else {
            problems.push(Error::new(ErrorCode::PassMismatch, path.to_owned()).got(pass.name));
            return None;
        };
        if found.as_ref().is_some_and(|known| *known != written) {
            problems.push(Error::new(ErrorCode::PassMismatch, path.to_owned()).got(pass.name));
            return None;
        }
        found = Some(written);
    }
    Some(found.map(|(colored, written)| {
        let listed: Vec<(&OutputDecl, bool)> = written
            .iter()
            .filter_map(|at| outputs.get(*at as usize))
            .map(|output| {
                let asks = output.module.as_deref().is_some_and(|module| {
                    crate::shared::asks(linker.text(module).unwrap_or_default())
                });
                (output, asks)
            })
            .collect();
        let shades = colored
            || listed.iter().any(|(output, _)| {
                output.module.as_deref().is_some_and(|module| {
                    crate::shared::reads_shaded(
                        linker.text(module).unwrap_or_default(),
                        output.name,
                    )
                })
            });
        crate::shared::pass_module(colored, &listed, shades)
    }))
}

pub(crate) fn lit_set(set: &[SetFormat]) -> bool {
    set.iter().any(|(target, ..)| *target == Target::Scene)
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
            let Some(layer) = pass_layer(pass) else {
                return false;
            };
            if writes(index) {
                return false;
            }
            if !shared(pass) {
                return true;
            }
            let mut writers = passes
                .iter()
                .enumerate()
                .filter(|&(other, _)| other != index && writes(other))
                .map(|(_, other)| other);
            writers
                .clone()
                .all(|other| pass_layer(other) != Some(layer))
                && writers.any(|other| {
                    shared(other)
                        && pass_layer(other).is_some_and(|at| at < layer)
                        && (other.views == PassViews::Every || other.views == pass.views)
                })
        })
        .collect()
}

pub(crate) fn reads_depth_direct(direct: &[bool], pass: usize) -> bool {
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
        Bound::Target(view) => Some(Handle::Target(final_target(*view))),
        Bound::Sampler(_) | Bound::Asset => None,
    }
}

pub(crate) fn core_target(name: &str) -> Option<Target> {
    match name {
        crate::shared::SCENE => Some(Target::Scene),
        crate::shared::DEPTH => Some(Target::Depth),
        _ => None,
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

fn attachments(pass: &PassDecl, outputs: &[OutputDecl], depth_writes: bool) -> Vec<Handle> {
    let mut found = Vec::new();
    for &(texture, _) in pass.colors.iter().chain(&pass.depths) {
        found.push(Handle::whole(texture));
    }
    if pass_layer(pass).is_some() {
        for (target, ..) in pass_set(outputs, pass) {
            found.push(Handle::Target(target));
        }
        if shared(pass) || depth_writes {
            found.push(Handle::Target(Target::Depth));
        }
    }
    for &view in &pass.targets {
        found.push(Handle::Target(final_target(view)));
    }
    if crate::shared::draws_canvas(pass) {
        found.push(Handle::Target(Target::Canvas));
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
        limits: &crate::pipelines::StageLimits,
        problems: &mut Vec<Error>,
    ) -> Option<Self> {
        let Composing {
            linker,
            declared,
            outputs,
            switches,
            core_frame,
            numbers,
            variables,
            frame_bound,
            material_entries,
            depth_writes,
            reaches,
            staged,
        } = composing;
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/material-group"),
            entries: &material_entries,
        });
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/empty-group"),
            entries: &[],
        });
        let doubled = doubled_of(&declared, &staged, &frame_bound, &linker);
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
                    let depth = depth_writes.get(index).copied().unwrap_or(true);
                    written.extend(
                        attachments(pass, &outputs, depth)
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
        let held: Vec<Option<(Handle, Ty)>> = (0..variables.len())
            .map(|at| {
                let (path, name, ty) = &core_frame[at];
                let handle = match numbers.get(at).and_then(|number| frame_bound.get(number)) {
                    Some(bound) => handle_of(bound),
                    None if linker.is_core(path) => core_target(name).map(Handle::Target),
                    None => None,
                };
                handle.map(|handle| (handle, *ty))
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
        let plan = match crate::frame_group::plan(&variables, &reaches, limits, &conflict) {
            Ok(plan) => plan,
            Err(mut refused) => {
                problems.append(&mut refused);
                return None;
            }
        };
        let (variants, frame_keys): (Vec<FrameVariant>, Vec<String>) = plan
            .variants
            .iter()
            .map(|variant| {
                let entries: Vec<(u32, Ty, ShaderStages)> = variant
                    .entries
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
        let over_limit = plan
            .over
            .iter()
            .map(|(pipeline, error)| (staged[*pipeline as usize].0.part, error.clone()))
            .collect();
        let mut pipelines = Vec::with_capacity(staged.len());
        for (index, (decl, stages, reached)) in staged.into_iter().enumerate() {
            let skipped = plan.skipped.get(index).copied().unwrap_or(false);
            let material = reached
                .iter()
                .any(|(module, found, _)| found.group == 1 && linker.is_core(module));
            let mut groups: Vec<OwnGroup> = Vec::new();
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            let mut view_reads = Vec::new();
            for (module, found, _) in &reached {
                if found.group == 0 {
                    if linker.is_core(module) {
                        view_reads.extend(core_target(&found.name));
                    }
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
            if skipped {
                groups.clear();
            }
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
            let variant = plan.of_pipeline.get(index).copied().unwrap_or(0);
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
            for at in (1..=top).filter(|_| !skipped) {
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
            format!("{:?}", decl.program).hash(&mut hasher);
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
                view_reads,
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
                skipped,
            });
        }
        let direct_depth = depth_direct_of(&declared, &depth_writes);
        let view_copies = copies(&declared, &pipelines, &direct_depth);
        let (graph, ids) = match graph_of(&declared, &outputs, &pipelines, &direct_depth, &doubled)
        {
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
                let core = if !linker.is_core(path) {
                    None
                } else if name == crate::shared::POST {
                    Some(CoreFrame::Post)
                } else if name == crate::shared::LINEAR {
                    Some(CoreFrame::Linear)
                } else {
                    core_target(name).map(CoreFrame::Target)
                };
                Some((number, FrameSlot { ty: *ty, core }))
            })
            .collect();
        let binding_bytes = binding_bytes_of(&pipelines);
        Some(Self {
            declared,
            outputs,
            graph,
            ids,
            pipelines,
            shaders: BTreeMap::new(),
            frame_slots,
            frame_bound,
            variants,
            switches,
            view_copies,
            doubled,
            direct_depth,
            binding_bytes,
            linker,
            over_limit,
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

fn doubled_of(
    declared: &Declared,
    staged: &[(PipelineDecl, Vec<Stage>, Reached)],
    frame_bound: &BTreeMap<u32, Bound>,
    linker: &Linker,
) -> Vec<bool> {
    let texture_of = |bound: &Bound| match bound {
        Bound::Texture(texture) | Bound::Slice { texture, .. } => Some(*texture),
        Bound::Buffer(_)
        | Bound::Window { .. }
        | Bound::Sampler(_)
        | Bound::Target(_)
        | Bound::Asset => None,
    };
    declared
        .textures
        .iter()
        .enumerate()
        .map(|(texture, decl)| {
            if decl.lifetime != Lifetime::Previous {
                return false;
            }
            let texture = texture as u32;
            declared.passes.iter().enumerate().any(|(index, pass)| {
                let mut written = pass
                    .colors
                    .iter()
                    .chain(&pass.depths)
                    .any(|(known, _)| *known == texture);
                let mut read = false;
                if let Work::Copy { from, to } = pass.work {
                    read |= from == Copied::Texture(texture);
                    written |= to == Copied::Texture(texture);
                }
                for (decl, _, reached) in staged
                    .iter()
                    .filter(|(decl, _, _)| decl_in_pass(decl, index as u32, pass))
                {
                    for (module, found, _) in reached {
                        let bound = if found.group == 0 {
                            linker
                                .binding_of(module, &found.name)
                                .and_then(|number| frame_bound.get(&number))
                                .cloned()
                        } else {
                            decl.bound.iter().find_map(|(var, bound)| {
                                let (known, name) =
                                    crate::linker::path_of((var.module(), var.name()));
                                (known == module && name == found.name).then(|| bound.clone())
                            })
                        };
                        if bound.as_ref().and_then(texture_of) == Some(texture) {
                            if found.ty.writes() {
                                written = true;
                            } else {
                                read = true;
                            }
                        }
                    }
                }
                written && read
            })
        })
        .collect()
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
        Some(Bound::Target(ViewTarget::Depth)) => Some(DEPTH_FORMAT),
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

pub(crate) fn layered(output: &OutputDecl) -> bool {
    !output.layers.is_empty()
}

pub(crate) fn output_location(
    outputs: &[OutputDecl],
    pass: &PassDecl,
    name: engine_core::TargetName,
) -> Option<usize> {
    let output = outputs
        .iter()
        .position(|output| output.name == name.name())?;
    pass_set(outputs, pass)
        .iter()
        .position(|(target, ..)| *target == Target::Output(output as u32))
}

fn set_formats(outputs: &[OutputDecl], pass: &PassDecl, layer: AttachPoint) -> Vec<SetFormat> {
    let named = pass
        .scene
        .map_or(engine_core::Writes::NOTHING, |scene| scene.pass.writes);
    let mut set = vec![(Target::Scene, SCENE_FORMAT, Clear::Color(BLACK))];
    for (index, output) in outputs.iter().enumerate() {
        if output.layers.contains(&layer)
            || named
                .targets()
                .iter()
                .any(|name| name.name() == output.name)
        {
            set.push((
                Target::Output(index as u32),
                output.format,
                Clear::Color(TRANSPARENT),
            ));
        }
    }
    set
}

pub(crate) fn target_read(
    view: ViewTarget,
    layer: Option<AttachPoint>,
    direct_depth: bool,
) -> Target {
    match (layer, view) {
        (Some(_), ViewTarget::Depth) if direct_depth => Target::Depth,
        (Some(layer), view) => Target::Copy(layer, view),
        (None, view) => final_target(view),
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn graph_of(
    declared: &Declared,
    outputs: &[OutputDecl],
    pipelines: &[BuiltPipeline],
    direct: &[bool],
    doubled: &[bool],
) -> Result<(StaticGraph, Ids), Error> {
    let mut resources = Vec::new();
    let mut ids = Ids {
        textures: Vec::new(),
        buffers: Vec::new(),
        targets: Vec::new(),
    };
    for (texture, &doubled) in declared.textures.iter().zip(doubled) {
        ids.textures.push(ResourceId::new(resources.len()));
        resources.push(ResourceEntry {
            label: texture.label,
            life: if doubled { Life::Kept } else { life(texture.lifetime) },
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
    target(Target::Scene, "scene", true, Life::Output, &mut ids);
    target(Target::Depth, "depth", true, Life::Output, &mut ids);
    for (index, output) in outputs.iter().enumerate() {
        target(
            Target::Output(index as u32),
            output.name,
            true,
            Life::Output,
            &mut ids,
        );
    }
    target(Target::Canvas, "canvas", false, Life::Output, &mut ids);
    let copied = copies(declared, pipelines, direct);
    for &(layer, view) in &copied {
        target(
            Target::Copy(layer, view),
            "view_copy",
            false,
            Life::Frame,
            &mut ids,
        );
    }
    let mut passes = Vec::with_capacity(declared.passes.len() + copied.len());
    for (index, pass) in declared.passes.iter().enumerate() {
        let part = declared.parts.get(pass.part as usize);
        let entry = PassEntry {
            label: pass.name,
            brick: part.map_or("", |part| part.package),
            registered: index as u32,
            layer: pass.layer,
            shared: shared(pass),
            group: pass.group,
            blends: pipelines
                .iter()
                .filter(|pipeline| in_pass(pipeline, index as u32, pass))
                .any(|pipeline| composes(&pipeline.decl.program)),
        };
        let scene = pass_layer(pass);
        let direct_depth = reads_depth_direct(direct, index);
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
                Bound::Target(view) => ids.target(target_read(*view, scene, direct_depth)),
                Bound::Sampler(_) | Bound::Asset => None,
            }
        };
        let mut current = Vec::new();
        for pipeline in pipelines
            .iter()
            .filter(|pipeline| in_pass(pipeline, index as u32, pass))
        {
            reads.extend(pipeline.reads.iter().filter_map(&bound_id));
            if scene.is_none() {
                reads.extend(
                    pipeline
                        .view_reads
                        .iter()
                        .filter_map(|target| ids.target(*target)),
                );
            }
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
                    Copied::Target(view) => ids.target(final_target(view)),
                    Copied::Canvas => ids.target(Target::Canvas),
                };
                reads.extend(id(from));
                writes.extend(id(to));
            }
            Work::Clear(buffer) => writes.extend(ids.buffers.get(buffer as usize).copied()),
            Work::Render => {}
        }
        if scene.is_some() {
            for (target, _, clear) in pass_set(outputs, pass) {
                if let Some(id) = ids.target(target) {
                    let access = if entry.shared {
                        GraphAccess::Draw(clear)
                    } else {
                        GraphAccess::Load
                    };
                    colors.push((id, access));
                }
            }
            if let Some(id) = ids.target(Target::Depth) {
                let access = if direct_depth {
                    GraphAccess::ReadOnly
                } else if entry.shared {
                    GraphAccess::Draw(Clear::Depth(FAR))
                } else {
                    GraphAccess::Load
                };
                depth = Some((id, access));
            }
        }
        for &view in &pass.targets {
            if let Some(id) = ids.target(final_target(view)) {
                colors.push((id, GraphAccess::Load));
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
    let canvas = ids.target(Target::Canvas);
    if let (Some(at), Some(canvas)) = (
        declared
            .passes
            .iter()
            .position(|pass| pass.name == crate::shared::COMPOSITE),
        canvas,
    ) {
        passes[at].colors.insert(
            0,
            (
                canvas,
                GraphAccess::Clear(Clear::Color(crate::graph::BLACK)),
            ),
        );
        for target in [Target::Scene, Target::Depth] {
            passes[at].reads.extend(ids.target(target));
        }
    }
    let screens: Vec<PassId> = declared
        .passes
        .iter()
        .enumerate()
        .filter(|(_, pass)| crate::shared::composites(pass))
        .map(|(index, _)| PassId::new(index))
        .collect();
    for (at, pass) in declared.passes.iter().enumerate() {
        let Some(canvas) = canvas.filter(|_| pass.layer == Some(AttachPoint::Overlay)) else {
            continue;
        };
        passes[at].colors.insert(
            0,
            (canvas, GraphAccess::Draw(Clear::Color(crate::graph::BLACK))),
        );
        passes[at].after.extend(screens.iter().copied());
    }
    for &(layer, view) in &copied {
        let (Some(copy), Some(from)) = (
            ids.target(Target::Copy(layer, view)),
            ids.target(final_target(view)),
        ) else {
            continue;
        };
        passes.push(GraphPass {
            entry: PassEntry {
                label: "view_copy",
                brick: "",
                registered: 0,
                layer: Some(layer),
                shared: true,
                group: None,
                blends: false,
            },
            reads: vec![from],
            writes: vec![copy],
            colors: Vec::new(),
            depth: None,
            after: Vec::new(),
            current: Vec::new(),
        });
    }
    let graph = StaticGraph::build(resources, passes)?;
    Ok((graph, ids))
}

fn composes(program: &Program) -> bool {
    match program {
        Program::Render(render) => {
            render.blend.is_some()
                || render
                    .targets
                    .as_ref()
                    .is_some_and(|targets| targets.iter().any(|(blend, _)| blend.is_some()))
                || matches!(render.depth, DepthUse::None | DepthUse::Overwrite)
        }
        Program::Compute(_) => false,
    }
}

fn copies(
    declared: &Declared,
    pipelines: &[BuiltPipeline],
    direct: &[bool],
) -> Vec<(AttachPoint, ViewTarget)> {
    let mut found: Vec<(AttachPoint, ViewTarget)> = Vec::new();
    for (index, pass) in declared.passes.iter().enumerate() {
        let Some(layer) = pass_layer(pass) else {
            continue;
        };
        let direct_depth = reads_depth_direct(direct, index);
        for pipeline in pipelines
            .iter()
            .filter(|pipeline| in_pass(pipeline, index as u32, pass))
        {
            for bound in &pipeline.reads {
                let Bound::Target(view) = bound else {
                    continue;
                };
                if target_read(*view, Some(layer), direct_depth) == Target::Copy(layer, *view)
                    && !found.contains(&(layer, *view))
                {
                    found.push((layer, *view));
                }
            }
        }
    }
    found
}
