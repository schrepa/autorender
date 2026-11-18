use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;

use bevy_ecs::world::World;
use engine_core::{
    Error, ErrorCode, FrameId, GeometryHandle, MaterialHandle, Readback, TextureHandle,
};

use crate::graph::{Bits, Clear, FAR, Load, Ops, Plan, Planned};
use crate::install::{Bound, Copied, PassDecl, Program, Reinstall, Scoped, ViewTarget, Work};
use crate::linker::Ty;
use crate::machine::{
    CAMERA_STRIDE, CoreFrame, DEPTH_FORMAT, Machine, Ready, SetFormat, Target, final_target,
    lit_set, pass_layer, pass_set, reads_depth_direct, shared, target_read,
};
use crate::profiling::GpuProfiler;
use crate::readback::{Deliver, Keep, Readbacks, Reader};
use crate::registration::{Installed, KindMember};
use crate::{AfterRead, Allocation, CameraUniform, DepthUse, SceneDraw, VertexLayout};

mod arena;
mod compile;
mod copies;
mod draw;
mod making;
mod passes;
mod recording;
mod resources;
mod run;
mod views;
mod warming;

pub(crate) use arena::Arena;
pub(crate) use compile::Compiler;
pub(crate) use recording::{Ended, Recording, Step};
pub(crate) use run::{Bindings, DroppedTally, KindAnswer, Pipelines, Run};
pub(crate) use views::Tables;

pub(crate) type FrameCore = Run;
pub(crate) type EncodeCore = Run;

const MOST_STREAMS: usize = 16;
const MOST_OFFSETS: usize = 16;
const MOST_GROUPS: usize = 8;
const MOST_COLORS: usize = 8;

#[derive(Clone, Copy)]
pub(crate) struct Streams {
    held: [(u32, u64); MOST_STREAMS],
    count: usize,
}

impl Streams {
    pub(crate) fn of(given: impl Iterator<Item = (u32, u64)>) -> Self {
        let mut held = [(0, 0); MOST_STREAMS];
        let mut count = 0;
        for (slot, stream) in held.iter_mut().zip(given) {
            *slot = stream;
            count += 1;
        }
        Self { held, count }
    }

    fn listed(&self) -> &[(u32, u64)] {
        &self.held[..self.count]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Holder {
    Frame(u32),
    Own(u32, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Held {
    Texture(u32),
    Buffer(u32),
    Targets,
    Cameras,
}

#[derive(Default)]
pub(crate) struct Holders {
    textures: Vec<Vec<Holder>>,
    buffers: Vec<Vec<Holder>>,
    targets: Vec<Holder>,
    cameras: Vec<Holder>,
    post: Vec<Holder>,
}

impl Holders {
    fn of(machine: &Machine) -> Self {
        let mut holders = Self {
            textures: vec![Vec::new(); machine.declared.textures.len()],
            buffers: vec![Vec::new(); machine.declared.buffers.len()],
            ..Self::default()
        };
        for (variant, found) in machine.variants.iter().enumerate() {
            let holder = Holder::Frame(variant as u32);
            for &(binding, _) in &found.entries {
                match (binding, machine.frame_bound.get(&binding)) {
                    (0, _) => holders.cameras.push(holder),
                    (_, Some(bound)) => holders.hold(bound, holder),
                    (_, None) => match machine
                        .frame_slots
                        .get(&binding)
                        .and_then(|slot| slot.core.as_ref())
                    {
                        Some(CoreFrame::Target(_)) => holders.targets.push(holder),
                        Some(CoreFrame::Post) => holders.post.push(holder),
                        None => {}
                    },
                }
            }
        }
        for (pipeline, built) in machine.pipelines.iter().enumerate() {
            for (group, own) in built.groups.iter().enumerate() {
                for (_, bound, _) in &own.entries {
                    holders.hold(bound, Holder::Own(pipeline as u32, group as u32));
                }
            }
        }
        holders
    }

    fn hold(&mut self, bound: &Bound, holder: Holder) {
        match bound {
            Bound::Texture(texture) | Bound::Slice { texture, .. } => {
                self.textures[*texture as usize].push(holder)
            }
            Bound::Buffer(buffer) | Bound::Window { buffer, .. } => {
                self.buffers[*buffer as usize].push(holder)
            }
            Bound::Target(_) => self.targets.push(holder),
            Bound::Sampler(_) | Bound::Asset => {}
        }
    }

    fn of_held(&self, held: Held) -> &[Holder] {
        match held {
            Held::Texture(texture) => &self.textures[texture as usize],
            Held::Buffer(buffer) => &self.buffers[buffer as usize],
            Held::Targets => &self.targets,
            Held::Cameras => &self.cameras,
        }
    }

    fn of_view(&self, machine: &Machine) -> std::collections::BTreeSet<Holder> {
        let textures = machine
            .declared
            .textures
            .iter()
            .enumerate()
            .filter(|(_, texture)| texture.per_view)
            .flat_map(|(index, _)| self.textures[index].iter());
        let buffers = machine
            .declared
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.per_view)
            .flat_map(|(index, _)| self.buffers[index].iter());
        self.targets
            .iter()
            .chain(&self.post)
            .chain(textures)
            .chain(buffers)
            .copied()
            .collect()
    }
}

pub(crate) struct Groups {
    pub(crate) made: [Option<wgpu::BindGroup>; 2],
}

impl Groups {
    fn fresh() -> Self {
        Self { made: [None, None] }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct PassLayout {
    formats: [Option<wgpu::TextureFormat>; MOST_COLORS],
    count: usize,
    depth: Option<(wgpu::TextureFormat, bool)>,
    plain: bool,
    main: bool,
}

impl PassLayout {
    fn formats(&self) -> &[Option<wgpu::TextureFormat>] {
        &self.formats[..self.count]
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Compiled {
    program: u64,
    formats: Vec<Option<wgpu::TextureFormat>>,
    depth: Option<(wgpu::TextureFormat, bool)>,
    plain: bool,
    main: bool,
    variant: Option<crate::surface::MaterialVariant>,
    needed: Needed,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
enum Needed {
    Later,
    First,
}

impl Compiled {
    fn same(&self, other: &Self) -> bool {
        (
            self.program,
            &self.formats,
            self.depth,
            self.plain,
            self.main,
            self.variant,
        ) == (
            other.program,
            &other.formats,
            other.depth,
            other.plain,
            other.main,
            other.variant,
        )
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Listing {
    canvas: wgpu::TextureFormat,
    compiled: Vec<Compiled>,
}

fn specialized(compiled: &Compiled) -> Option<crate::surface::MaterialVariant> {
    compiled.variant
}

const GROUPED: bool = cfg!(any(not(target_arch = "wasm32"), feature = "profiling"));

const PLACEHOLDER_BYTES: u64 = 4_096;

pub(crate) struct Allocated {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) size: wgpu::Extent3d,
    pub(crate) mips: u32,
    pub(crate) slices: Vec<SliceView>,
}

type SliceView = (
    Range<u32>,
    Range<u32>,
    Option<wgpu::TextureViewDimension>,
    wgpu::TextureView,
);

pub(crate) struct Made {
    pub(crate) label: &'static str,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) usage: wgpu::TextureUsages,
    pub(crate) dimension: wgpu::TextureDimension,
}

impl Allocated {
    fn new(device: &wgpu::Device, made: &Made, size: wgpu::Extent3d, mips: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(made.label),
            size,
            mip_level_count: mips.max(1),
            sample_count: 1,
            dimension: made.dimension,
            format: made.format,
            usage: made.usage,
            view_formats: &[],
        });
        let layered =
            made.dimension == wgpu::TextureDimension::D2 && size.depth_or_array_layers > 1;
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: layered.then_some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            texture,
            view,
            size,
            mips: mips.max(1),
            slices: Vec::new(),
        }
    }

    fn layers(&self) -> u32 {
        if self.texture.dimension() == wgpu::TextureDimension::D3 {
            1
        } else {
            self.size.depth_or_array_layers
        }
    }

    fn slice(
        &mut self,
        mips: &Range<u32>,
        layers: &Range<u32>,
        dimension: Option<wgpu::TextureViewDimension>,
    ) -> Option<&wgpu::TextureView> {
        let mips = mips.start..mips.end.min(self.mips);
        let layers = layers.start..layers.end.min(self.layers());
        if mips.start >= mips.end || layers.start >= layers.end {
            return None;
        }
        let at = match self
            .slices
            .iter()
            .position(|(known_mips, known_layers, known, _)| {
                *known_mips == mips && *known_layers == layers && *known == dimension
            }) {
            Some(at) => at,
            None => {
                let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mips.start,
                    mip_level_count: Some(mips.end - mips.start),
                    base_array_layer: layers.start,
                    array_layer_count: Some(layers.end - layers.start),
                    dimension,
                    ..Default::default()
                });
                self.slices.push((mips, layers, dimension, view));
                self.slices.len() - 1
            }
        };
        Some(&self.slices[at].3)
    }
}

pub(crate) struct Asked {
    pub(crate) part: u32,
    pub(crate) tag: u32,
    pub(crate) buffer: u32,
    pub(crate) range: Range<u64>,
    pub(crate) then: AfterRead,
    pub(crate) once: bool,
    pub(crate) deliver: Deliver,
}

pub(crate) struct Arrived {
    part: u32,
    tag: u32,
    read: Readback<Option<Vec<u8>>>,
    deliver: Deliver,
}

pub(crate) struct Filled {
    pub(crate) base: u32,
    pub(crate) textures: Vec<TextureHandle>,
    pub(crate) generation: u64,
    pub(crate) used: u64,
    pub(crate) groups: Vec<(u32, wgpu::BindGroup)>,
}

pub(crate) struct Open {
    pub(crate) pass: wgpu::RenderPass<'static>,
    pub(crate) attachments: Vec<u64>,
    pub(crate) layout: PassLayout,
    pub(crate) offset: u32,
    pub(crate) shared: bool,
    pub(crate) layered: bool,
    pub(crate) label: Option<u32>,
}

pub(crate) struct Assets {
    pub(crate) meshes: crate::mesh_arena::GpuMeshes,
    pub(crate) materials: crate::materials::MaterialStore,
}

#[derive(Default)]
pub(crate) struct Queued {
    pub(crate) before: Option<wgpu::CommandEncoder>,
    pub(crate) refused: Option<Error>,
    pub(crate) cleared: Vec<u32>,
    pub(crate) ending: Vec<(Copied, Copied, Vec<crate::CopyRegion>)>,
    pub(crate) copies: Vec<(u32, Vec<crate::CopyRegion>)>,
    pub(crate) asked: Vec<Asked>,
}

#[derive(Default)]
pub(crate) struct Timed {
    pub(crate) asked: Option<&'static str>,
    pub(crate) open: Option<&'static str>,
}

#[derive(Default)]
pub(crate) struct Reached {
    pub(crate) generation: Option<u64>,
    pub(crate) parts: Vec<bool>,
}

pub(crate) enum Placeholder {
    Buffer(wgpu::Buffer),
    Texture(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

pub(crate) enum RawShape {
    Pulled {
        vertices: Range<u32>,
        instances: Range<u32>,
        material: Option<MaterialHandle>,
    },
    Buffers {
        vertices: Streams,
        indices: Option<u32>,
        elements: Range<u32>,
        instances: Range<u32>,
    },
    Indirect {
        vertices: Streams,
        indices: Option<u32>,
        args: u32,
        offset: u64,
        material: Option<MaterialHandle>,
    },
    IndirectAt {
        vertices: Streams,
        indices: Option<u32>,
        args: u32,
        offset: u64,
    },
    MeshIndirect {
        mesh: GeometryHandle,
        material: Option<MaterialHandle>,
        args: u32,
        offset: u64,
        streams: Streams,
    },
    Mesh {
        mesh: GeometryHandle,
        material: Option<MaterialHandle>,
        instances: Range<u32>,
    },
}

#[derive(Default)]
pub(crate) struct Spare {
    colors: Vec<(u64, Option<wgpu::TextureView>, wgpu::TextureFormat, Ops)>,
    attachments: Vec<u64>,
    cameras: Vec<u8>,
}

const TARGET_KEY: u64 = 1 << 62;

fn target_key(target: Target) -> u64 {
    let (kind, index) = match target {
        Target::Scene => (0, 0),
        Target::Depth => (1, 0),
        Target::Canvas => (2, 0),
        Target::Copy(layer, view) => {
            let view = match view {
                ViewTarget::Scene => 0,
                ViewTarget::Depth => 1,
            };
            (5, view << 8 | layer as u64)
        }
    };
    TARGET_KEY | kind << 32 | index
}

fn fault(path: String, code: ErrorCode) -> Error {
    Error::new(code, path)
}

fn bound_texture(bound: &Bound) -> Option<u32> {
    match bound {
        Bound::Texture(texture) | Bound::Slice { texture, .. } => Some(*texture),
        Bound::Buffer(_)
        | Bound::Window { .. }
        | Bound::Sampler(_)
        | Bound::Target(_)
        | Bound::Asset => None,
    }
}

fn texture_uses(machine: &Machine, texture: u32) -> impl Iterator<Item = Ty> + '_ {
    let own = machine.pipelines.iter().flat_map(move |pipeline| {
        pipeline.groups.iter().flat_map(move |group| {
            group
                .entries
                .iter()
                .filter(move |(_, bound, _)| bound_texture(bound) == Some(texture))
                .map(|(_, _, ty)| *ty)
        })
    });
    let frame = machine
        .frame_bound
        .iter()
        .filter_map(move |(binding, bound)| {
            (bound_texture(bound) == Some(texture))
                .then(|| machine.frame_slots.get(binding).map(|slot| slot.ty))
                .flatten()
        });
    own.chain(frame)
}

fn texture_usage(machine: &Machine, texture: u32) -> wgpu::TextureUsages {
    let mut usage = wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_DST
        | wgpu::TextureUsages::COPY_SRC;
    let attached = machine.declared.passes.iter().any(|pass| {
        pass.colors
            .iter()
            .chain(&pass.depths)
            .any(|(known, _)| *known == texture)
    });
    if attached {
        usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    if texture_uses(machine, texture).any(|ty| matches!(ty, Ty::StorageTexture { .. })) {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    usage
}

fn texture_dimension(machine: &Machine, texture: u32) -> wgpu::TextureDimension {
    let volume = texture_uses(machine, texture).any(|ty| {
        matches!(
            ty,
            Ty::Texture {
                view: wgpu::TextureViewDimension::D3,
                ..
            } | Ty::StorageTexture {
                view: wgpu::TextureViewDimension::D3,
                ..
            }
        )
    });
    if volume {
        wgpu::TextureDimension::D3
    } else {
        wgpu::TextureDimension::D2
    }
}

fn target_format(target: Target) -> Option<wgpu::TextureFormat> {
    match target {
        Target::Scene | Target::Copy(_, ViewTarget::Scene) => Some(crate::machine::SCENE_FORMAT),
        Target::Depth | Target::Copy(_, ViewTarget::Depth) => Some(DEPTH_FORMAT),
        Target::Canvas => None,
    }
}

fn uniform_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl Run {
    pub(crate) fn listing(&self) -> Vec<u8> {
        let mut compiled = self.pipelines.compiled.clone();
        compiled.sort_by_cached_key(|entry| serde_json::to_string(entry).unwrap_or_default());
        compiled.dedup();
        let listing = Listing {
            canvas: self.canvas.surface,
            compiled,
        };
        serde_json::to_vec(&listing).unwrap_or_default()
    }

    pub(crate) fn create_listed(&mut self, record: &[u8]) -> Result<(), Error> {
        if record.is_empty() {
            return Ok(());
        }
        let listing: Listing = serde_json::from_slice(record).map_err(|_| {
            crate::platform::failed("$.pipelines").expected(engine_core::Expected::Json)
        })?;
        let canvas = self.canvas.surface;
        let mut keys = listing.compiled;
        for key in &mut keys {
            for format in key.formats.iter_mut().flatten() {
                if *format == listing.canvas {
                    *format = canvas;
                }
            }
        }
        self.pipelines.listed = Some(keys);
        self.pipelines.unlisted = 0;
        self.compile_first();
        if self.reached.generation.is_some() {
            self.compile_listed();
        }
        Ok(())
    }

    pub(crate) fn set(&self) -> &views::ViewSet {
        &self.sets.list[self.sets.at]
    }

    pub(crate) fn set_mut(&mut self) -> &mut views::ViewSet {
        &mut self.sets.list[self.sets.at]
    }

    fn view_texture(&self, texture: u32) -> bool {
        self.machine
            .declared
            .textures
            .get(texture as usize)
            .is_some_and(|decl| decl.per_view)
    }

    fn view_buffer(&self, buffer: u32) -> bool {
        self.machine
            .declared
            .buffers
            .get(buffer as usize)
            .is_some_and(|decl| decl.per_view)
    }

    pub(crate) fn texture_tables(&self, texture: u32) -> &views::Tables {
        if self.view_texture(texture) {
            &self.set().tables
        } else {
            &self.tables
        }
    }

    pub(crate) fn texture_tables_mut(&mut self, texture: u32) -> &mut views::Tables {
        if self.view_texture(texture) {
            &mut self.set_mut().tables
        } else {
            &mut self.tables
        }
    }

    pub(crate) fn buffer_tables_mut(&mut self, buffer: u32) -> &mut views::Tables {
        if self.view_buffer(buffer) {
            &mut self.set_mut().tables
        } else {
            &mut self.tables
        }
    }

    pub(crate) fn buffer_held(&self, buffer: u32) -> Option<&(wgpu::Buffer, u64)> {
        let tables = if self.view_buffer(buffer) {
            &self.set().tables
        } else {
            &self.tables
        };
        tables.buffers.get(buffer as usize)?.as_ref()
    }

    pub(crate) fn touch(&mut self, held: Held) {
        for &holder in self.bindings.holders.of_held(held) {
            let fresh = if self.sets.holders.contains(&holder) {
                let at = self.sets.at;
                let mut fresh = false;
                for (index, set) in self.sets.list.iter_mut().enumerate() {
                    fresh |= set.stale.insert(holder) && index == at;
                }
                fresh
            } else {
                self.bindings.stale.insert(holder)
            };
            if !fresh {
                continue;
            }
            let changed = match held {
                Held::Texture(texture) => self.machine.declared.textures[texture as usize].label,
                Held::Buffer(buffer) => self.machine.declared.buffers[buffer as usize].label,
                Held::Targets => "targets",
                Held::Cameras => "cameras",
            };
            engine_profile::event(
                "rebuilt",
                "bind_group",
                &[("why", engine_profile::Fact::Text(changed))],
            );
        }
    }

    pub(crate) fn planned(&self) -> Option<&Planned> {
        self.planning
            .plan
            .as_ref()?
            .passes
            .get(self.planning.planned_at)
    }

    pub(crate) fn path(&self, what: &str) -> String {
        let name = self
            .machine
            .declared
            .parts
            .get(self.part as usize)
            .map_or("", |part| part.name);
        format!("$.parts.{name}.{what}")
    }

    pub(crate) fn fault(&mut self, error: Error) {
        self.installs.faults.push(error);
    }

    pub(crate) fn drawable(&self, part: u32) -> bool {
        self.installs
            .turned
            .get(part as usize)
            .is_some_and(Option::is_some)
    }

    pub(crate) fn take_states(&mut self, states: Option<&engine_registry::BrickStates>) {
        for part in 0..self.parts.len() {
            let present = self.parts[part].is_some();
            let turned = match self.installs.gates.get(part).copied().flatten() {
                Some(gate) => states.and_then(|states| states.installed(gate)),
                None => Some(0),
            };
            if let Some(slot) = self.installs.turned.get_mut(part) {
                *slot = turned.filter(|_| present);
            }
        }
        self.settle_parts(states);
    }

    pub(crate) fn settle_parts(&mut self, states: Option<&engine_registry::BrickStates>) {
        for part in 0..self.parts.len() {
            let turned = self.installs.turned.get(part).copied().flatten();
            let wanted = match self.installs.gates.get(part).copied().flatten() {
                Some(gate) => states.and_then(|states| states.installed(gate)),
                None => turned,
            };
            if turned == wanted && (turned.is_some() || self.parts[part].is_none()) {
                continue;
            }
            if let Some(slot) = self.installs.turned.get_mut(part) {
                *slot = None;
            }
            if self.parts[part].take().is_some() {
                self.release_part(part as u32);
            }
        }
    }

    pub(crate) fn take_member(&mut self, part: u32) -> Option<Box<dyn Installed>> {
        if !self.drawable(part) {
            return None;
        }
        self.parts.get_mut(part as usize).and_then(Option::take)
    }

    pub(crate) fn give_member(&mut self, part: u32, held: Box<dyn Installed>) {
        if let Some(slot) = self.parts.get_mut(part as usize) {
            *slot = Some(held);
        }
    }

    pub(crate) fn scene_draw(&self) -> Option<SceneDraw> {
        let scene = self
            .machine
            .declared
            .passes
            .get(self.pass as usize)?
            .scene?;
        Some(SceneDraw {
            pass: crate::Scene::named(scene.name),
            bit: scene.bit,
            role: scene.pass.role(),
            index: Some(self.uniforms.camera_slot),
        })
    }

    pub(crate) fn scene_members(&self) -> Vec<KindMember> {
        let Some(pass) = self.machine.declared.passes.get(self.pass as usize) else {
            return Vec::new();
        };
        let Some(scene) = pass.scene else {
            return Vec::new();
        };
        self.installs
            .draw_entries
            .iter()
            .filter(|member| {
                pass.members.contains(&member.part) && member.kind.joins(scene.name, &scene.pass)
            })
            .cloned()
            .collect()
    }

    pub(crate) fn run_pass(&mut self, pass: u32) {
        let owned = self
            .machine
            .declared
            .passes
            .get(pass as usize)
            .is_some_and(|declared| declared.part == self.part);
        if owned && !self.machine.graph.starved(pass as usize) {
            self.planning.runs.insert(pass as usize);
        }
    }

    pub(crate) fn allocate(&mut self, texture: u32, size: wgpu::Extent3d, mips: u32) -> Allocation {
        self.allocate_with(texture, None, size, mips)
    }

    pub(crate) fn allocate_as(
        &mut self,
        texture: u32,
        format: wgpu::TextureFormat,
        size: wgpu::Extent3d,
        mips: u32,
    ) -> Allocation {
        self.allocate_with(texture, Some(format), size, mips)
    }

    fn allocate_with(
        &mut self,
        texture: u32,
        format: Option<wgpu::TextureFormat>,
        size: wgpu::Extent3d,
        mips: u32,
    ) -> Allocation {
        let Some(decl) = self.machine.declared.textures.get(texture as usize) else {
            return Allocation::Fresh;
        };
        let at = texture as usize;
        let wanted = format.unwrap_or(decl.format);
        let tables = self.texture_tables(texture);
        let same = tables.textures[at]
            .as_ref()
            .is_some_and(|made| made.texture.format() == wanted);
        if tables.sizes[at] == Some((size, mips)) && same {
            return Allocation::Kept;
        }
        let why = match tables.sizes[at] {
            None => "first",
            Some(_) if !same => "format",
            Some(_) => "size",
        };
        let dimension = self
            .bindings
            .dimensions
            .get(at)
            .copied()
            .unwrap_or(wgpu::TextureDimension::D2);
        let mut usage = texture_usage(&self.machine, texture);
        if format.is_some() {
            usage &= wanted
                .guaranteed_format_features(self.device.features())
                .allowed_usages;
        }
        if mips > 1 && crate::mipmaps::drawn(wanted, dimension) {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        }
        let made = Made {
            label: decl.label,
            format: wanted,
            usage,
            dimension,
        };
        let fresh = Allocated::new(&self.device, &made, size, mips);
        let tables = self.texture_tables_mut(texture);
        tables.textures[at] = Some(fresh);
        tables.sizes[at] = Some((size, mips));
        engine_profile::event(
            "rebuilt",
            made.label,
            &[("why", engine_profile::Fact::Text(why))],
        );
        self.touch(Held::Texture(texture));
        Allocation::Fresh
    }

    pub(crate) fn grow(&mut self, texture: u32, size: wgpu::Extent3d, mips: u32) -> Allocation {
        let at = texture as usize;
        let Some(Some((known, levels))) = self.texture_tables(texture).sizes.get(at).copied()
        else {
            return self.allocate(texture, size, mips);
        };
        if (known, levels) == (size, mips) {
            return self.allocate(texture, size, mips);
        }
        let tables = self.texture_tables_mut(texture);
        let previous = std::mem::take(&mut tables.textures[at]);
        tables.sizes[at] = None;
        let mut allocation = self.allocate(texture, size, mips);
        if levels == mips {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine/part-grow"),
                });
            let new = &self.texture_tables(texture).textures[at];
            if let (Some(old), Some(new)) = (&previous, new) {
                for level in 0..old.mips.min(new.mips) {
                    let extent = wgpu::Extent3d {
                        width: (known.width.min(size.width) >> level).max(1),
                        height: (known.height.min(size.height) >> level).max(1),
                        depth_or_array_layers: old.layers().min(new.layers()),
                    };
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &old.texture,
                            mip_level: level,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &new.texture,
                            mip_level: level,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        extent,
                    );
                }
            }
            self.queue.submit_now([encoder.finish()]);
            allocation = Allocation::Kept;
        }
        allocation
    }

    pub(crate) fn release(&mut self, texture: u32) {
        let at = texture as usize;
        let tables = self.texture_tables_mut(texture);
        if tables.textures.get_mut(at).and_then(Option::take).is_none() {
            return;
        }
        tables.sizes[at] = None;
        self.touch(Held::Texture(texture));
    }

    pub(crate) fn renew_buffer(&mut self, buffer: u32, bytes: u64) -> Allocation {
        self.release_buffer(buffer);
        self.allocate_buffer(buffer, bytes)
    }

    pub(crate) fn clear_buffer_range(&mut self, buffer: u32, range: Range<u64>) {
        let Some((made, size)) = self.buffer_held(buffer) else {
            let error = fault(self.path("clear_buffer_range"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        if range.start > range.end
            || range.end > *size
            || !range.start.is_multiple_of(4)
            || !range.end.is_multiple_of(4)
        {
            let error = fault(self.path("clear_buffer_range"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        let made = made.clone();
        let device = &self.device;
        let encoder = self.queued.before.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine/part-clears"),
            })
        });
        encoder.clear_buffer(&made, range.start, Some(range.end - range.start));
    }

    pub(crate) fn release_buffer(&mut self, buffer: u32) {
        let Some(held) = self
            .buffer_tables_mut(buffer)
            .buffers
            .get_mut(buffer as usize)
        else {
            return;
        };
        if held.take().is_some() {
            self.touch(Held::Buffer(buffer));
        }
    }

    pub(crate) fn release_part(&mut self, part: u32) {
        let textures: Vec<u32> = (0..self.machine.declared.textures.len() as u32)
            .filter(|&texture| self.machine.declared.textures[texture as usize].part == part)
            .collect();
        for texture in textures {
            self.release(texture);
        }
        let buffers: Vec<u32> = (0..self.machine.declared.buffers.len() as u32)
            .filter(|&buffer| self.machine.declared.buffers[buffer as usize].part == part)
            .collect();
        for buffer in buffers {
            self.release_buffer(buffer);
        }
        let machine = &self.machine;
        for set in &mut self.sets.list {
            set.release(machine, Some(part));
        }
        self.installed_anew(part);
    }

    pub(crate) fn installed_anew(&mut self, part: u32) {
        let next = self.written().next();
        if let Some(installed) = self.installs.installed.get_mut(part as usize) {
            *installed = next;
        }
        self.forget_ran(part);
    }

    pub(crate) fn written(&self) -> FrameId {
        FrameId(self.frames.asked)
    }

    pub(crate) fn arrived(
        &mut self,
        (part, tag): (u32, u32),
        read: Readback<Option<Vec<u8>>>,
        deliver: Deliver,
    ) {
        self.reading.arrived.push(Arrived {
            part,
            tag,
            read,
            deliver,
        });
    }

    pub(crate) fn deliver(&mut self, world: &mut World) {
        for arrived in std::mem::take(&mut self.reading.arrived) {
            let installed = self
                .installs
                .installed
                .get(arrived.part as usize)
                .copied()
                .unwrap_or_default();
            if arrived.read.written < installed {
                continue;
            }
            let read = Readback {
                value: arrived.read.value.as_deref(),
                written: arrived.read.written,
            };
            (arrived.deliver)(world, arrived.tag, read);
        }
    }

    pub(crate) fn forget_ran(&mut self, part: u32) {
        for (pass, declared) in self.machine.declared.passes.iter().enumerate() {
            if declared.part == part {
                for set in &mut self.sets.list {
                    set.ran.remove(pass);
                }
            }
        }
    }

    pub(crate) fn forget_global(&mut self, part: u32) {
        for (pass, declared) in self.machine.declared.passes.iter().enumerate() {
            if declared.part == part && !self.sets.passes.contains(pass) {
                for set in &mut self.sets.list {
                    set.ran.remove(pass);
                }
            }
        }
    }

    pub(crate) fn forget_viewed(&mut self) {
        let at = self.sets.at;
        for pass in self.sets.passes.iter() {
            self.sets.list[at].ran.remove(pass);
        }
    }

    pub(crate) fn ran_now(&mut self, pass: usize) {
        let (at, global) = (self.sets.at, !self.sets.passes.contains(pass));
        for (index, set) in self.sets.list.iter_mut().enumerate() {
            if global || index == at {
                set.ran.insert(pass);
            }
        }
    }

    pub(crate) fn refuse(&mut self, error: Error) {
        if self.queued.refused.is_none() {
            self.queued.refused = Some(error);
        }
    }

    pub(crate) fn allocate_buffer(&mut self, buffer: u32, bytes: u64) -> Allocation {
        let Some(decl) = self.machine.declared.buffers.get(buffer as usize) else {
            return Allocation::Fresh;
        };
        let at = buffer as usize;
        let bytes = bytes.max(16).next_multiple_of(4);
        if self
            .buffer_held(buffer)
            .is_some_and(|(_, known)| *known == bytes)
        {
            return Allocation::Kept;
        }
        let usage = decl.usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let label = decl.label;
        let why = if self.buffer_held(buffer).is_some() {
            "size"
        } else {
            "first"
        };
        let made = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: false,
        });
        self.buffer_tables_mut(buffer).buffers[at] = Some((made, bytes));
        engine_profile::event("rebuilt", label, &[("why", engine_profile::Fact::Text(why))]);
        self.touch(Held::Buffer(buffer));
        Allocation::Fresh
    }

    pub(crate) fn write(&mut self, buffer: u32, offset: u64, bytes: &[u8]) {
        let Some((made, size)) = self.buffer_held(buffer) else {
            let error = fault(self.path("write"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        if offset + bytes.len() as u64 > *size || !bytes.len().is_multiple_of(4) {
            let error = fault(self.path("write"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        self.queue
            .write_declared_buffer(buffer, made, offset, bytes);
    }

    fn written_texture(&self, texture: u32) -> Option<&Allocated> {
        self.texture_tables(texture)
            .textures
            .get(texture as usize)?
            .as_ref()
    }

    pub(crate) fn write_texture(
        &mut self,
        texture: u32,
        level: u32,
        at: wgpu::Origin3d,
        bytes: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        let Some(made) = self.written_texture(texture) else {
            let error = fault(self.path("write_texture"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        if level >= made.mips {
            let error = fault(self.path("write_texture"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        self.queue.write_declared_texture(
            texture,
            wgpu::TexelCopyTextureInfo {
                texture: &made.texture,
                mip_level: level,
                origin: at,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            layout,
            size,
        );
    }
}

pub(crate) enum Resolved {
    Buffer(wgpu::Buffer),
    Dynamic(wgpu::Buffer, u64),
    View(wgpu::TextureView),
    Sampler(wgpu::Sampler),
    Placeholder(Ty),
}

#[derive(Clone, Debug)]
pub(crate) enum AttachRaw {
    Declared,
    Depth(u32, Option<u32>),
    Color(u32, u32, Range<u32>),
    ColorDepth(u32, Option<u32>, u32),
    Colors(Vec<(u32, u32)>),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum CountRaw {
    Groups([u32; 3]),
    Indirect(u32, u64),
}
