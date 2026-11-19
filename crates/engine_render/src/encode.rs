use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode, FrameId, GeometryHandle, MaterialHandle, PassBit, TextureHandle};

use crate::graph::{Bits, Clear, Load, Ops, Plan};
use crate::install::{Bound, PassDecl, Program, Reinstall, Scoped};
use crate::linker::Ty;
use crate::machine::{
    CAMERA_STRIDE, Machine, Ready, SetFormat, Target, lit_set, shared,
};
use crate::profiling::GpuProfiler;
use crate::readback::Readbacks;
use crate::registration::{Installed, KindMember};
use crate::{Allocation, CameraUniform, DepthUse};

mod compile;
mod draw;
mod making;
mod passes;
mod recording;
mod resources;
mod run;
mod views;
mod warming;

pub(crate) use recording::{Ended, Recording, Step};
pub(crate) use resources::{BatchEntryRaw, HeldBatches};
pub(crate) use run::{Bindings, DroppedTally, KindTallies, Pipelines, Run};
pub(crate) use views::Tables;

pub(crate) type FrameCore = Run;
pub(crate) type EncodeCore = Run;

const MOST_COLORS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Holder {
    Frame(u32),
    Own(u32, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Held {
    Buffer(u32),
    Targets,
    Cameras,
}

#[derive(Default)]
pub(crate) struct Holders {
    buffers: Vec<Vec<Holder>>,
    targets: Vec<Holder>,
    cameras: Vec<Holder>,
    post: Vec<Holder>,
    own_base: Vec<u32>,
}

impl Holders {
    fn of(machine: &Machine) -> Self {
        let mut base = machine.variants.len() as u32;
        let own_base = machine
            .pipelines
            .iter()
            .map(|built| {
                let at = base;
                base += built.groups.len() as u32;
                at
            })
            .collect();
        let mut holders = Self {
            buffers: vec![Vec::new(); machine.declared.buffers.len()],
            own_base,
            ..Self::default()
        };
        for (variant, found) in machine.variants.iter().enumerate() {
            let holder = Holder::Frame(variant as u32);
            for &(binding, _) in &found.entries {
                match (binding, machine.frame_bound.get(&binding)) {
                    (0, _) => holders.cameras.push(holder),
                    (_, Some(bound)) => holders.hold(bound, holder),
                    (_, None) => {
                        if machine.frame_slots.get(&binding).is_some_and(|slot| slot.post) {
                            holders.post.push(holder);
                        }
                    }
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

    pub(crate) fn index(&self, holder: Holder) -> u32 {
        match holder {
            Holder::Frame(variant) => variant,
            Holder::Own(pipeline, group) => self
                .own_base
                .get(pipeline as usize)
                .map_or(u32::MAX, |base| base + group),
        }
    }

    fn hold(&mut self, bound: &Bound, holder: Holder) {
        match bound {
            Bound::Buffer(buffer) => self.buffers[*buffer as usize].push(holder),
            Bound::Target(_) => self.targets.push(holder),
            Bound::Sampler(_) | Bound::Asset => {}
        }
    }

    fn of_held(&self, held: Held) -> &[Holder] {
        match held {
            Held::Buffer(buffer) => &self.buffers[buffer as usize],
            Held::Targets => &self.targets,
            Held::Cameras => &self.cameras,
        }
    }

    fn of_view(&self) -> std::collections::BTreeSet<Holder> {
        self.targets.iter().chain(&self.post).copied().collect()
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
    depth: Option<wgpu::TextureFormat>,
    plain: bool,
    main: bool,
}

impl PassLayout {
    fn formats(&self) -> &[Option<wgpu::TextureFormat>] {
        &self.formats[..self.count]
    }
}

const GROUPED: bool = cfg!(any(not(target_arch = "wasm32"), feature = "profiling"));

pub(crate) struct Allocated {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) size: wgpu::Extent3d,
    views: Vec<(Option<wgpu::TextureViewDimension>, wgpu::TextureView)>,
}

pub(crate) struct Made {
    pub(crate) label: &'static str,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) usage: wgpu::TextureUsages,
    pub(crate) samples: u32,
}

impl Allocated {
    fn new(device: &wgpu::Device, made: &Made, size: wgpu::Extent3d, mips: u32) -> Self {
        let texture = crate::gpu_calls::create_texture(device, &wgpu::TextureDescriptor {
            label: Some(made.label),
            size,
            mip_level_count: mips.max(1),
            sample_count: made.samples,
            dimension: wgpu::TextureDimension::D2,
            format: made.format,
            usage: made.usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: (size.depth_or_array_layers > 1)
                .then_some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        Self {
            texture,
            view,
            size,
            views: Vec::new(),
        }
    }

    fn viewed(&mut self, dimension: Option<wgpu::TextureViewDimension>) -> &wgpu::TextureView {
        let at = match self.views.iter().position(|(known, _)| *known == dimension) {
            Some(at) => at,
            None => {
                let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension,
                    ..Default::default()
                });
                self.views.push((dimension, view));
                self.views.len() - 1
            }
        };
        &self.views[at].1
    }
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
    pub(crate) layout: u32,
    pub(crate) offset: u32,
    pub(crate) shared: bool,
    pub(crate) label: Option<u32>,
}

pub(crate) struct Assets {
    pub(crate) meshes: crate::mesh_arena::GpuMeshes,
    pub(crate) textures: crate::textures::TextureStore,
    pub(crate) batches: HeldBatches,
}

#[derive(Default)]
pub(crate) struct Reached {
    pub(crate) generation: Option<u64>,
    pub(crate) parts: Vec<bool>,
}

pub(crate) enum Placeholder {
    Texture(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

#[derive(Default)]
pub(crate) struct Spare {
    colors: Vec<(u64, Option<wgpu::TextureView>, wgpu::TextureFormat, Ops)>,
    attachments: Vec<u64>,
    cameras: Vec<u8>,
}

fn fault(path: String, code: ErrorCode) -> Error {
    Error::new(code, path)
}

fn view_dimension(ty: Ty) -> Option<wgpu::TextureViewDimension> {
    match ty {
        Ty::Texture { view, .. } | Ty::StorageTexture { view, .. } => Some(view),
        Ty::Uniform | Ty::Storage { .. } | Ty::Sampler | Ty::Comparison => None,
    }
}

fn uniform_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    crate::gpu_calls::create_buffer(device, &wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl Run {
    pub(crate) fn set(&self) -> &views::ViewSet {
        &self.sets.list[self.sets.at]
    }

    pub(crate) fn set_mut(&mut self) -> &mut views::ViewSet {
        &mut self.sets.list[self.sets.at]
    }

    pub(crate) fn buffer_held(&self, buffer: u32) -> Option<&(wgpu::Buffer, u64)> {
        self.tables.buffers.get(buffer as usize)?.as_ref()
    }

    pub(crate) fn touch(&mut self, held: Held) {
        for &holder in self.bindings.holders.of_held(held) {
            if self.sets.holders.contains(&holder) {
                for set in &mut self.sets.list {
                    set.stale.insert(holder);
                }
            } else {
                self.bindings.stale.insert(holder);
            }
        }
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

    pub(crate) fn scene_bit(&self) -> Option<PassBit> {
        self.machine
            .declared
            .passes
            .get(self.pass as usize)?
            .scene?
            .bit
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
        let Some(decl) = self.machine.declared.textures.get(texture as usize) else {
            return Allocation::Fresh;
        };
        let at = texture as usize;
        if self.tables.sizes[at] == Some((size, mips)) {
            return Allocation::Kept;
        }
        let made = Made {
            label: decl.label,
            format: decl.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            samples: 1,
        };
        self.tables.textures[at] = Some(Allocated::new(&self.device, &made, size, mips));
        self.tables.sizes[at] = Some((size, mips));
        Allocation::Fresh
    }

    pub(crate) fn release(&mut self, texture: u32) {
        let at = texture as usize;
        if self.tables.textures.get_mut(at).and_then(Option::take).is_some() {
            self.tables.sizes[at] = None;
        }
    }

    pub(crate) fn release_buffer(&mut self, buffer: u32) {
        let Some(held) = self.tables.buffers.get_mut(buffer as usize) else {
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
        self.forget_ran(part);
    }

    pub(crate) fn written(&self) -> FrameId {
        FrameId(self.frames.asked)
    }

    pub(crate) fn refuse(&mut self, error: Error) {
        if self.refused.is_none() {
            self.refused = Some(error);
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
        let made = crate::gpu_calls::create_buffer(&self.device, &wgpu::BufferDescriptor {
            label: Some(decl.label),
            size: bytes,
            usage: decl.usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.tables.buffers[at] = Some((made, bytes));
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
        self.queue.write_declared_buffer(buffer, made, offset, bytes);
    }

    pub(crate) fn write_texture(
        &mut self,
        texture: u32,
        at: wgpu::Origin3d,
        bytes: &[u8],
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        let Some(made) = self.tables.textures.get(texture as usize).and_then(Option::as_ref) else {
            let error = fault(self.path("write_texture"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        self.queue.write_declared_texture(
            texture,
            wgpu::TexelCopyTextureInfo {
                texture: &made.texture,
                mip_level: 0,
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
