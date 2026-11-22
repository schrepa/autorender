use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use bevy_ecs::world::World;
use engine_core::{Asset, Error, ErrorCode, FrameId, Material, MeshFile, PassBit};

use crate::graph::{Bits, Clear, Load, Ops, Plan};
use crate::install::{Bound, PassDecl, Program, Reinstall, Scoped};
use crate::linker::Ty;
use crate::machine::{
    CAMERA_STRIDE, Machine, SetFormat, Target, lit_set, shared,
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

pub(crate) use compile::Resolve;
pub(crate) use draw::DrawRecord;
pub(crate) use recording::{Ended, Recording, Step};
pub(crate) use resources::{BatchEntryRaw, HeldBatches};
pub(crate) use run::{Bindings, DroppedTally, KindTallies, Pipelines, Reading, Run};
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
    Texture(u32),
    Targets,
    Cameras,
}

#[derive(Default)]
pub(crate) struct Holders {
    buffers: Vec<Vec<Holder>>,
    textures: Vec<Vec<Holder>>,
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
            textures: vec![Vec::new(); machine.declared.textures.len()],
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
            Bound::Texture(texture) | Bound::TextureMip(texture, _) => {
                self.textures[*texture as usize].push(holder);
            }
            Bound::Target(_) => self.targets.push(holder),
            Bound::Sampler(_) | Bound::Asset => {}
        }
    }

    fn of_held(&self, held: Held) -> &[Holder] {
        match held {
            Held::Buffer(buffer) => &self.buffers[buffer as usize],
            Held::Texture(texture) => &self.textures[texture as usize],
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

pub(crate) const LABELLED: bool = cfg!(any(not(target_arch = "wasm32"), feature = "dev"));
const GROUPED: bool = LABELLED;

pub(crate) const fn debug_label(name: &'static str) -> Option<&'static str> {
    if LABELLED { Some(name) } else { None }
}

pub(crate) struct Allocated {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) size: wgpu::Extent3d,
    pub(crate) mip_levels: u32,
    default_dimension: wgpu::TextureViewDimension,
    views: Vec<(Option<wgpu::TextureViewDimension>, Option<u32>, wgpu::TextureView)>,
}

pub(crate) struct Made {
    pub(crate) label: &'static str,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) dimension: wgpu::TextureDimension,
    pub(crate) usage: wgpu::TextureUsages,
    pub(crate) samples: u32,
}

fn default_view_dimension(
    dimension: wgpu::TextureDimension,
    size: wgpu::Extent3d,
) -> wgpu::TextureViewDimension {
    match dimension {
        wgpu::TextureDimension::D1 => wgpu::TextureViewDimension::D1,
        wgpu::TextureDimension::D2 if size.depth_or_array_layers > 1 => {
            wgpu::TextureViewDimension::D2Array
        }
        wgpu::TextureDimension::D2 => wgpu::TextureViewDimension::D2,
        wgpu::TextureDimension::D3 => wgpu::TextureViewDimension::D3,
    }
}

impl Allocated {
    fn new(device: &wgpu::Device, made: &Made, size: wgpu::Extent3d, mip_levels: u32) -> Self {
        let texture = crate::gpu_calls::create_texture(device, &wgpu::TextureDescriptor {
            label: Some(made.label),
            size,
            mip_level_count: mip_levels,
            sample_count: made.samples,
            dimension: made.dimension,
            format: made.format,
            usage: made.usage,
            view_formats: &[],
        });
        let default_dimension = default_view_dimension(made.dimension, size);
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: match made.dimension {
                wgpu::TextureDimension::D3 => Some(wgpu::TextureViewDimension::D3),
                wgpu::TextureDimension::D2 if size.depth_or_array_layers > 1 => {
                    Some(wgpu::TextureViewDimension::D2Array)
                }
                wgpu::TextureDimension::D1 | wgpu::TextureDimension::D2 => None,
            },
            ..Default::default()
        });
        Self {
            texture,
            view,
            size,
            mip_levels,
            default_dimension,
            views: Vec::new(),
        }
    }

    fn viewed(&mut self, dimension: Option<wgpu::TextureViewDimension>) -> &wgpu::TextureView {
        if dimension.is_none() || dimension == Some(self.default_dimension) {
            return &self.view;
        }
        let key = (dimension, None);
        let at = match self
            .views
            .iter()
            .position(|(known, level, _)| (*known, *level) == key)
        {
            Some(at) => at,
            None => {
                let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension,
                    ..Default::default()
                });
                self.views.push((dimension, None, view));
                self.views.len() - 1
            }
        };
        &self.views[at].2
    }

    fn viewed_mip(
        &mut self,
        dimension: Option<wgpu::TextureViewDimension>,
        mip: u32,
    ) -> Option<&wgpu::TextureView> {
        if mip >= self.mip_levels {
            return None;
        }
        let key = (dimension, Some(mip));
        let at = match self
            .views
            .iter()
            .position(|(known, level, _)| (*known, *level) == key)
        {
            Some(at) => at,
            None => {
                let view = self.texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension,
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
                self.views.push((dimension, Some(mip), view));
                self.views.len() - 1
            }
        };
        Some(&self.views[at].2)
    }
}

pub(crate) struct Filled {
    pub(crate) base: u32,
    pub(crate) textures: Vec<Asset<engine_core::Texture>>,
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
    pub(crate) planned_at: usize,
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

#[derive(Clone)]
pub(crate) enum Placeholder {
    Texture(wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

#[derive(Default)]
pub(crate) struct Spare {
    colors: Vec<(u64, Option<wgpu::TextureView>, wgpu::TextureFormat, Ops)>,
    attachments: Vec<u64>,
    cameras: Vec<u8>,
    pub(crate) kept: Vec<draw::Kept>,
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
            self.parts[part] = None;
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

    pub(crate) fn allocate_mips(
        &mut self,
        texture: u32,
        size: wgpu::Extent3d,
        mip_levels: u32,
    ) -> Allocation {
        let Some(decl) = self.machine.declared.textures.get(texture as usize) else {
            return Allocation::Fresh;
        };
        let max_dim = match decl.dimension {
            wgpu::TextureDimension::D1 => size.width,
            wgpu::TextureDimension::D2 => size.width.max(size.height),
            wgpu::TextureDimension::D3 => size
                .width
                .max(size.height)
                .max(size.depth_or_array_layers),
        };
        let max_mips = if max_dim == 0 {
            0
        } else {
            32 - max_dim.leading_zeros()
        };
        if size.width == 0
            || size.height == 0
            || size.depth_or_array_layers == 0
            || mip_levels == 0
            || mip_levels > max_mips
        {
            let error = fault(self.path("allocate_mips"), ErrorCode::OutOfRange)
                .allowed(engine_core::Allowed::Range {
                    min: 1.0,
                    max: f64::from(max_mips.max(1)),
                })
                .got(mip_levels);
            self.fault(error);
            return Allocation::Fresh;
        }
        let at = texture as usize;
        if self.tables.textures[at]
            .as_ref()
            .is_some_and(|made| made.size == size && made.mip_levels == mip_levels)
        {
            return Allocation::Kept;
        }
        let made = Made {
            label: decl.label,
            format: decl.format,
            dimension: decl.dimension,
            usage: decl.usage
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            samples: 1,
        };
        self.tables.textures[at] = Some(Allocated::new(&self.device, &made, size, mip_levels));
        self.touch(Held::Texture(texture));
        Allocation::Fresh
    }

    pub(crate) fn take_readback(&mut self, readback: u32) -> Option<Vec<u8>> {
        let owned = self
            .machine
            .declared
            .readbacks
            .get(readback as usize)
            .is_some_and(|decl| decl.part == self.part);
        if !owned {
            let error = fault(self.path("take_readback"), ErrorCode::UnknownVariable).got(readback);
            self.fault(error);
            return None;
        }
        self.reading
            .arrived
            .get_mut(readback as usize)
            .and_then(Option::take)
            .map(|(_, bytes)| bytes)
    }

    pub(crate) fn clear_buffer(&mut self, buffer: u32, range: std::ops::Range<u64>) {
        let Some((buf, size)) = self
            .buffer_held(buffer)
            .map(|(made, bytes)| (made.clone(), *bytes))
        else {
            let error = fault(self.path("clear_buffer"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        if range.start > range.end
            || range.end > size
            || !range.start.is_multiple_of(4)
            || !range.end.is_multiple_of(4)
        {
            let error = fault(self.path("clear_buffer"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        let len = range.end - range.start;
        if len == 0 {
            return;
        }
        self.end_render();
        if let Some(encoder) = self.recording.encoder() {
            encoder.clear_buffer(&buf, range.start, Some(len));
        }
    }

    pub(crate) fn copy_buffer(
        &mut self,
        src: u32,
        src_offset: u64,
        dst: u32,
        dst_offset: u64,
        size: u64,
    ) {
        let (Some((src_buf, src_size)), Some((dst_buf, dst_size))) = (
            self.buffer_held(src)
                .map(|(made, bytes)| (made.clone(), *bytes)),
            self.buffer_held(dst)
                .map(|(made, bytes)| (made.clone(), *bytes)),
        ) else {
            let error = fault(self.path("copy_buffer"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        let src_ok = src_offset.is_multiple_of(4)
            && src_offset
                .checked_add(size)
                .is_some_and(|end| end <= src_size);
        let dst_ok = dst_offset.is_multiple_of(4)
            && dst_offset
                .checked_add(size)
                .is_some_and(|end| end <= dst_size);
        if !size.is_multiple_of(4) || !src_ok || !dst_ok {
            let error = fault(self.path("copy_buffer"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        if size == 0 {
            return;
        }
        if src == dst && src_offset < dst_offset + size && dst_offset < src_offset + size {
            let error = fault(self.path("copy_buffer"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        }
        self.end_render();
        if let Some(encoder) = self.recording.encoder() {
            encoder.copy_buffer_to_buffer(&src_buf, src_offset, &dst_buf, dst_offset, size);
        }
    }

    pub(crate) fn copy_to_readback(
        &mut self,
        readback: u32,
        buffer: u32,
        range: std::ops::Range<u64>,
    ) {
        let owned = self
            .machine
            .declared
            .readbacks
            .get(readback as usize)
            .is_some_and(|decl| decl.part == self.part);
        if !owned {
            let error = fault(
                self.path("copy_to_readback.readback"),
                ErrorCode::UnknownVariable,
            )
            .got(readback);
            self.fault(error);
            return;
        }
        let Some((buf, size)) = self
            .buffer_held(buffer)
            .map(|(made, bytes)| (made.clone(), *bytes))
        else {
            let error = fault(self.path("copy_to_readback.buffer"), ErrorCode::InvalidValue);
            self.fault(error);
            return;
        };
        if range.start > range.end
            || range.end > size
            || !range.start.is_multiple_of(4)
            || !range.end.is_multiple_of(4)
        {
            let error = fault(self.path("copy_to_readback.range"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        if range.start == range.end {
            return;
        }
        self.end_render();
        let (device, written, slots, part) = (
            self.device.clone(),
            self.written(),
            self.reading.slots,
            self.part,
        );
        let Some(encoder) = self.recording.encoder() else {
            return;
        };
        self.reading.list.record(
            (&device, encoder),
            (&buf, range),
            (crate::readback::Reader::Part(part, readback), written),
            slots,
        );
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
            .is_some_and(|(_, known)| *known >= bytes)
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
        self.write_texture_mip(texture, 0, at, bytes, layout, size);
    }

    pub(crate) fn write_texture_mip(
        &mut self,
        texture: u32,
        mip_level: u32,
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
        if mip_level >= made.mip_levels {
            let error = fault(self.path("write_texture_mip"), ErrorCode::OutOfRange);
            self.fault(error);
            return;
        }
        self.queue.write_declared_texture(
            texture,
            wgpu::TexelCopyTextureInfo {
                texture: &made.texture,
                mip_level,
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
    Missing,
}
