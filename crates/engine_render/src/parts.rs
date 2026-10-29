use std::any::{Any, TypeId};
use std::sync::Arc;

use engine_core::{
    Error, ErrorCode, Expected, MeshHandle, RecordKey, RenderFrame, RenderMaterial, RenderMesh,
    RenderSnapshot, RenderStats, RenderTexture,
};

use crate::bindings::SharedGroups;
use crate::cached::Bound;
use crate::entries::{MeshEntries, PassKind};
use crate::faces::{FaceCast, FaceOffer};
use crate::gpu_calls::Recorded;
use crate::gpu_data::LightingUniform;
use crate::graph::{
    Clear, FrameGraph, FrameShape, GraphTable, Load, MAX_COLORS, Node, Ops, PassDecl, ResourceDecl,
    ResourceId, Targets,
};
use crate::pieces::{PieceAt, PieceDecl};
use crate::renderer::Core;
use crate::temporal::TemporalFrame;

#[derive(Clone, Copy, Debug)]
pub struct FrameKind {
    pub gpu_mode: bool,
    pub temporal: bool,
    pub display: bool,
    pub debug: bool,
    pub instance_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Passes {
    pub opaque_kind: PassKind,
    pub impostor_pass: Option<(Targets, bool)>,
}

pub struct FrameInfo<'s> {
    pub snapshot: &'s RenderSnapshot,
    pub frame: RenderFrame<'s>,
    pub kind: FrameKind,
    pub temporal: Option<TemporalFrame>,
    pub lighting: Option<LightingUniform>,
    pub passes: Option<Passes>,
    pub shape: Option<FrameShape>,
    pub kept: u64,
}

pub struct PartAssets<'s> {
    pub meshes: Vec<(MeshHandle, &'s RenderMesh)>,
    pub materials: Vec<&'s RenderMaterial>,
    pub textures: Vec<&'s RenderTexture>,
}

#[derive(Default)]
pub struct Shading {
    pub values: Vec<(&'static str, f64)>,
    pub warm: Vec<(&'static str, f64)>,
}

impl Shading {
    pub fn set(&mut self, name: &'static str, value: f64) {
        self.values.push((name, value));
    }

    pub fn warm(&mut self, name: &'static str, value: f64) {
        self.warm.push((name, value));
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Jitter {
    pub offset: [f32; 2],
    pub sequence: u64,
}

pub struct Views<'a> {
    pub cameras: &'a [wgpu::Buffer],
    pub format: wgpu::TextureFormat,
    pub mirrored: bool,
    pub still: bool,
}

pub type RecordStart = (u64, Vec<RecordKey>, Arc<dyn Any + Send + Sync>);

pub type MadeRecord = (RecordKey, Result<Vec<u8>, String>);

impl FrameInfo<'_> {
    pub fn runs(&self, table: &GraphTable, pass: &str) -> bool {
        table
            .pass_named(pass)
            .is_some_and(|pass| self.kept & (1 << pass.index()) != 0)
    }
}

pub fn scene_target(table: &GraphTable, shape: &FrameShape) -> Option<ResourceId> {
    table.resource(match shape.targets {
        Targets::Canvas => "canvas",
        Targets::Post | Targets::Temporal => "scene",
        Targets::Debug => "debug_ids",
    })
}

pub fn main_targets(table: &GraphTable, shape: &FrameShape) -> Vec<(ResourceId, Clear)> {
    let transparent = Clear::Color(crate::graph::TRANSPARENT);
    let scene = Clear::Color(crate::graph::SCENE_CLEAR);
    let named: Vec<(&str, Clear)> = match shape.targets {
        Targets::Canvas => vec![("canvas", Clear::Color(shape.canvas_clear))],
        Targets::Post => vec![
            ("scene", scene),
            ("glow", transparent),
            ("surface", transparent),
        ],
        Targets::Temporal => vec![
            ("scene", scene),
            ("motion", transparent),
            ("glow", transparent),
            ("surface", transparent),
        ],
        Targets::Debug => vec![
            ("debug_ids", transparent),
            ("debug_count", transparent),
            ("debug_terrain", transparent),
        ],
    };
    named
        .into_iter()
        .filter_map(|(name, clear)| Some((table.resource(name)?, clear)))
        .collect()
}

pub struct Begin<'f> {
    pub core: &'f mut Core,
    pub parts: &'f Parts,
    pub snapshot: &'f RenderSnapshot,
    pub frame: RenderFrame<'f>,
    pub gpu_mode: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BeginOut {
    pub forget_temporal: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UniformsOut {
    pub forget_lighting: bool,
}

pub struct Step<'f> {
    pub core: &'f mut Core,
    pub parts: &'f Parts,
    pub table: &'f GraphTable,
    pub frame: &'f FrameInfo<'f>,
}

impl Step<'_> {}

#[derive(Clone, Copy)]
pub struct View<'f> {
    pub core: &'f Core,
    pub parts: &'f Parts,
    pub table: &'f GraphTable,
    pub frame: &'f FrameInfo<'f>,
}

pub struct NodeViews {
    pub colors: [Option<wgpu::TextureView>; MAX_COLORS],
    pub depth: Option<wgpu::TextureView>,
    pub region: Option<(u32, u32)>,
}

pub struct Encode<'f> {
    pub core: &'f mut Core,
    pub parts: &'f Parts,
    pub table: &'f GraphTable,
    pub frame: &'f FrameInfo<'f>,
    pub node: &'f Node,
    pub views: &'f NodeViews,
}

impl Encode<'_> {
    pub fn view(&self) -> View<'_> {
        View {
            core: &*self.core,
            parts: self.parts,
            table: self.table,
            frame: self.frame,
        }
    }

    pub fn render_pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &'static str,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> wgpu::RenderPass<'e> {
        render_pass(encoder, self.node, self.views, label, timestamp_writes)
    }

    pub fn draw_hook(&self, at: DrawAt<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        self.parts.draw_hook(at, &self.view(), pass)
    }
}

pub fn render_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    node: &Node,
    views: &NodeViews,
    label: &'static str,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> wgpu::RenderPass<'e> {
    let mut colors = [None, None, None, None];
    for ((slot, (_, ops)), view) in colors.iter_mut().zip(node.colors()).zip(&views.colors) {
        *slot = view.as_ref().map(|view| wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: color_operations(ops),
        });
    }
    let count = colors.iter().take_while(|color| color.is_some()).count();
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &colors[..count],
        depth_stencil_attachment: node
            .depth()
            .zip(views.depth.as_ref())
            .map(|((_, ops), view)| wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: ops.map(depth_operations),
                stencil_ops: None,
            }),
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some(region) = views.region {
        crate::targets::draw_within(&mut pass, region);
    }
    pass
}

fn color_operations(ops: Ops) -> wgpu::Operations<wgpu::Color> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Color([r, g, b, a])) => {
                wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a })
            }
            Load::Clear(Clear::Depth(_)) | Load::Load => wgpu::LoadOp::Load,
        },
        store: store_operation(ops),
    }
}

fn depth_operations(ops: Ops) -> wgpu::Operations<f32> {
    wgpu::Operations {
        load: match ops.load {
            Load::Clear(Clear::Depth(depth)) => wgpu::LoadOp::Clear(depth),
            Load::Clear(Clear::Color(_)) | Load::Load => wgpu::LoadOp::Load,
        },
        store: store_operation(ops),
    }
}

fn store_operation(ops: Ops) -> wgpu::StoreOp {
    if ops.store {
        wgpu::StoreOp::Store
    } else {
        wgpu::StoreOp::Discard
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpaqueSlot {
    Prepass,
    Meshes,
    Ground,
    Cover,
    Objects,
    Sky,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hook {
    Opaque(OpaqueSlot),
    ShadowCasters,
    ProbeFace,
}

#[derive(Clone, Copy)]
pub enum DrawAt<'a> {
    Opaque {
        slot: OpaqueSlot,
        phase: usize,
    },
    ShadowCasters {
        map: usize,
        casters: &'a [wgpu::RenderPipeline; 2],
        cards: Option<&'a wgpu::RenderPipeline>,
    },
    ProbeFace {
        face: usize,
        center: glam::Vec3,
        lit: bool,
    },
}

impl DrawAt<'_> {
    pub fn hook(&self) -> Hook {
        match self {
            Self::Opaque { slot, .. } => Hook::Opaque(*slot),
            Self::ShadowCasters { .. } => Hook::ShadowCasters,
            Self::ProbeFace { .. } => Hook::ProbeFace,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PreludeStage {
    Tables,
    Uploads,
    Inputs,
    Derived,
}

#[derive(Default)]
pub struct Prelude {
    encoders: Vec<(PreludeStage, &'static str, wgpu::CommandEncoder)>,
    buffers: Vec<(PreludeStage, &'static str, wgpu::CommandBuffer)>,
    order: Vec<(PreludeStage, &'static str, bool)>,
    spans: Vec<(&'static str, &'static str)>,
    owner: &'static str,
}

impl Prelude {
    pub fn encoder(
        &mut self,
        device: &wgpu::Device,
        stage: PreludeStage,
        span: &'static str,
        label: &'static str,
    ) -> &mut wgpu::CommandEncoder {
        self.note(span);
        let found = self
            .encoders
            .iter()
            .position(|&(known, name, _)| known == stage && name == span);
        let index = found.unwrap_or_else(|| {
            let encoder = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            self.encoders.push((stage, span, encoder));
            self.order.push((stage, span, true));
            self.encoders.len() - 1
        });
        &mut self.encoders[index].2
    }

    pub fn push_encoder(
        &mut self,
        stage: PreludeStage,
        span: &'static str,
        encoder: wgpu::CommandEncoder,
    ) {
        self.note(span);
        self.encoders.push((stage, span, encoder));
        self.order.push((stage, span, true));
    }

    pub fn push_buffer(
        &mut self,
        stage: PreludeStage,
        span: &'static str,
        buffer: wgpu::CommandBuffer,
    ) {
        self.note(span);
        self.buffers.push((stage, span, buffer));
        self.order.push((stage, span, false));
    }

    pub fn reserve(&mut self, span: &'static str) {
        self.note(span);
    }

    pub fn attribute(&mut self, owner: &'static str) {
        self.owner = owner;
    }

    #[cfg(feature = "profiling")]
    pub fn owners(&self) -> &[(&'static str, &'static str)] {
        &self.spans
    }

    fn note(&mut self, span: &'static str) {
        if !self.spans.iter().any(|&(known, _)| known == span) {
            self.spans.push((span, self.owner));
        }
    }

    pub fn take(self) -> Vec<(&'static str, Recorded)> {
        let mut encoders = self.encoders.into_iter();
        let mut buffers = self.buffers.into_iter();
        let spans = self.spans;
        let mut parts: Vec<(PreludeStage, usize, &'static str, Recorded)> = self
            .order
            .into_iter()
            .filter_map(|(stage, span, encoder)| {
                let recorded = if encoder {
                    Recorded::Encoder(encoders.next()?.2)
                } else {
                    Recorded::Buffer(buffers.next()?.2)
                };
                let rank = spans
                    .iter()
                    .position(|&(known, _)| known == span)
                    .unwrap_or(spans.len());
                Some((stage, rank, span, recorded))
            })
            .collect();
        parts.sort_by_key(|&(stage, rank, _, _)| (stage, rank));
        parts
            .into_iter()
            .map(|(_, _, span, recorded)| (span, recorded))
            .collect()
    }
}

pub struct PartDecl {
    pub name: &'static str,
    pub brick: &'static str,
    pub passes: &'static [PassDecl],
    pub resources: &'static [ResourceDecl],
    pub hooks: &'static [Hook],
    pub contributes: &'static [crate::bindings::Contribution],
    pub effects: &'static [crate::composite::PixelEffect],
    pub labels: &'static [&'static str],
    pub lazy: bool,
    pub faces: &'static [FaceOffer],
    pub needs: &'static [&'static str],
    pub piece: Option<PieceDecl>,
}

pub struct Create<'f> {
    pub core: &'f Core,
}

pub struct PartEntry {
    pub decl: PartDecl,
    pub create: fn(&Create<'_>) -> Box<dyn GpuPart>,
}

pub trait GpuPart: Any {
    fn install(&mut self, _table: &GraphTable) {}

    fn begin(&mut self, _cx: &mut Begin<'_>, _out: &mut BeginOut) {}

    fn forget_targets(&mut self, _targets: Targets) {}

    fn temporal_frame(&mut self, _cx: &mut Step<'_>) -> Option<TemporalFrame> {
        None
    }

    fn uniforms(
        &mut self,
        _cx: &mut Step<'_>,
        _lighting: &mut LightingUniform,
        _out: &mut UniformsOut,
    ) {
    }

    fn prepare(&mut self, _cx: &mut Step<'_>, _prelude: &mut Prelude) {}

    fn prepare_draws(&mut self, _cx: &mut Step<'_>, _encoder: &mut wgpu::CommandEncoder) {}

    fn defers(&self) -> bool {
        false
    }

    fn finish(&mut self, _cx: &mut Step<'_>) {}

    fn declare(&self, _cx: &View<'_>, _graph: &mut FrameGraph) {}

    fn encode(
        &mut self,
        _pass: usize,
        _cx: &mut Encode<'_>,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        0
    }

    fn draw(&self, _at: DrawAt<'_>, _cx: &View<'_>, _pass: &mut wgpu::RenderPass<'_>) -> u32 {
        0
    }

    fn attachment(&self, _resource: &str) -> Option<wgpu::TextureView> {
        None
    }

    fn contribute<'s>(&'s self, _group: &str, _out: &mut Vec<(&'static str, Bound<'s>)>) {}

    fn after_submit(&mut self) {}

    fn capacity_bytes(&self) -> u64 {
        0
    }

    fn stats(&self, _cx: &View<'_>, _stats: &mut RenderStats) {}

    fn poll(&mut self) -> bool {
        false
    }

    fn disable(&mut self) -> bool {
        false
    }

    fn disabled(&self) -> bool {
        false
    }

    #[cfg(feature = "dev")]
    fn inject(&mut self) -> bool {
        false
    }

    #[cfg(feature = "dev")]
    fn encode_fault(&self, _encoder: &mut wgpu::CommandEncoder) {}

    fn set_cooking(&mut self, _cooking: bool) {}

    fn shading(&self, _snapshot: &RenderSnapshot, _shading: &mut Shading) {}

    fn place_piece(&mut self, _at: PieceAt) {}

    fn piece(&self) -> Option<&[u8]> {
        None
    }

    fn produces(&self, _resource: ResourceId) -> bool {
        false
    }

    fn wants(&self, _resource: ResourceId) -> bool {
        false
    }

    fn jitter(&mut self, _cx: &mut Step<'_>) -> Option<Jitter> {
        None
    }

    fn asset_generation(&self) -> u64 {
        0
    }

    fn assets<'s>(&self, _snapshot: &'s RenderSnapshot, _assets: &mut PartAssets<'s>) {}

    fn batch_group(&self, _batch: usize) -> Option<u32> {
        None
    }

    fn views(&self) -> Option<Views<'_>> {
        None
    }

    fn mesh_entries(&self) -> Option<&'static MeshEntries> {
        None
    }

    fn start_records(&mut self, _starts: &mut Vec<RecordStart>) {}

    fn take_records(&mut self) -> Vec<MadeRecord> {
        Vec::new()
    }

    fn records_left(&self) -> Option<(&'static str, u64)> {
        None
    }
}

struct Slot {
    entry: &'static PartEntry,
    part: Option<Box<dyn GpuPart>>,
}

struct Offered {
    face: TypeId,
    slot: usize,
    cast: Box<dyn Any + Send + Sync>,
}

#[derive(Default)]
pub struct Parts {
    slots: Vec<Slot>,
    names: &'static [&'static str],
    offered: Vec<Offered>,
}

impl Parts {
    pub fn install(
        &mut self,
        entries: &[&'static PartEntry],
        core: &Core,
    ) -> Result<GraphTable, Error> {
        if entries.len() > 64 {
            return Err(Error::new(ErrorCode::RenderFailed, "$.parts")
                .at_most(64)
                .got(entries.len()));
        }
        let mut kept = std::mem::take(&mut self.slots);
        for &entry in entries {
            if let Some(twice) = entries
                .iter()
                .filter(|other| other.decl.name == entry.decl.name)
                .nth(1)
            {
                return Err(Error::new(
                    ErrorCode::DuplicateBrick,
                    format!("$.parts.{}", twice.decl.name),
                )
                .expected(Expected::Unique)
                .got(twice.decl.brick));
            }
            let reused = kept
                .iter()
                .position(|slot| std::ptr::eq(slot.entry, entry))
                .map(|index| kept.swap_remove(index));
            let slot = match reused {
                Some(slot) => slot,
                None => {
                    let _owner = crate::memory::owned(entry.decl.brick);
                    Slot {
                        entry,
                        part: Some((entry.create)(&Create { core })),
                    }
                }
            };
            self.slots.push(slot);
        }
        self.offered.clear();
        for (index, slot) in self.slots.iter().enumerate() {
            for offer in slot.entry.decl.faces {
                let face = (offer.face)();
                if let Some(first) = self.offered.iter().find(|known| known.face == face) {
                    return Err(Error::new(
                        ErrorCode::SlotFilledTwice,
                        format!("$.faces.{}", offer.name()),
                    )
                    .expected(Expected::Unique)
                    .got(slot.entry.decl.brick)
                    .nearest(self.slots[first.slot].entry.decl.brick));
                }
                self.offered.push(Offered {
                    face,
                    slot: index,
                    cast: (offer.cast)(),
                });
            }
        }
        let mut table = GraphTable::default();
        for (index, slot) in self.slots.iter().enumerate() {
            let decl = &slot.entry.decl;
            table.add_part(index, decl.brick, decl.passes, decl.resources)?;
        }
        for slot in &mut self.slots {
            if let Some(part) = slot.part.as_mut() {
                part.install(&table);
            }
        }
        let names: Vec<&'static str> = self.slots.iter().map(|slot| slot.entry.decl.name).collect();
        self.names = Box::leak(names.into_boxed_slice());
        Ok(table)
    }

    pub fn get<T: GpuPart>(&self) -> Option<&T> {
        self.slots.iter().find_map(|slot| {
            let part: &dyn Any = slot.part.as_deref()?;
            part.downcast_ref::<T>()
        })
    }

    pub fn get_mut<T: GpuPart>(&mut self) -> Option<&mut T> {
        self.slots.iter_mut().find_map(|slot| {
            let part: &mut dyn Any = slot.part.as_deref_mut()?;
            part.downcast_mut::<T>()
        })
    }

    pub fn face<F: ?Sized + 'static>(&self) -> Option<&F> {
        let face = TypeId::of::<F>();
        let offered = self.offered.iter().find(|offered| offered.face == face)?;
        let cast = offered.cast.downcast_ref::<FaceCast<F>>()?;
        let part: &dyn Any = self.slots.get(offered.slot)?.part.as_deref()?;
        (cast.get)(part)
    }

    pub fn face_mut<F: ?Sized + 'static>(&mut self) -> Option<&mut F> {
        let face = TypeId::of::<F>();
        let offered = self.offered.iter().find(|offered| offered.face == face)?;
        let get_mut = offered.cast.downcast_ref::<FaceCast<F>>()?.get_mut;
        let slot = offered.slot;
        let part: &mut dyn Any = self.slots.get_mut(slot)?.part.as_deref_mut()?;
        get_mut(part)
    }

    pub fn offers(&self, brick: &str) -> Vec<&'static str> {
        self.slots
            .iter()
            .filter(|slot| slot.entry.decl.brick == brick)
            .flat_map(|slot| slot.entry.decl.faces.iter().map(FaceOffer::name))
            .collect()
    }

    pub fn produces(&self, resource: impl Into<Option<ResourceId>>) -> bool {
        let Some(resource) = resource.into() else {
            return false;
        };
        self.slots
            .iter()
            .filter_map(|slot| slot.part.as_deref())
            .any(|part| part.produces(resource))
    }

    pub fn set_cooking(&mut self, cooking: bool) {
        for slot in &mut self.slots {
            if let Some(part) = slot.part.as_mut() {
                part.set_cooking(cooking);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn decl(&self, index: usize) -> &'static PartDecl {
        &self.slots[index].entry.decl
    }

    pub fn names(&self) -> &'static [&'static str] {
        self.names
    }

    pub fn part(&self, index: usize) -> Option<&dyn GpuPart> {
        self.slots.get(index)?.part.as_deref()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static PartDecl, &dyn GpuPart)> {
        self.slots
            .iter()
            .filter_map(|slot| Some((&slot.entry.decl, slot.part.as_deref()?)))
    }

    pub fn with_part<R>(
        &mut self,
        index: usize,
        act: impl FnOnce(&mut dyn GpuPart, &Parts) -> R,
    ) -> Option<R> {
        let mut part = self.slots.get_mut(index)?.part.take()?;
        let result = act(part.as_mut(), self);
        self.slots[index].part = Some(part);
        Some(result)
    }

    pub fn draw_hook(&self, at: DrawAt<'_>, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        let hook = at.hook();
        self.slots
            .iter()
            .filter(|slot| slot.entry.decl.hooks.contains(&hook))
            .filter_map(|slot| slot.part.as_deref())
            .map(|part| part.draw(at, cx, pass))
            .sum()
    }

    pub fn named_by(&self, message: &str) -> u64 {
        let mut named = 0;
        for (index, slot) in self.slots.iter().enumerate() {
            if slot
                .entry
                .decl
                .labels
                .iter()
                .any(|label| message.contains(label))
            {
                if named != 0 {
                    return 0;
                }
                named = 1 << index;
            }
        }
        named
    }

    pub fn disable(&mut self, named: u64) -> u64 {
        let mut disabled = 0;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let Some(part) = slot.part.as_mut() else {
                continue;
            };
            let bit = 1 << index;
            if part.poll() || (named & bit != 0 && slot.entry.decl.lazy && part.disable()) {
                disabled |= bit;
            }
        }
        disabled
    }

    pub fn disabled(&self) -> u64 {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.part.as_deref().is_some_and(|part| part.disabled()))
            .fold(0, |bits, (index, _)| bits | (1 << index))
    }

    #[cfg(feature = "dev")]
    pub fn lazy_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.slots
            .iter()
            .filter(|slot| slot.entry.decl.lazy)
            .map(|slot| slot.entry.decl.name)
    }

    #[cfg(feature = "dev")]
    pub fn inject(&mut self, name: &str) -> bool {
        self.slots
            .iter_mut()
            .find(|slot| slot.entry.decl.name == name)
            .and_then(|slot| slot.part.as_mut())
            .is_some_and(|part| part.inject())
    }

    pub fn after_submit(&mut self) {
        for slot in &mut self.slots {
            if let Some(part) = slot.part.as_mut() {
                let _owner = crate::memory::owned(slot.entry.decl.brick);
                part.after_submit();
            }
        }
    }
}

pub const CORE_BRICK: &str = "core";

pub fn standard() -> [&'static PartEntry; 17] {
    [
        &crate::gpu_presentation::PART,
        &crate::renderer::OPAQUE_PART,
        &crate::depth_pyramid::PART,
        &crate::terrain::PART,
        &crate::scatter::PART,
        &crate::occupancy::PART,
        &crate::shadows::PART,
        &crate::lights::PART,
        &crate::sky::PART,
        &crate::probes::PART,
        &crate::ground_cover::PART,
        &crate::water::PART,
        &crate::post::PART,
        &crate::transparent::PART,
        &crate::temporal::PART,
        &crate::exposure::PART,
        &crate::debug_view::PART,
    ]
}

pub fn installed(bricks: &engine_core::BrickRegistry) -> Vec<&'static PartEntry> {
    standard()
        .into_iter()
        .filter(|entry| entry.decl.brick == CORE_BRICK || bricks.has(entry.decl.brick))
        .collect()
}

pub fn shared_of(entries: &[&'static PartEntry]) -> Result<SharedGroups, Error> {
    let mut shared = SharedGroups::default();
    for entry in entries {
        for contribution in entry.decl.contributes {
            shared.add(entry.decl.brick, contribution)?;
        }
    }
    Ok(shared)
}

pub fn standard_passes() -> Vec<&'static str> {
    standard()
        .iter()
        .flat_map(|entry| entry.decl.passes)
        .map(|pass| pass.name)
        .collect()
}

pub fn standard_shared() -> SharedGroups {
    shared_of(&standard()).unwrap_or_default()
}

pub fn standard_features() -> Vec<(&'static str, bool)> {
    let mut features: Vec<(&'static str, bool)> = Vec::new();
    for entry in standard() {
        if !features.iter().any(|&(brick, _)| brick == entry.decl.brick) {
            features.push((entry.decl.brick, true));
        }
    }
    features
}
