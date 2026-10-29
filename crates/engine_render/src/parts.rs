use std::any::Any;

use engine_core::{Error, ErrorCode, Expected, RenderFrame, RenderSnapshot, RenderStats};

use crate::cached::Bound;
use crate::entries::PassKind;
use crate::gpu_calls::Recorded;
use crate::gpu_data::LightingUniform;
use crate::graph::{
    Clear, FrameGraph, FrameShape, GraphTable, Load, MAX_COLORS, Node, Ops, PassDecl,
    ResourceDecl, ResourceId, Targets,
};
use crate::renderer::Core;
use crate::temporal::TemporalFrame;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameKind {
    pub gpu_mode: bool,
    pub temporal: bool,
    pub display: bool,
    pub debug: bool,
    pub instance_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Passes {
    pub opaque_kind: PassKind,
    pub impostor_pass: Option<(Targets, bool)>,
}

pub(crate) struct FrameInfo<'s> {
    pub snapshot: &'s RenderSnapshot,
    pub frame: RenderFrame<'s>,
    pub kind: FrameKind,
    pub temporal: Option<TemporalFrame>,
    pub lighting: Option<LightingUniform>,
    pub passes: Option<Passes>,
    pub shape: Option<FrameShape>,
}

pub(crate) fn scene_target(table: &GraphTable, shape: &FrameShape) -> Option<ResourceId> {
    table.resource(match shape.targets {
        Targets::Canvas => "canvas",
        Targets::Post | Targets::Temporal => "scene",
        Targets::Debug => "debug_ids",
    })
}

pub(crate) fn main_targets(table: &GraphTable, shape: &FrameShape) -> Vec<(ResourceId, Clear)> {
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

pub(crate) struct Begin<'f> {
    pub core: &'f mut Core,
    pub parts: &'f Parts,
    pub snapshot: &'f RenderSnapshot,
    pub frame: RenderFrame<'f>,
    pub gpu_mode: bool,
    pub display: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BeginOut {
    pub forget_temporal: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UniformsOut {
    pub forget_lighting: bool,
}

pub(crate) struct Step<'f> {
    pub core: &'f mut Core,
    pub parts: &'f Parts,
    pub table: &'f GraphTable,
    pub frame: &'f FrameInfo<'f>,
}

impl Step<'_> {
    pub(crate) fn view(&self) -> View<'_> {
        View {
            core: &*self.core,
            parts: self.parts,
            table: self.table,
            frame: self.frame,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct View<'f> {
    pub core: &'f Core,
    pub parts: &'f Parts,
    pub table: &'f GraphTable,
    pub frame: &'f FrameInfo<'f>,
}

pub(crate) struct NodeViews {
    pub colors: [Option<wgpu::TextureView>; MAX_COLORS],
    pub depth: Option<wgpu::TextureView>,
    pub region: Option<(u32, u32)>,
}

pub(crate) struct Encode<'f> {
    pub core: &'f mut Core,
    pub parts: &'f Parts,
    pub table: &'f GraphTable,
    pub frame: &'f FrameInfo<'f>,
    pub node: &'f Node,
    pub views: &'f NodeViews,
}

impl Encode<'_> {
    pub(crate) fn view(&self) -> View<'_> {
        View {
            core: &*self.core,
            parts: self.parts,
            table: self.table,
            frame: self.frame,
        }
    }

    pub(crate) fn render_pass<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &'static str,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) -> wgpu::RenderPass<'e> {
        render_pass(encoder, self.node, self.views, label, timestamp_writes)
    }

    pub(crate) fn draw_hook(&self, at: DrawAt<'_>, pass: &mut wgpu::RenderPass<'_>) -> u32 {
        self.parts.draw_hook(at, &self.view(), pass)
    }
}

pub(crate) fn render_pass<'e>(
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
        depth_stencil_attachment: node.depth().zip(views.depth.as_ref()).map(
            |((_, ops), view)| wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: ops.map(depth_operations),
                stencil_ops: None,
            },
        ),
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
pub(crate) enum OpaqueSlot {
    Prepass,
    Meshes,
    Ground,
    Cover,
    Objects,
    Sky,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hook {
    Opaque(OpaqueSlot),
    ShadowCasters,
    ProbeFace,
}

#[derive(Clone, Copy)]
pub(crate) enum DrawAt<'a> {
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
    pub(crate) fn hook(&self) -> Hook {
        match self {
            Self::Opaque { slot, .. } => Hook::Opaque(*slot),
            Self::ShadowCasters { .. } => Hook::ShadowCasters,
            Self::ProbeFace { .. } => Hook::ProbeFace,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum PreludeStage {
    Tables,
    Uploads,
    Inputs,
    Derived,
}

#[derive(Default)]
pub(crate) struct Prelude {
    encoders: Vec<(PreludeStage, &'static str, wgpu::CommandEncoder)>,
    buffers: Vec<(PreludeStage, &'static str, wgpu::CommandBuffer)>,
    order: Vec<(PreludeStage, &'static str, bool)>,
}

impl Prelude {
    pub(crate) fn encoder(
        &mut self,
        device: &wgpu::Device,
        stage: PreludeStage,
        span: &'static str,
        label: &'static str,
    ) -> &mut wgpu::CommandEncoder {
        let found = self
            .encoders
            .iter()
            .position(|&(known, name, _)| known == stage && name == span);
        let index = found.unwrap_or_else(|| {
            let encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            self.encoders.push((stage, span, encoder));
            self.order.push((stage, span, true));
            self.encoders.len() - 1
        });
        &mut self.encoders[index].2
    }

    pub(crate) fn push_encoder(
        &mut self,
        stage: PreludeStage,
        span: &'static str,
        encoder: wgpu::CommandEncoder,
    ) {
        self.encoders.push((stage, span, encoder));
        self.order.push((stage, span, true));
    }

    pub(crate) fn push_buffer(
        &mut self,
        stage: PreludeStage,
        span: &'static str,
        buffer: wgpu::CommandBuffer,
    ) {
        self.buffers.push((stage, span, buffer));
        self.order.push((stage, span, false));
    }

    pub(crate) fn take(self) -> Vec<(&'static str, Recorded)> {
        let mut encoders = self.encoders.into_iter();
        let mut buffers = self.buffers.into_iter();
        let mut parts: Vec<(PreludeStage, &'static str, Recorded)> = self
            .order
            .into_iter()
            .filter_map(|(stage, span, encoder)| {
                let recorded = if encoder {
                    Recorded::Encoder(encoders.next()?.2)
                } else {
                    Recorded::Buffer(buffers.next()?.2)
                };
                Some((stage, span, recorded))
            })
            .collect();
        parts.sort_by_key(|&(stage, _, _)| stage);
        parts
            .into_iter()
            .map(|(_, span, recorded)| (span, recorded))
            .collect()
    }
}

pub(crate) struct PartDecl {
    pub name: &'static str,
    pub brick: &'static str,
    pub passes: &'static [PassDecl],
    pub resources: &'static [ResourceDecl],
    pub hooks: &'static [Hook],
    pub contributes: &'static [crate::bindings::Contribution],
    pub effects: &'static [crate::composite::PixelEffect],
    pub labels: &'static [&'static str],
    pub lazy: bool,
}

pub(crate) struct Create<'f> {
    pub core: &'f Core,
    pub parts: &'f Parts,
}

pub(crate) struct PartEntry {
    pub decl: PartDecl,
    pub create: fn(&Create<'_>) -> Box<dyn GpuPart>,
}

pub(crate) trait GpuPart: Any {
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
}

struct Slot {
    entry: &'static PartEntry,
    part: Option<Box<dyn GpuPart>>,
}

#[derive(Default)]
pub(crate) struct Parts {
    slots: Vec<Slot>,
    names: &'static [&'static str],
}

impl Parts {
    pub(crate) fn install(
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
                None => Slot {
                    entry,
                    part: Some((entry.create)(&Create { core, parts: self })),
                },
            };
            self.slots.push(slot);
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

    pub(crate) fn get<T: GpuPart>(&self) -> Option<&T> {
        self.slots.iter().find_map(|slot| {
            let part: &dyn Any = slot.part.as_deref()?;
            part.downcast_ref::<T>()
        })
    }

    pub(crate) fn get_mut<T: GpuPart>(&mut self) -> Option<&mut T> {
        self.slots.iter_mut().find_map(|slot| {
            let part: &mut dyn Any = slot.part.as_deref_mut()?;
            part.downcast_mut::<T>()
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn decl(&self, index: usize) -> &'static PartDecl {
        &self.slots[index].entry.decl
    }

    pub(crate) fn names(&self) -> &'static [&'static str] {
        self.names
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&'static PartDecl, &dyn GpuPart)> {
        self.slots
            .iter()
            .filter_map(|slot| Some((&slot.entry.decl, slot.part.as_deref()?)))
    }

    pub(crate) fn with_part<R>(
        &mut self,
        index: usize,
        act: impl FnOnce(&mut dyn GpuPart, &Parts) -> R,
    ) -> Option<R> {
        let mut part = self.slots.get_mut(index)?.part.take()?;
        let result = act(part.as_mut(), self);
        self.slots[index].part = Some(part);
        Some(result)
    }

    pub(crate) fn each(&mut self, mut act: impl FnMut(&mut dyn GpuPart, &Parts)) {
        for index in 0..self.slots.len() {
            self.with_part(index, &mut act);
        }
    }

    pub(crate) fn draw_hook(
        &self,
        at: DrawAt<'_>,
        cx: &View<'_>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> u32 {
        let hook = at.hook();
        self.slots
            .iter()
            .filter(|slot| slot.entry.decl.hooks.contains(&hook))
            .filter_map(|slot| slot.part.as_deref())
            .map(|part| part.draw(at, cx, pass))
            .sum()
    }

    pub(crate) fn named_by(&self, message: &str) -> u64 {
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

    pub(crate) fn disable(&mut self, named: u64) -> u64 {
        let mut disabled = 0;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let Some(part) = slot.part.as_mut() else {
                continue;
            };
            let bit = 1 << index;
            if part.poll() || (named & bit != 0 && part.disable()) {
                disabled |= bit;
            }
        }
        disabled
    }

    pub(crate) fn disabled(&self) -> u64 {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.part.as_deref().is_some_and(|part| part.disabled()))
            .fold(0, |bits, (index, _)| bits | (1 << index))
    }

    #[cfg(feature = "dev")]
    pub(crate) fn lazy_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.slots
            .iter()
            .filter(|slot| slot.entry.decl.lazy)
            .map(|slot| slot.entry.decl.name)
    }

    #[cfg(feature = "dev")]
    pub(crate) fn inject(&mut self, name: &str) -> bool {
        self.slots
            .iter_mut()
            .find(|slot| slot.entry.decl.name == name)
            .and_then(|slot| slot.part.as_mut())
            .is_some_and(|part| part.inject())
    }

    pub(crate) fn after_submit(&mut self) {
        for slot in &mut self.slots {
            if let Some(part) = slot.part.as_mut() {
                part.after_submit();
            }
        }
    }
}
