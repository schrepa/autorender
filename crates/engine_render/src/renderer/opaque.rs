use super::*;
use crate::debug_view::{DebugViewPass, DrawKind, DrawRecord, bind_record};
use crate::graph::{
    Access, Clear, FAR, FrameShape, Lifetime, PassBuilder, PassDecl, PassId, ResourceDecl,
    ResourceId, Stage,
};
use crate::parts::{Create, DrawAt, Encode, OpaqueSlot, PartDecl, PartEntry, View};
use crate::temporal::TemporalPass;

pub(crate) static OPAQUE_PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "opaque",
        brick: crate::parts::CORE_BRICK,
        passes: &[
            PassDecl {
                name: "opaque",
                stage: Stage::Opaque,
            },
            PassDecl {
                name: "opaque_late",
                stage: Stage::OpaqueLate,
            },
        ],
        resources: &[
            ResourceDecl {
                name: "canvas",
                lifetime: Lifetime::Output,
            },
            ResourceDecl {
                name: "depth",
                lifetime: Lifetime::Frame,
            },
        ],
        hooks: &[],
        contributes: &[],
        effects: &[],
        labels: &[],
        lazy: false,
    },
    create: create_opaque,
};

fn create_opaque(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(OpaquePart::default())
}

const SLOTS: [OpaqueSlot; 6] = [
    OpaqueSlot::Prepass,
    OpaqueSlot::Meshes,
    OpaqueSlot::Ground,
    OpaqueSlot::Cover,
    OpaqueSlot::Objects,
    OpaqueSlot::Sky,
];

#[derive(Clone, Copy, Default)]
pub(crate) struct OpaqueReads {
    pub(crate) depth: Option<ResourceId>,
    pub(crate) poses: Option<ResourceId>,
    pub(crate) visible: Option<ResourceId>,
    placed: Option<ResourceId>,
    occupancy: Option<ResourceId>,
    shadow_maps: Option<ResourceId>,
    lights: Option<ResourceId>,
    probes: Option<ResourceId>,
    bounce: Option<ResourceId>,
    cloud_dome: Option<ResourceId>,
    cover: Option<ResourceId>,
    leaves: Option<ResourceId>,
}

impl OpaqueReads {
    pub(crate) fn of(table: &GraphTable) -> Self {
        Self {
            depth: table.resource("depth"),
            poses: table.resource("poses"),
            visible: table.resource("visible"),
            placed: table.resource("placed"),
            occupancy: table.resource("occupancy"),
            shadow_maps: table.resource("shadow_maps"),
            lights: table.resource("lights"),
            probes: table.resource("probes"),
            bounce: table.resource("bounce"),
            cloud_dome: table.resource("cloud_dome"),
            cover: table.resource("cover"),
            leaves: table.resource("leaves"),
        }
    }

    pub(crate) fn declare_phase(
        &self,
        table: &GraphTable,
        shape: &FrameShape,
        builder: PassBuilder<'_>,
        phase: usize,
    ) {
        let first = phase == 0;
        let forward = first && !shape.deferred;
        let mut opaque = builder
            .read(self.poses)
            .read_written(self.visible)
            .read_written_if(first, self.placed)
            .read(self.occupancy)
            .read_written(self.shadow_maps)
            .read_written(self.lights)
            .read(self.probes)
            .read(self.bounce)
            .read(self.cloud_dome)
            .read_if(forward, self.cover)
            .read_written_if(forward, self.leaves);
        for (target, clear) in crate::parts::main_targets(table, shape) {
            opaque = opaque.color(target, Access::Draw(clear));
        }
        if let Some(depth) = self.depth {
            let _opaque = opaque.depth(depth, Access::Draw(Clear::Depth(FAR)));
        }
    }

    fn declare_late(&self, table: &GraphTable, shape: &FrameShape, builder: PassBuilder<'_>) {
        let mut late = builder
            .read_written(self.placed)
            .read(self.occupancy)
            .read_written(self.shadow_maps)
            .read_written(self.lights)
            .read(self.probes)
            .read(self.bounce)
            .read(self.cloud_dome)
            .read(self.cover);
        for (target, _) in crate::parts::main_targets(table, shape) {
            late = late.color(target, Access::Load);
        }
        if let Some(depth) = self.depth {
            let _late = late.depth(depth, Access::Load);
        }
    }
}

#[derive(Default)]
struct OpaquePart {
    opaque: Option<PassId>,
    late: Option<PassId>,
    reads: OpaqueReads,
}

impl GpuPart for OpaquePart {
    fn install(&mut self, table: &GraphTable) {
        self.opaque = table.pass_named("opaque");
        self.late = table.pass_named("opaque_late");
        self.reads = OpaqueReads::of(table);
    }

    fn declare(&self, cx: &View<'_>, graph: &mut crate::graph::FrameGraph) {
        let Some(shape) = cx.frame.shape else {
            return;
        };
        if let Some(opaque) = self.opaque {
            self.reads.declare_phase(cx.table, &shape, graph.add(cx.table, opaque), 0);
        }
        if shape.deferred
            && let Some(late) = self.late
        {
            self.reads.declare_late(cx.table, &shape, graph.add(cx.table, late));
        }
    }

    fn encode(
        &mut self,
        pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        if pass == 0 {
            encode_phase(cx, encoder, 0)
        } else {
            encode_late(cx, encoder)
        }
    }
}

pub(crate) fn encode_phase(
    cx: &Encode<'_>,
    encoder: &mut wgpu::CommandEncoder,
    phase: usize,
) -> u32 {
    let Some(shape) = cx.frame.shape else {
        return 0;
    };
    let phases = usize::from(shape.phases);
    let forward = phase > 0 || !shape.deferred;
    let (begins, ends) = (phase == 0, forward && phase + 1 == phases);
    let timestamp_writes = (begins || ends)
        .then(|| cx.core.profiler.span("opaque_meshes"))
        .flatten()
        .map(|span| span.render_part(begins, ends));
    let mut pass = cx.render_pass(encoder, "engine/opaque-meshes", timestamp_writes);
    let (first, last) = (phase == 0, phase + 1 == phases);
    draw_slots(cx, &mut pass, phase, |slot| match slot {
        OpaqueSlot::Prepass | OpaqueSlot::Meshes => true,
        OpaqueSlot::Ground | OpaqueSlot::Cover | OpaqueSlot::Objects => forward && first,
        OpaqueSlot::Sky => forward && last,
    })
}

fn encode_late(cx: &Encode<'_>, encoder: &mut wgpu::CommandEncoder) -> u32 {
    let Some(shape) = cx.frame.shape else {
        return 0;
    };
    let ends = shape.phases == 1;
    let timestamp_writes = ends
        .then(|| cx.core.profiler.span("opaque_meshes"))
        .flatten()
        .map(|span| span.render_part(false, true));
    let mut pass = cx.render_pass(encoder, "engine/opaque-meshes", timestamp_writes);
    draw_slots(cx, &mut pass, 0, |slot| match slot {
        OpaqueSlot::Cover | OpaqueSlot::Objects => true,
        OpaqueSlot::Sky => ends,
        OpaqueSlot::Prepass | OpaqueSlot::Meshes | OpaqueSlot::Ground => false,
    })
}

fn draw_slots(
    cx: &Encode<'_>,
    pass: &mut wgpu::RenderPass<'_>,
    phase: usize,
    drawn: impl Fn(OpaqueSlot) -> bool,
) -> u32 {
    let view = cx.view();
    let instances = cx.frame.kind.instance_count > 0;
    let mut draws = 0;
    for slot in SLOTS {
        if !drawn(slot) {
            continue;
        }
        draws += match slot {
            OpaqueSlot::Prepass => cx.core.draw_prepass(&view, pass, phase),
            OpaqueSlot::Meshes if instances => cx.core.draw_opaque(&view, pass, phase),
            OpaqueSlot::Meshes
            | OpaqueSlot::Ground
            | OpaqueSlot::Cover
            | OpaqueSlot::Objects
            | OpaqueSlot::Sky => 0,
        };
        draws += cx.draw_hook(DrawAt::Opaque { slot, phase }, pass);
    }
    draws
}

pub(crate) fn debug_records<'a>(cx: &View<'a>) -> Option<&'a DebugViewPass> {
    cx.frame
        .shape
        .is_some_and(|shape| shape.targets == Targets::Debug)
        .then(|| cx.parts.get::<DebugViewPass>())
        .flatten()
}

impl Core {
    pub(crate) fn draw_prepass(
        &self,
        cx: &View<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        phase: usize,
    ) -> u32 {
        if cx.frame.kind.instance_count > 0 && !self.draw_order.masked.is_empty() {
            self.draw_meshes(cx, pass, phase, true)
        } else {
            0
        }
    }

    fn draw_opaque(&self, cx: &View<'_>, pass: &mut wgpu::RenderPass<'_>, phase: usize) -> u32 {
        let mut draws = self.draw_meshes(cx, pass, phase, false);
        let debug = debug_records(cx);
        let casts = cx.frame.runs(cx.table, "shadow_maps");
        if cx.frame.kind.gpu_mode
            && let Some((targets, lit)) = cx.frame.passes.and_then(|passes| passes.impostor_pass)
            && let (Some(pipeline), Some(group)) = (
                self.impostors.pipelines.pipeline(targets, lit),
                self.impostors.atlas.group(),
            )
        {
            let (ids, args, offset) = self.gpu_presentation.impostor_source(phase);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, group, &[]);
            pass.set_vertex_buffer(0, ids.slice(..));
            bind_record(
                pass,
                debug,
                DrawRecord::of(DrawKind::Card, LOD_LEVELS, casts),
            );
            pass.draw_indirect(args, offset);
            draws += 1;
            if phase == 0 {
                let (cards, args, offset) = self.gpu_presentation.fading_cards();
                pass.set_vertex_buffer(0, cards);
                pass.draw_indirect(args, offset);
                draws += 1;
            }
        }
        draws
    }

    pub(crate) fn draw_meshes(
        &self,
        cx: &View<'_>,
        pass: &mut wgpu::RenderPass<'_>,
        phase: usize,
        prepass: bool,
    ) -> u32 {
        let gpu_mode = cx.frame.kind.gpu_mode;
        let Some(passes) = cx.frame.passes else {
            return 0;
        };
        let opaque_kind = passes.opaque_kind;
        let temporal_pass = cx
            .frame
            .temporal
            .as_ref()
            .zip(cx.parts.get::<TemporalPass>());
        let lit = opaque_kind.lit && !prepass;
        pass.set_bind_group(0, self.shared_group("frame", &["refraction"]), &[]);
        if lit {
            pass.set_bind_group(3, self.shared_group("lights", &[]), &[]);
        }
        pass.set_bind_group(
            2,
            temporal_pass.map_or_else(
                || self.instances.pose_group(0),
                |(frame, temporal)| temporal.pose_group(frame),
            ),
            &[],
        );
        if !gpu_mode {
            pass.set_vertex_buffer(1, self.instances.slots().slice(..));
        }
        let debug = debug_records(cx);
        let casts = cx.frame.runs(cx.table, "shadow_maps");
        let mut draws = 0;
        let mut binds = BatchBinds::default();
        let order = if prepass {
            &self.draw_order.masked
        } else {
            &self.draw_order.order
        };
        for &index in order {
            let index = index as usize;
            let batch = &self.plan.batches[index];
            let bound = if prepass {
                self.bind_prepass(pass, &mut binds, opaque_kind, batch)
            } else {
                Some(self.bind_batch(pass, &mut binds, (opaque_kind, false), batch))
            };
            let Some((mesh, draw)) = bound else {
                continue;
            };
            if gpu_mode {
                for level in 0..lod_levels(mesh) {
                    if level > 0 && mesh.plan[level - 1].1 == 0 {
                        continue;
                    }
                    let (ids, args, offset) = self
                        .gpu_presentation
                        .draw_source(batch, index, phase, level);
                    binds.bindings.ids(pass, 1, ids);
                    bind_record(pass, debug, DrawRecord::of(DrawKind::Mesh, level, casts));
                    let timed = self
                        .profiler
                        .begin_draw(pass, || draw_label(batch, (phase, Some(level)), prepass));
                    pass.draw_indexed_indirect(args, offset);
                    self.profiler.end_draw(pass, timed);
                    draws += 1;
                }
            } else {
                bind_record(pass, debug, DrawRecord::of(DrawKind::Mesh, 0, casts));
                let timed = self
                    .profiler
                    .begin_draw(pass, || draw_label(batch, (phase, None), prepass));
                pass.draw_indexed(
                    draw.index_range(),
                    draw.base_vertex,
                    batch.instances.clone(),
                );
                self.profiler.end_draw(pass, timed);
                draws += 1;
            }
        }
        draws
    }
}

fn draw_label(batch: &DrawBatch, (phase, level): (usize, Option<usize>), prepass: bool) -> String {
    let material = batch
        .material
        .map_or_else(|| "none".to_owned(), |material| material.slot().to_string());
    let mesh = batch.mesh.slot();
    let stage = if prepass { " depth pre-pass" } else { "" };
    match level {
        Some(level) => {
            format!("mesh {mesh} material {material} phase {phase} level {level}{stage}")
        }
        None => format!("mesh {mesh} material {material} phase {phase}{stage}"),
    }
}
