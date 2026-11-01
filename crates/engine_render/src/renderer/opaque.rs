use super::*;
use crate::graph::{
    Access, Clear, FAR, FrameShape, Lifetime, PassBuilder, PassDecl, PassId, ResourceDecl,
    ResourceId, Stage,
};
use crate::parts::{Create, DrawAt, Encode, OpaqueSlot, PartDecl, PartEntry, View};

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
        resolve: None,
        labels: &[],
        lazy: false,
        needs: &[],
        piece: None,
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
            self.reads
                .declare_phase(cx.table, &shape, graph.add(cx.table, opaque), 0);
        }
        if shape.deferred
            && let Some(late) = self.late
        {
            self.reads
                .declare_late(cx.table, &shape, graph.add(cx.table, late));
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
