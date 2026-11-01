use engine_render::graph::{
    FrameGraph, GraphTable, Lifetime, PassDecl, PassId, ResourceDecl, ResourceId, Stage,
};
use engine_render::parts::{Create, Encode, GpuPart, PartDecl, PartEntry, View};
use engine_render::renderer::OpaqueReads;

pub(crate) static PART: PartEntry = PartEntry {
    decl: PartDecl {
        name: "culling",
        brick: "culling",
        passes: &[
            PassDecl {
                name: "occlusion_test",
                stage: Stage::Occlusion,
            },
            PassDecl {
                name: "opaque_phase_two",
                stage: Stage::OpaqueSecond,
            },
        ],
        resources: &[ResourceDecl {
            name: "pyramid",
            lifetime: Lifetime::Kept,
        }],
        hooks: &[],
        contributes: &[],
        resolve: None,
        labels: &[],
        lazy: false,
        needs: &[],
        piece: None,
    },
    create: create_culling,
};

fn create_culling(_: &Create<'_>) -> Box<dyn GpuPart> {
    Box::new(CullingPart::default())
}

#[derive(Default)]
pub(crate) struct CullingPart {
    test: Option<PassId>,
    second: Option<PassId>,
    pyramid: Option<ResourceId>,
    reads: OpaqueReads,
}

impl GpuPart for CullingPart {
    fn install(&mut self, table: &GraphTable) {
        self.test = table.pass_named("occlusion_test");
        self.second = table.pass_named("opaque_phase_two");
        self.pyramid = table.resource("pyramid");
        self.reads = OpaqueReads::of(table);
    }

    fn declare(&self, cx: &View<'_>, graph: &mut FrameGraph) {
        let Some(shape) = cx.frame.shape.filter(|shape| shape.phases == 2) else {
            return;
        };
        if let Some(test) = self.test {
            let _test = graph
                .add(cx.table, test)
                .read(self.reads.depth)
                .read(self.reads.poses)
                .write(self.pyramid)
                .write(self.reads.visible);
        }
        if let Some(second) = self.second {
            self.reads
                .declare_phase(cx.table, &shape, graph.add(cx.table, second), 1);
        }
    }

    fn encode(
        &mut self,
        pass: usize,
        cx: &mut Encode<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> u32 {
        if pass == 0 {
            encode_occlusion_test(cx, encoder);
        } else {
            engine_render::renderer::encode_phase(cx, encoder, 1);
        }
        0
    }
}

fn encode_occlusion_test(cx: &mut Encode<'_>, encoder: &mut wgpu::CommandEncoder) {
    let odd = cx
        .frame
        .temporal
        .as_ref()
        .is_some_and(|frame| frame.parity == 1);
    let core = &mut *cx.core;
    let span = core.profiler.span("occlusion");
    let Some(pyramid) = core.culling.pyramid.as_ref() else {
        return;
    };
    let Some(view) = pyramid.view() else {
        return;
    };
    let poses = core.instances.poses(usize::from(odd));
    pyramid.encode(encoder, span);
    core.gpu_presentation.encode_occlusion(
        &core.device,
        encoder,
        (view, view.generation()),
        poses,
        span,
    );
}
