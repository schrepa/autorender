use bevy_ecs::prelude::{Res, Resource, SystemParamItem};
use engine_core::{AttachPoint, Error, PassViews, ViewFacts, gpu_consts, gpu_struct};
use engine_render::{
    Allocation, Attach, Bind, Buffer, DEBUG_TARGETS, DebugShown, DepthUse, Encode, FULLSCREEN,
    GpuPart, Install, Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, Shape, wgpu,
};

use crate::probe::Probe;
use crate::shaders::debug_view::overlay;

gpu_consts! {
    pub const DEBUG_VIEW_KINDS: u32 = 1;
    pub const DEBUG_VIEW_LEVELS: u32 = 2;
    pub const DEBUG_VIEW_CARDS: u32 = 3;
    pub const DEBUG_VIEW_OVERDRAW: u32 = 6;
    pub const DEBUG_VIEW_NORMALS: u32 = 7;
    pub const PROBED_IDS: u32 = 0;
    pub const PROBED_COUNT: u32 = 64;
    pub const PROBED_WORDS: u32 = 256;
}

pub(crate) const GENERATED: &[&str] = &[WGSL];

gpu_struct! {
    #[derive(Resource)]
    pub(crate) struct DebugViewPiece {
        pub view: [u32; 4],
        pub probe: [u32; 4],
    }
}

const TRIANGLE: Shape<'static, OverlayGpu, Own> = Shape::Pulled {
    vertices: 0..3,
    instances: 0..1,
};

pub(crate) struct OverlayGpu {
    pass: Pass<Self>,
    pipeline: Pipeline<Self, Own>,
    placed: Buffer<Self>,
    pixels: [u32; 4],
    showing: bool,
}

impl GpuPart for OverlayGpu {
    type Reads = (Res<'static, DebugShown>, Res<'static, Probe>);
    const NAME: &'static str = "debug_view_overlay";

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let placed = cx.view_buffer("debug_view_placed", Lifetime::Kept);
        let pass = cx
            .pass("debug_view_overlay")
            .at(AttachPoint::Overlay)
            .views(PassViews::Main)
            .span("debug_view")
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: FULLSCREEN,
                fragment: Some(overlay::FS_DEBUG_OVERLAY),
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::None,
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &[
                (overlay::PLACED, Bind::Buffer(placed)),
                (overlay::DEBUG_IDS, Bind::Target(DEBUG_TARGETS[0])),
                (overlay::DEBUG_COUNT, Bind::Target(DEBUG_TARGETS[1])),
                (overlay::DEBUG_WORDS, Bind::Target(DEBUG_TARGETS[2])),
            ],
        )?;
        Ok(Self {
            pass,
            pipeline,
            placed,
            pixels: [0; 4],
            showing: false,
        })
    }

    fn prepare(
        &mut self,
        (shown, probe): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        self.showing = shown.owner.is_some() && !probe.capturing();
        if self.showing {
            cx.run(self.pass);
        }
    }

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if !self.showing {
            return;
        }
        self.pixels = view.pixels;
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.placed, size_of::<[u32; 4]>() as u64);
        cx.write(self.placed, 0, &[view.pixels]);
    }

    fn encode(&mut self, _: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let mut draw = cx.render(Attach::Declared);
        draw.scissor(self.pixels);
        draw.draw(&self.pipeline, TRIANGLE);
        0
    }
}
