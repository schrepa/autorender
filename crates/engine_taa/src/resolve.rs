use engine_core::{AttachPoint, Entry, Error, PassViews, ViewFacts};
use engine_render::{
    Access, Allocation, Attach, Bind, Buffer, DepthUse, Encode, FULLSCREEN, GpuPart, Install,
    Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, Shape, TargetName, Texture, wgpu,
};

use crate::shaders::taa::{resolve, resolved};
use crate::temporal::TaaView;

const RESOLVE_PASS: &str = "temporal_resolve";
const RESOLVED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub(crate) struct ResolveGpu {
    resolve: Pass<Self>,
    resolved: Texture<Self>,
    history: Texture<Self>,
    view: Buffer<Self>,
    resolving: Pipeline<Self, Own>,
    whole: (u32, u32),
}

impl GpuPart for ResolveGpu {
    type Reads = ();

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let resolved_texture = cx.view_texture("resolved", RESOLVED_FORMAT, Lifetime::Frame);
        let history = cx.view_texture("history", RESOLVED_FORMAT, Lifetime::Previous);
        let view = cx.view_buffer("taa_view", Lifetime::Kept);
        cx.frame(resolved::RESOLVED_COLOR, Bind::Texture(resolved_texture))?;
        let resolve_pass = cx
            .pass(RESOLVE_PASS)
            .at(AttachPoint::Post)
            .views(PassViews::Main)
            .label("engine/taa-resolve")
            .span("taa_resolve")
            .color(resolved_texture, Access::Clear)
            .color(history, Access::Clear)
            .id();
        let resolving = cx.render_pipeline(
            resolve_pass,
            &screen(resolve::FS_TEMPORAL_RESOLVE),
            &[
                (resolve::SCENE_COLOR, Bind::Target(TargetName::COLOR)),
                (resolve::HISTORY, Bind::Texture(history)),
                (resolve::TAA_VIEW, Bind::Buffer(view)),
                (resolve::MOTION_VECTORS, Bind::Target(TargetName::MOTION)),
                (resolve::BLENDED_MASK, Bind::Target(TargetName::TRANSPARENCY)),
            ],
        )?;
        Ok(Self {
            resolve: resolve_pass,
            resolved: resolved_texture,
            history,
            view,
            resolving,
            whole: (0, 0),
        })
    }

    fn prepare(&mut self, (): (), _cx: &mut Prepare<'_, Self>) {}

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        let Some(offset) = view.jitter else {
            cx.release(self.resolved);
            return;
        };
        let display = view.size();
        let whole = wgpu::Extent3d {
            width: display.x.max(1),
            height: display.y.max(1),
            depth_or_array_layers: 1,
        };
        let kept = cx.allocate(self.history, whole, 1) == Allocation::Kept;
        let values = TaaView {
            offset: offset.to_array(),
            history: u32::from(kept && view.history.valid()),
            pad: 0,
        };
        let (Allocation::Fresh | Allocation::Kept) =
            cx.allocate_buffer(self.view, size_of::<TaaView>().next_multiple_of(16) as u64);
        cx.write(self.view, 0, &[values]);
        self.whole = (whole.width, whole.height);
        let (Allocation::Fresh | Allocation::Kept) = cx.allocate(self.resolved, whole, 1);
        cx.run(self.resolve);
    }

    fn encode(&mut self, _pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let mut draw = cx.render(Attach::Declared);
        draw.within(self.whole);
        draw.draw(
            &self.resolving,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
            },
        );
        0
    }
}

fn screen(fragment: Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: FULLSCREEN,
        fragment: Some(fragment),
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}
