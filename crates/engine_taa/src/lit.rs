use engine_core::{Entry, Error, PassName, ViewFacts};
use engine_render::{
    Access, Allocation, Attach, Bind, Buffer, DepthUse, Encode, FULLSCREEN, GpuPart, Install,
    Lifetime, Own, Pass, Pipeline, Prepare, RenderPipeline, SCENE_FORMAT, Shape, TargetName,
    Texture, wgpu,
};

use crate::TemporalPass;
use crate::shaders::taa::{lit as lighting, resolve, resolved};
use crate::temporal::TaaView;

pub(crate) const RESOLVE_PASS: &str = "temporal_resolve";
const RESOLVED_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub(crate) struct LitGpu {
    pass: Pass<Self>,
    resolve: Pass<Self>,
    lit: Texture<Self>,
    resolved: Texture<Self>,
    history: Texture<Self>,
    view: Buffer<Self>,
    lighting: Pipeline<Self, Own>,
    resolving: Pipeline<Self, Own>,
    drawn: (u32, u32),
    whole: (u32, u32),
}

impl GpuPart for LitGpu {
    type Reads = ();

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let lit = cx.view_texture("lit", SCENE_FORMAT, Lifetime::Frame);
        let resolved_texture = cx.view_texture("resolved", RESOLVED_FORMAT, Lifetime::Frame);
        let history = cx.view_texture("history", RESOLVED_FORMAT, Lifetime::Previous);
        let view = cx.view_buffer("taa_view", Lifetime::Kept);
        cx.frame(resolved::RESOLVED_COLOR, Bind::Texture(resolved_texture))?;
        let pass = cx
            .screen::<TemporalPass>()
            .label("engine/taa-lit")
            .span("taa_lit")
            .color(lit, Access::Clear)
            .id();
        let resolve_pass = cx
            .pass(RESOLVE_PASS)
            .at(TemporalPass::PASS.at)
            .views(TemporalPass::PASS.views)
            .label("engine/taa-resolve")
            .span("taa_resolve")
            .after(pass)
            .color(resolved_texture, Access::Clear)
            .color(history, Access::Clear)
            .id();
        let lighting = cx.render_pipeline(
            pass,
            &screen(lighting::FS_LIT),
            &[(lighting::SCENE_COLOR, Bind::Target(TargetName::COLOR))],
        )?;
        let resolving = cx.render_pipeline(
            resolve_pass,
            &screen(resolve::FS_TEMPORAL_RESOLVE),
            &[
                (resolve::LIT, Bind::Texture(lit)),
                (resolve::HISTORY, Bind::Texture(history)),
                (resolve::TAA_VIEW, Bind::Buffer(view)),
            ],
        )?;
        Ok(Self {
            pass,
            resolve: resolve_pass,
            lit,
            resolved: resolved_texture,
            history,
            view,
            lighting,
            resolving,
            drawn: (0, 0),
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
        let render = view.render;
        let whole = wgpu::Extent3d {
            width: display.x.max(1),
            height: display.y.max(1),
            depth_or_array_layers: 1,
        };
        cx.allocate(self.lit, whole, 1);
        cx.allocate(self.resolved, whole, 1);
        let kept = cx.allocate(self.history, whole, 1) == Allocation::Kept;
        cx.allocate_buffer(self.view, size_of::<TaaView>().next_multiple_of(16) as u64);
        cx.write(
            self.view,
            0,
            &[TaaView {
                offset: offset.to_array(),
                history: u32::from(kept && view.history.valid()),
                pad: 0,
            }],
        );
        let within = |size: u32, whole: u32| size.clamp(1, whole.max(1));
        self.drawn = (within(render.x, display.x), within(render.y, display.y));
        self.whole = (whole.width, whole.height);
        cx.run(self.pass);
        cx.run(self.resolve);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let (pipeline, region) = if pass == self.resolve {
            (&self.resolving, self.whole)
        } else {
            (&self.lighting, self.drawn)
        };
        let mut draw = cx.render(Attach::Declared);
        draw.within(region);
        draw.draw(
            pipeline,
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
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}
