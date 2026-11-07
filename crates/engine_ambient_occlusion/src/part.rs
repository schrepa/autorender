use engine_core::{AttachPoint, Entry, Error, PassViews, ViewFacts};
use engine_render::{
    Access, Attach, Bind, DepthUse, Encode, FULLSCREEN, GpuPart, Install, Lifetime,
    Own, Pass, Pipeline, Prepare, RenderPipeline, Shape, Texture, wgpu,
};

use crate::shaders::ambient_occlusion::faces::ambient_occlusion as face;
use crate::shaders::ambient_occlusion::occlusion as shading;

const RAW_OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

pub(crate) struct AmbientGpu {
    occluding: Pass<Self>,
    filtering: Pass<Self>,
    raw: Texture<Self>,
    occlusion: Texture<Self>,
    occlusion_pipeline: Pipeline<Self, Own>,
    filter_pipeline: Pipeline<Self, Own>,
    drawn: (u32, u32),
}

fn fullscreen(fragment: Entry) -> RenderPipeline<'static> {
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

impl GpuPart for AmbientGpu {
    type Reads = ();
    const NAME: &'static str = "ambient_occlusion";
    const LAZY: bool = true;
    const LABELS: &'static [&'static str] = &["engine/ambient-occlusion"];

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let raw = cx.view_texture("raw_occlusion", RAW_OCCLUSION_FORMAT, Lifetime::Kept);
        let occlusion = cx.view_texture("occlusion", OCCLUSION_FORMAT, Lifetime::Kept);
        cx.frame(face::OCCLUSION, Bind::Texture(occlusion))?;
        let occluding = cx
            .pass("ambient_occlusion")
            .at(AttachPoint::Inputs)
            .views(PassViews::Main)
            .label("engine/ambient-occlusion")
            .span("ambient_occlusion")
            .color(raw, Access::Load)
            .id();
        let filtering = cx
            .pass("effects_filter")
            .at(AttachPoint::Inputs)
            .views(PassViews::Main)
            .label("engine/ambient-occlusion-filter")
            .span("effects_filter")
            .color(occlusion, Access::Clear)
            .id();
        let occlusion_pipeline =
            cx.render_pipeline(occluding, &fullscreen(shading::FS_OCCLUSION), &[])?;
        let filter_pipeline = cx.render_pipeline(
            filtering,
            &fullscreen(shading::FS_EFFECTS_FILTER),
            &[(shading::RAW_OCCLUSION, Bind::Texture(raw))],
        )?;
        Ok(Self {
            occluding,
            filtering,
            raw,
            occlusion,
            occlusion_pipeline,
            filter_pipeline,
            drawn: (0, 0),
        })
    }

    fn prepare(&mut self, (): (), _cx: &mut Prepare<'_, Self>) {}

    fn view(&mut self, view: &ViewFacts, cx: &mut Prepare<'_, Self>) {
        if !view.target.surface() || cx.view_index() != 0 {
            cx.release(self.occlusion);
            return;
        }
        let display = view.size();
        let render = view.render;
        let extent = wgpu::Extent3d {
            width: display.x.max(1),
            height: display.y.max(1),
            depth_or_array_layers: 1,
        };
        cx.allocate(self.raw, extent, 1);
        cx.allocate(self.occlusion, extent, 1);
        let within = |size: u32, whole: u32| size.clamp(1, whole.max(1));
        self.drawn = (within(render.x, display.x), within(render.y, display.y));
        cx.run(self.occluding);
        cx.run(self.filtering);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let pipeline = if pass == self.occluding {
            &self.occlusion_pipeline
        } else {
            &self.filter_pipeline
        };
        let mut draw = cx.render(Attach::Declared);
        draw.within(self.drawn);
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
