use bevy_ecs::prelude::*;
use engine_core::{Error, View};
use engine_render::{
    Access, Attach, Bind, DepthUse, Encode, FULLSCREEN, GpuPart, Install, Lifetime, Own, Pass,
    Pipeline, Prepare, RenderPipeline, SCENE_FORMAT, Shape, Texture, ViewTarget, wgpu,
};

use crate::shaders::taa::{lit as lighting, resolve};

pub(crate) struct LitGpu {
    pass: Pass<Self>,
    lit: Texture<Self>,
    pipeline: Pipeline<Self, Own>,
    drawn: (u32, u32),
}

impl GpuPart for LitGpu {
    type Reads = Res<'static, View>;

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let lit = cx.texture("lit", SCENE_FORMAT, Lifetime::Frame);
        cx.frame(resolve::LIT, Bind::Texture(lit))?;
        let pass = cx
            .pass("temporal_lit")
            .label("engine/taa-lit")
            .span("taa_lit")
            .color(lit, Access::Clear)
            .id();
        let pipeline = cx.render_pipeline(
            pass,
            &RenderPipeline {
                vertex: FULLSCREEN,
                fragment: Some(lighting::FS_LIT),
                debug: None,
                vertices: &[],
                primitive: wgpu::PrimitiveState::default(),
                depth: DepthUse::None,
                blend: None,
                writes: wgpu::ColorWrites::ALL,
            },
            &[(lighting::SCENE_COLOR, Bind::Target(ViewTarget::Scene))],
        )?;
        Ok(Self {
            pass,
            lit,
            pipeline,
            drawn: (0, 0),
        })
    }

    fn prepare(&mut self, view: SystemParamItem<'_, '_, Self::Reads>, cx: &mut Prepare<'_, Self>) {
        if !view.temporal() {
            return;
        }
        let display = view.display_size();
        let render = view.render_size();
        cx.allocate(
            self.lit,
            wgpu::Extent3d {
                width: display.x.max(1),
                height: display.y.max(1),
                depth_or_array_layers: 1,
            },
            1,
        );
        let within = |size: u32, whole: u32| size.clamp(1, whole.max(1));
        self.drawn = (within(render.x, display.x), within(render.y, display.y));
        cx.run(self.pass);
    }

    fn encode(&mut self, _pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let mut draw = cx.render(Attach::Declared, None);
        draw.within(self.drawn);
        draw.draw(
            &self.pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
            },
        );
        0
    }
}
