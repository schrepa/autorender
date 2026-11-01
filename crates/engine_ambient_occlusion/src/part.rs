use bevy_ecs::prelude::*;
use engine_core::{Entry, Error, View};
use engine_render::{
    Access, Attach, Bind, DebugShown, DepthUse, Encode, GpuPart, Install, Lifetime, Own, Pass,
    Pipeline, Prepare, RenderPipeline, Shape, Texture, wgpu,
};

use crate::shaders::ambient_occlusion::{apply, occlusion as shading};

const RAW_OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

pub(crate) struct AmbientGpu {
    occluding: Pass<Self>,
    filtering: Pass<Self>,
    raw: Texture<Self>,
    occlusion: Texture<Self>,
    occlusion_pipeline: Pipeline<Self, Own>,
    filter_pipeline: Pipeline<Self, Own>,
}

fn fullscreen(fragment: Entry) -> RenderPipeline<'static> {
    RenderPipeline {
        vertex: shading::VS_FULLSCREEN,
        fragment: Some(fragment),
        debug: None,
        vertices: &[],
        primitive: wgpu::PrimitiveState::default(),
        depth: DepthUse::None,
        blend: None,
        writes: wgpu::ColorWrites::ALL,
    }
}

impl GpuPart for AmbientGpu {
    type Reads = (Res<'static, View>, Res<'static, DebugShown>);

    fn install(cx: &mut Install<'_, Self>) -> Result<Self, Error> {
        let raw = cx.texture("raw_occlusion", RAW_OCCLUSION_FORMAT, Lifetime::Frame);
        let occlusion = cx.texture("occlusion", OCCLUSION_FORMAT, Lifetime::Frame);
        cx.frame(apply::OCCLUSION, Bind::Texture(occlusion))?;
        let occluding = cx.pass("ambient_occlusion").color(raw, Access::Clear).id();
        let filtering = cx.pass("effects_filter").color(occlusion, Access::Clear).id();
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
        })
    }

    fn prepare(
        &mut self,
        (view, shown): SystemParamItem<'_, '_, Self::Reads>,
        cx: &mut Prepare<'_, Self>,
    ) {
        if shown.owner.is_some() {
            return;
        }
        let size = view.render_size();
        let extent = wgpu::Extent3d {
            width: size.x.max(1),
            height: size.y.max(1),
            depth_or_array_layers: 1,
        };
        cx.allocate(self.raw, extent, 1);
        cx.allocate(self.occlusion, extent, 1);
        cx.run(self.occluding);
        cx.run(self.filtering);
    }

    fn encode(&mut self, pass: Pass<Self>, cx: &mut Encode<'_, Self>) -> u32 {
        let pipeline = if pass == self.occluding {
            &self.occlusion_pipeline
        } else {
            &self.filter_pipeline
        };
        cx.render(Attach::Declared, None).draw(
            pipeline,
            Shape::Pulled {
                vertices: 0..3,
                instances: 0..1,
            },
        );
        0
    }
}
