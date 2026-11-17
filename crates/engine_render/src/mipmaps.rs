use engine_core::{Entry, Error, Var};

use crate::encode::Compiler;
use crate::install::{Bound, Program, RenderDecl, Scoped};

const MODULE: &str = "engine::mipmaps";
const GROUP: u32 = 2;
const FORMATS: [wgpu::TextureFormat; 2] = [
    wgpu::TextureFormat::Rgba8UnormSrgb,
    wgpu::TextureFormat::Rgba8Unorm,
];

pub(crate) fn drawn(format: wgpu::TextureFormat, dimension: wgpu::TextureDimension) -> bool {
    dimension == wgpu::TextureDimension::D2 && FORMATS.contains(&format)
}

pub(crate) struct Mipmaps {
    pipeline: u32,
}

impl crate::GpuPart for Mipmaps {
    type Reads = ();
    const NAME: &'static str = "mipmaps";

    fn install(cx: &mut crate::Install<'_, Self>) -> Result<Self, Error> {
        let module = crate::shaders::core_module(MODULE);
        let pipeline = cx.core.pipeline(
            Scoped::Outside(Self::NAME),
            Program::Render(RenderDecl {
                vertex: Entry::generated(module, "vs_mip"),
                fragment: Some(Entry::generated(module, "fs_mip")),
                vertices: Vec::new(),
                primitive: wgpu::PrimitiveState::default(),
                depth: crate::DepthUse::None,
                stencil: wgpu::StencilState::default(),
                targets: vec![(None, wgpu::ColorWrites::ALL)],
                samples: 1,
                faces: crate::install::Faces::Declared,
            }),
            vec![
                (Var::generated(module, "source"), Bound::Asset),
                (Var::generated(module, "source_sampler"), Bound::Asset),
            ],
        )?;
        Ok(Self { pipeline })
    }

    fn prepare(&mut self, (): (), _cx: &mut crate::Prepare<'_, Self>) {}
}

pub struct MipGenerator {
    sampler: wgpu::Sampler,
}

impl MipGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        let sampler = crate::materials::gpu_sampler(
            device,
            "engine/texture-mip-sampler",
            engine_core::Sampler {
                mipmap_filter: Some(engine_core::Filter::Nearest),
                ..engine_core::Sampler::LINEAR_CLAMP
            },
        );
        Self { sampler }
    }

    pub(crate) fn record(
        &self,
        (device, encoder): (&wgpu::Device, &mut wgpu::CommandEncoder),
        compiler: &mut Compiler<'_>,
        texture: &wgpu::Texture,
        layer: Option<u32>,
    ) -> Option<u32> {
        let Some(pipeline) = compiler.part::<Mipmaps>().map(|mipmaps| mipmaps.pipeline) else {
            return Some(0);
        };
        let (_, made) = compiler.compile_now(
            pipeline,
            (&[Some(texture.format())], None),
            (false, false),
            None,
        )?;
        let Some(layout) = compiler.group_layout(pipeline, GROUP) else {
            return Some(0);
        };
        let levels = texture.mip_level_count();
        let views: Vec<wgpu::TextureView> = (0..levels)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("engine/texture-mip-level"),
                    dimension: layer.map(|_| wgpu::TextureViewDimension::D2),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    base_array_layer: layer.unwrap_or(0),
                    array_layer_count: layer.map(|_| 1),
                    ..Default::default()
                })
            })
            .collect();
        for pair in views.windows(2) {
            let group = crate::pipelines::bind_group(
                device,
                "engine/texture-mip-group",
                &layout,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: crate::pipelines::view(&pair[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: crate::pipelines::sampler(&self.sampler),
                    },
                ],
            );
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/texture-mip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pair[1],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&made);
            pass.set_bind_group(GROUP, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        Some(levels - 1)
    }
}
