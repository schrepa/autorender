//! Mip chains of material textures, generated on the GPU when a texture
//! becomes resident: one small render pass per level, each averaging the
//! level above. Uploads stay the base level's texels; the chain adds at most a
//! third of them in GPU memory and costs no CPU work beyond command encoding.
//! Color textures average in linear light through their sRGB format; data
//! textures average their stored values.

use crate::bindings::MIPMAPS;
use crate::gpu_data::MIPMAP_SHADER;
use crate::shaders::entry::mipmaps;

pub(crate) struct MipGenerator {
    // sRGB color and linear data targets.
    pipelines: [wgpu::RenderPipeline; 2],
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl MipGenerator {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let layout = crate::pipelines::layout(device, &MIPMAPS);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/mipmaps.wgsl"),
            source: wgpu::ShaderSource::Wgsl(MIPMAP_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/texture-mip-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = [
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::Rgba8Unorm,
        ]
        .map(|format| {
            crate::pipelines::fullscreen(
                device,
                &crate::pipelines::Fullscreen {
                    label: "engine/texture-mips",
                    layout: &pipeline_layout,
                    module: &module,
                    vertex: mipmaps::vs_mip,
                    fragment: mipmaps::fs_mip,
                    constants: &[],
                    targets: &[Some(format.into())],
                    depth: None,
                },
            )
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/texture-mip-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            pipelines,
            layout,
            sampler,
        }
    }

    /// Records passes that fill every level of `texture` after the first from
    /// the one above; returns how many levels it fills. Record after the
    /// queue write of level 0 and submit before sampling the texture.
    pub(crate) fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) -> u32 {
        self.record_views(device, encoder, texture, None)
    }

    /// [`Self::record`] for layer `layer` of an array texture: impostors',
    /// after a bake renders its first level.
    pub(crate) fn record_layer(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        layer: u32,
    ) -> u32 {
        self.record_views(device, encoder, texture, Some(layer))
    }

    fn record_views(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        layer: Option<u32>,
    ) -> u32 {
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
            let group = crate::pipelines::group(
                device,
                "engine/texture-mip-group",
                &MIPMAPS,
                &self.layout,
                &[
                    ("source", crate::pipelines::view(&pair[0])),
                    ("source_sampler", crate::pipelines::sampler(&self.sampler)),
                ],
            );
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/texture-mip"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &pair[1],
                    depth_slice: None,
                    resolve_target: None,
                    // The triangle covers every texel of the level.
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
            pass.set_pipeline(&self.pipelines[usize::from(!texture.format().is_srgb())]);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        levels - 1
    }
}
