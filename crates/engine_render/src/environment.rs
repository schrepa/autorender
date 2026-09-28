//! Environment maps (decision 043): the sky drawn into a radiance cube when
//! the atmosphere changes, its levels averaged down, then each level of the
//! environment map blurred from them by the GGX lobe of its roughness.
//! Surfaces read their reflections from the environment map at their
//! roughness, and screen-space reflections (decision 025) fall back to it.
//! The map is the first cube of an array that later cubes extend.

use crate::gpu_data::{ENVIRONMENT_SHADER, LightingUniform, mip_levels};

/// Linear radiance: the sky and what reflects it exceed 1.
pub(crate) const ENVIRONMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Texels across each face of the radiance cube and environment maps.
pub(crate) const ENVIRONMENT_SIZE: u32 = 128;

pub(crate) struct Environment {
    // The radiance cube the sky is drawn into and its level views: one render
    // target per face of each level, and the groups reading level `i` to
    // average level `i + 1` down.
    radiance: wgpu::Texture,
    radiance_targets: Vec<Vec<wgpu::TextureView>>,
    down_groups: Vec<wgpu::BindGroup>,
    // Reads every radiance level for the blur.
    prefilter_group: wgpu::BindGroup,
    // The environment maps: `maps_view` samples them as a cube array, and
    // `map_targets[layer][level]` renders one face of one level.
    maps: wgpu::Texture,
    maps_view: wgpu::TextureView,
    map_targets: Vec<Vec<wgpu::TextureView>>,
    sky: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    prefilter: wgpu::RenderPipeline,
    sky_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    levels: u32,
    // The atmosphere values the sky map was drawn with.
    drawn: Option<[[f32; 4]; 5]>,
}

impl Drop for Environment {
    fn drop(&mut self) {
        self.radiance.destroy();
        self.maps.destroy();
    }
}

impl Environment {
    /// `lighting` is the uniform buffer of the atmosphere the sky is drawn
    /// with.
    pub(crate) fn new(device: &wgpu::Device, lighting: &wgpu::Buffer) -> Self {
        let levels = mip_levels(ENVIRONMENT_SIZE, ENVIRONMENT_SIZE);
        let cube = |label, layers| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: ENVIRONMENT_SIZE,
                    height: ENVIRONMENT_SIZE,
                    depth_or_array_layers: layers,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: ENVIRONMENT_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let radiance = cube("engine/environment-radiance", 6);
        let maps = cube("engine/environment-maps", 6);
        let face_view = |texture: &wgpu::Texture, layer: u32, level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: level,
                mip_level_count: Some(1),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        let cube_view = |texture: &wgpu::Texture, level: u32, count: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                base_mip_level: level,
                mip_level_count: Some(count),
                base_array_layer: 0,
                array_layer_count: Some(6),
                ..Default::default()
            })
        };
        let radiance_targets = (0..levels)
            .map(|level| {
                (0..6)
                    .map(|face| face_view(&radiance, face, level))
                    .collect()
            })
            .collect();
        let map_targets = (0..6)
            .map(|face| {
                (0..levels)
                    .map(|level| face_view(&maps, face, level))
                    .collect()
            })
            .collect();
        let maps_view = maps.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/environment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(ENVIRONMENT_SHADER.into()),
        });
        let pipeline = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_environment"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(ENVIRONMENT_FORMAT.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let sky = pipeline("fs_environment_sky");
        let down = pipeline("fs_environment_down");
        let prefilter = pipeline("fs_environment_prefilter");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("engine/environment-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let sampler_entry = wgpu::BindGroupEntry {
            binding: 22,
            resource: wgpu::BindingResource::Sampler(&sampler),
        };
        let sky_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/environment-sky-group"),
            layout: &sky.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 1,
                resource: lighting.as_entire_binding(),
            }],
        });
        let down_groups = (0..levels.saturating_sub(1))
            .map(|level| {
                let source = cube_view(&radiance, level, 1);
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("engine/environment-down-group"),
                    layout: &down.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 20,
                            resource: wgpu::BindingResource::TextureView(&source),
                        },
                        sampler_entry.clone(),
                    ],
                })
            })
            .collect();
        let every_level = cube_view(&radiance, 0, levels);
        let prefilter_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/environment-prefilter-group"),
            layout: &prefilter.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: wgpu::BindingResource::TextureView(&every_level),
                },
                sampler_entry,
            ],
        });
        Self {
            radiance,
            radiance_targets,
            down_groups,
            prefilter_group,
            maps,
            maps_view,
            map_targets,
            sky,
            down,
            prefilter,
            sky_group,
            sampler,
            levels,
            drawn: None,
        }
    }

    /// The environment maps as a cube array and the sampler surfaces read
    /// them with, for the main passes' first group.
    pub(crate) fn binding(&self) -> (&wgpu::TextureView, &wgpu::Sampler) {
        (&self.maps_view, &self.sampler)
    }

    /// Records the sky map's redraw when `lighting`'s sky differs from the
    /// one it holds; returns whether it did.
    pub(crate) fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        lighting: &LightingUniform,
    ) -> bool {
        let sky = [
            lighting.sun,
            lighting.sun_color,
            lighting.zenith,
            lighting.horizon,
            lighting.ground,
        ];
        if self.drawn == Some(sky) {
            return false;
        }
        self.drawn = Some(sky);
        let pass = |encoder: &mut wgpu::CommandEncoder,
                    target: &wgpu::TextureView,
                    pipeline: &wgpu::RenderPipeline,
                    group: &wgpu::BindGroup,
                    item: u32| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("engine/environment"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, item..item + 1);
        };
        for (face, target) in self.radiance_targets[0].iter().enumerate() {
            pass(encoder, target, &self.sky, &self.sky_group, face as u32);
        }
        for level in 1..self.levels as usize {
            for (face, target) in self.radiance_targets[level].iter().enumerate() {
                let item = face as u32;
                pass(
                    encoder,
                    target,
                    &self.down,
                    &self.down_groups[level - 1],
                    item,
                );
            }
        }
        for (face, targets) in self.map_targets.iter().enumerate() {
            for (level, target) in targets.iter().enumerate() {
                let item = face as u32 + 6 * level as u32;
                pass(
                    encoder,
                    target,
                    &self.prefilter,
                    &self.prefilter_group,
                    item,
                );
            }
        }
        true
    }

    /// Bytes of the radiance cube and environment maps with their levels.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let face = u64::from(ENVIRONMENT_SIZE) * u64::from(ENVIRONMENT_SIZE) * 8;
        // Two cubes of six faces, each with a third more for its levels.
        2 * 6 * face * 4 / 3
    }
}
