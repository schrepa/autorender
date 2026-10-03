//! Environment maps (decision 043): cubes surfaces read their reflections
//! from at their roughness, each level blurred by the GGX lobe of its
//! roughness from a radiance cube with its levels averaged down. The first
//! cube holds the sky, drawn into the radiance cube whenever the atmosphere
//! changes; the others hold reflection probes, whose captures draw into the
//! same radiance cube face by face. Screen-space reflections (decision 025)
//! fall back to them.

use crate::gpu_data::{ENVIRONMENT_SHADER, LightingUniform, mip_levels};

/// Linear radiance: the sky and what reflects it exceed 1.
pub(crate) const ENVIRONMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

// The textures of one face size and cube count, and their views. Replaced
// maps are dropped, not destroyed: growing the array copies from them in the
// frame that replaces them.
struct Maps {
    size: u32,
    levels: u32,
    cubes: u32,
    // The radiance cube (kept alive beside its views): one render target per
    // face of each level, and the groups reading level `i` to average level
    // `i + 1` down.
    _radiance: wgpu::Texture,
    radiance_targets: Vec<Vec<wgpu::TextureView>>,
    down_groups: Vec<wgpu::BindGroup>,
    // Reads every radiance level for the blur.
    prefilter_group: wgpu::BindGroup,
    // The environment maps, sampled as a cube array.
    maps: wgpu::Texture,
    maps_view: wgpu::TextureView,
}

pub(crate) struct Environment {
    maps: Maps,
    sky: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    prefilter: wgpu::RenderPipeline,
    sky_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    // The atmosphere values the sky map was drawn with.
    drawn: Option<[[f32; 4]; 5]>,
    // Advances whenever the maps are replaced: groups binding them rebuild.
    generation: u64,
}

impl Environment {
    /// Maps of `size` texels a face holding the sky, drawn with the
    /// atmosphere of the uniform buffer `lighting`.
    pub(crate) fn new(device: &wgpu::Device, lighting: &wgpu::Buffer, size: u32) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/environment.wgsl"),
            source: wgpu::ShaderSource::Wgsl(ENVIRONMENT_SHADER.into()),
        });
        let pipeline = |entry: &str| {
            crate::pipelines::render(
                device,
                &wgpu::RenderPipelineDescriptor {
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
                },
            )
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
        let sky_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("engine/environment-sky-group"),
            layout: &sky.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 1,
                resource: lighting.as_entire_binding(),
            }],
        });
        let maps = create_maps(device, (&down, &prefilter, &sampler), size, 1);
        Self {
            maps,
            sky,
            down,
            prefilter,
            sky_group,
            sampler,
            drawn: None,
            generation: 0,
        }
    }

    /// The environment maps as a cube array and the sampler surfaces read
    /// them with, for the main passes' first group.
    pub(crate) fn binding(&self) -> (&wgpu::TextureView, &wgpu::Sampler) {
        (&self.maps.maps_view, &self.sampler)
    }

    /// Advances whenever the maps are replaced.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    /// Texels across each face.
    pub(crate) fn size(&self) -> u32 {
        self.maps.size
    }

    /// Cubes the maps hold, the sky's first.
    pub(crate) fn cubes(&self) -> u32 {
        self.maps.cubes
    }

    /// Replaces the maps with ones of `size` texels a face holding `cubes`
    /// cubes, unless they already are: at a new size every cube is lost (the
    /// sky redraws), while more cubes of the same size copy those held.
    /// Returns whether the cubes held were lost.
    pub(crate) fn ensure(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        size: u32,
        cubes: u32,
    ) -> bool {
        if self.maps.size == size && self.maps.cubes == cubes {
            return false;
        }
        let maps = create_maps(
            device,
            (&self.down, &self.prefilter, &self.sampler),
            size,
            cubes,
        );
        let kept = self.maps.size == size && cubes > self.maps.cubes;
        if kept {
            for level in 0..self.maps.levels {
                let extent = size >> level;
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.maps.maps,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &maps.maps,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: extent,
                        height: extent,
                        depth_or_array_layers: self.maps.cubes * 6,
                    },
                );
            }
        } else {
            self.drawn = None;
        }
        self.maps = maps;
        self.generation += 1;
        !kept
    }

    /// Records the sky map's redraw when `lighting`'s sky differs from the
    /// one it holds; returns whether it did.
    pub(crate) fn encode_sky(
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
        for (face, target) in self.maps.radiance_targets[0].iter().enumerate() {
            face_pass(encoder, target, &self.sky, &self.sky_group, face as u32);
        }
        self.finish(encoder, 0);
        true
    }

    /// The radiance cube's first level face `face`, which a probe's capture
    /// draws into before [`Self::finish`].
    pub(crate) fn radiance_face(&self, face: usize) -> &wgpu::TextureView {
        &self.maps.radiance_targets[0][face]
    }

    /// Records the radiance cube's levels averaged down from its first and
    /// every level of cube `cube` of the maps blurred from them.
    pub(crate) fn finish(&self, encoder: &mut wgpu::CommandEncoder, cube: u32) {
        for level in 1..self.maps.levels as usize {
            for (face, target) in self.maps.radiance_targets[level].iter().enumerate() {
                let group = &self.maps.down_groups[level - 1];
                face_pass(encoder, target, &self.down, group, face as u32);
            }
        }
        for face in 0..6 {
            for level in 0..self.maps.levels {
                let target = self.maps.maps.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    base_array_layer: cube * 6 + face,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let item = face + 6 * level;
                face_pass(
                    encoder,
                    &target,
                    &self.prefilter,
                    &self.maps.prefilter_group,
                    item,
                );
            }
        }
    }

    /// Bytes of the radiance cube and the maps with their levels.
    pub(crate) fn capacity_bytes(&self) -> u64 {
        let face = u64::from(self.maps.size) * u64::from(self.maps.size) * 8;
        // Six faces a cube, each with a third more for its levels.
        (1 + u64::from(self.maps.cubes)) * 6 * face * 4 / 3
    }
}

fn create_maps(
    device: &wgpu::Device,
    (down, prefilter, sampler): (&wgpu::RenderPipeline, &wgpu::RenderPipeline, &wgpu::Sampler),
    size: u32,
    cubes: u32,
) -> Maps {
    let levels = mip_levels(size, size);
    let cube = |label, layers, extra| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENVIRONMENT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | extra,
            view_formats: &[],
        })
    };
    let radiance = cube(
        "engine/environment-radiance",
        6,
        wgpu::TextureUsages::empty(),
    );
    // Growing the array copies the cubes held into the next one.
    let maps = cube(
        "engine/environment-maps",
        cubes * 6,
        wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
    );
    let radiance_targets = (0..levels)
        .map(|level| {
            (0..6)
                .map(|face| {
                    radiance.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_mip_level: level,
                        mip_level_count: Some(1),
                        base_array_layer: face,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect()
        })
        .collect();
    let cube_view = |level: u32, count: u32| {
        radiance.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_mip_level: level,
            mip_level_count: Some(count),
            base_array_layer: 0,
            array_layer_count: Some(6),
            ..Default::default()
        })
    };
    let sampler_entry = wgpu::BindGroupEntry {
        binding: 22,
        resource: wgpu::BindingResource::Sampler(sampler),
    };
    let down_groups = (0..levels.saturating_sub(1))
        .map(|level| {
            let source = cube_view(level, 1);
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
    let every_level = cube_view(0, levels);
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
    let maps_view = maps.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::CubeArray),
        ..Default::default()
    });
    Maps {
        size,
        levels,
        cubes,
        _radiance: radiance,
        radiance_targets,
        down_groups,
        prefilter_group,
        maps,
        maps_view,
    }
}

// Draws item `item` (face, plus 6 per level) of `pipeline` over `target`.
fn face_pass(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
    item: u32,
) {
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
}
