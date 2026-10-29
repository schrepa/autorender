use std::cell::OnceCell;

use crate::bindings::{ENVIRONMENT_DOWN, ENVIRONMENT_PREFILTER, ENVIRONMENT_SKY};
use crate::cached::{Bound, Versioned};
use crate::gpu_data::{LightingUniform, mip_levels};
use crate::probes::FACES;
use crate::shaders::entry::environment;

pub(crate) const ENVIRONMENT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const FACES_PER_PASS: usize = 3;

struct Maps {
    size: u32,
    levels: u32,
    cubes: u32,
    _radiance: wgpu::Texture,
    radiance_targets: Vec<Vec<wgpu::TextureView>>,
    down_groups: Vec<wgpu::BindGroup>,
    prefilter_group: wgpu::BindGroup,
    maps: wgpu::Texture,
    maps_view: Versioned<wgpu::TextureView>,
    cube_targets: Vec<OnceCell<Vec<wgpu::TextureView>>>,
}

struct Layouts {
    sky: crate::pipelines::GroupLayout,
    down: crate::pipelines::GroupLayout,
    prefilter: crate::pipelines::GroupLayout,
}

pub(crate) struct Environment {
    maps: Maps,
    sky: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    prefilter: wgpu::RenderPipeline,
    layouts: Layouts,
    sky_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    lighting: wgpu::Buffer,
    clouds: (u64, [u8; 2], bool),
    air: (
        wgpu::Buffer,
        wgpu::TextureView,
        wgpu::TextureView,
        wgpu::Sampler,
    ),
    drawn: Option<[[f32; 4]; 7]>,
}

impl Environment {
    pub(crate) fn new(
        device: &wgpu::Device,
        lighting: &wgpu::Buffer,
        size: u32,
        clouds: crate::clouds::DomeBinding<'_>,
        air: crate::sky::AirBindings<'_>,
    ) -> Self {
        let layouts = Layouts {
            sky: crate::pipelines::layout(device, &ENVIRONMENT_SKY),
            down: crate::pipelines::layout(device, &ENVIRONMENT_DOWN),
            prefilter: crate::pipelines::layout(device, &ENVIRONMENT_PREFILTER),
        };
        let targets: [Option<wgpu::ColorTargetState>; FACES_PER_PASS] =
            std::array::from_fn(|_| Some(ENVIRONMENT_FORMAT.into()));
        let pipeline = |entry: crate::shaders::Entry,
                        declared: &'static crate::pipeline_layouts::PipelineLayout| {
            let layout = crate::pipelines::layout_of(device, declared);
            crate::pipelines::fullscreen(
                device,
                &crate::pipelines::Fullscreen {
                    label: entry.name,
                    layout: &layout,
                    vertex: environment::vs_environment,
                    fragment: entry,
                    constants: &[],
                    targets: &targets,
                    depth: None,
                },
            )
        };
        let sky = pipeline(
            environment::fs_environment_sky,
            &crate::pipeline_layouts::ENVIRONMENT_SKY_PIPELINES,
        );
        let down = pipeline(
            environment::fs_environment_down,
            &crate::pipeline_layouts::ENVIRONMENT_DOWN_PIPELINES,
        );
        let prefilter = pipeline(
            environment::fs_environment_prefilter,
            &crate::pipeline_layouts::ENVIRONMENT_PREFILTER_PIPELINES,
        );
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
        let clouds_key = clouds.key;
        let sky_group = sky_group(device, &layouts.sky, lighting, clouds, air);
        let maps = create_maps(device, (&layouts, &sampler), size, 1);
        Self {
            maps,
            sky,
            down,
            prefilter,
            layouts,
            sky_group,
            sampler,
            lighting: lighting.clone(),
            clouds: clouds_key,
            air: (air.0.clone(), air.1.clone(), air.2.clone(), air.3.clone()),
            drawn: None,
        }
    }

    pub(crate) fn set_clouds(
        &mut self,
        device: &wgpu::Device,
        clouds: crate::clouds::DomeBinding<'_>,
    ) {
        if self.clouds != clouds.key {
            let air = (&self.air.0, &self.air.1, &self.air.2, &self.air.3);
            self.clouds = clouds.key;
            self.sky_group = sky_group(device, &self.layouts.sky, &self.lighting, clouds, air);
        }
    }

    pub(crate) fn binding(&self) -> (Bound<'_>, Bound<'_>) {
        (
            self.maps.maps_view.bound(),
            Bound::fixed(crate::pipelines::sampler(&self.sampler)),
        )
    }

    pub(crate) fn size(&self) -> u32 {
        self.maps.size
    }

    pub(crate) fn cubes(&self) -> u32 {
        self.maps.cubes
    }

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
        let maps = create_maps(device, (&self.layouts, &self.sampler), size, cubes);
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
        !kept
    }

    pub(crate) fn needs_encoding(
        &self,
        size: u32,
        cubes: u32,
        lighting: &LightingUniform,
        (clouds, air): ([f32; 4], [f32; 4]),
    ) -> bool {
        self.maps.size != size
            || self.maps.cubes != cubes
            || self.drawn != Some(sky(lighting, clouds, air))
    }

    pub(crate) fn encode_sky(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        lighting: &LightingUniform,
        (clouds, air): ([f32; 4], [f32; 4]),
    ) -> (bool, bool) {
        let sky = sky(lighting, clouds, air);
        if self.drawn == Some(sky) {
            return (false, false);
        }
        let beyond_clouds = self.drawn.is_none_or(|drawn| {
            drawn
                .iter()
                .zip(&sky)
                .enumerate()
                .any(|(index, (old, new))| index != 5 && old != new)
        });
        self.drawn = Some(sky);
        for first in (0..FACES).step_by(FACES_PER_PASS) {
            faces_pass(
                encoder,
                &self.maps.radiance_targets[0][first..first + FACES_PER_PASS],
                &self.sky,
                &self.sky_group,
                first as u32,
            );
        }
        self.finish(encoder, 0);
        (true, beyond_clouds)
    }

    pub(crate) fn radiance_face(&self, face: usize) -> &wgpu::TextureView {
        &self.maps.radiance_targets[0][face]
    }

    pub(crate) fn finish(&self, encoder: &mut wgpu::CommandEncoder, cube: u32) {
        let maps = &self.maps;
        for level in 1..maps.levels as usize {
            for first in (0..FACES).step_by(FACES_PER_PASS) {
                faces_pass(
                    encoder,
                    &maps.radiance_targets[level][first..first + FACES_PER_PASS],
                    &self.down,
                    &maps.down_groups[level - 1],
                    first as u32,
                );
            }
        }
        let (texture, levels) = (&maps.maps, maps.levels);
        let targets = maps.cube_targets[cube as usize].get_or_init(|| {
            (0..levels)
                .flat_map(|level| {
                    (0..FACES as u32).map(move |face| {
                        texture.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_mip_level: level,
                            mip_level_count: Some(1),
                            base_array_layer: cube * FACES as u32 + face,
                            array_layer_count: Some(1),
                            ..Default::default()
                        })
                    })
                })
                .collect()
        });
        for level in 0..levels as usize {
            for first in (0..FACES).step_by(FACES_PER_PASS) {
                let start = level * FACES + first;
                faces_pass(
                    encoder,
                    &targets[start..start + FACES_PER_PASS],
                    &self.prefilter,
                    &maps.prefilter_group,
                    start as u32,
                );
            }
        }
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        let face = u64::from(self.maps.size) * u64::from(self.maps.size) * 8;
        (1 + u64::from(self.maps.cubes)) * 6 * face * 4 / 3
    }
}

fn create_maps(
    device: &wgpu::Device,
    (layouts, sampler): (&Layouts, &wgpu::Sampler),
    size: u32,
    cubes: u32,
) -> Maps {
    let levels = mip_levels(size, size);
    let cube = |label, layers, extra| {
        crate::targets::texture_2d(
            device,
            label,
            (size, size),
            layers,
            levels,
            ENVIRONMENT_FORMAT,
            crate::targets::RENDER_TARGET | extra,
        )
    };
    let radiance = cube(
        "engine/environment-radiance",
        6,
        wgpu::TextureUsages::empty(),
    );
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
    let sampler_entry = ("cube_sampler", crate::pipelines::sampler(sampler));
    let down_groups = (0..levels.saturating_sub(1))
        .map(|level| {
            let source = cube_view(level, 1);
            crate::pipelines::group(
                device,
                "engine/environment-down-group",
                &ENVIRONMENT_DOWN,
                &layouts.down,
                &[
                    ("level_source", crate::pipelines::view(&source)),
                    sampler_entry.clone(),
                ],
            )
        })
        .collect();
    let every_level = cube_view(0, levels);
    let prefilter_group = crate::pipelines::group(
        device,
        "engine/environment-prefilter-group",
        &ENVIRONMENT_PREFILTER,
        &layouts.prefilter,
        &[
            ("radiance", crate::pipelines::view(&every_level)),
            sampler_entry,
        ],
    );
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
        maps_view: Versioned::new(maps_view),
        cube_targets: std::iter::repeat_with(OnceCell::new)
            .take(cubes as usize)
            .collect(),
    }
}

fn faces_pass(
    encoder: &mut wgpu::CommandEncoder,
    targets: &[wgpu::TextureView],
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
    item: u32,
) {
    let attachments: [_; FACES_PER_PASS] = std::array::from_fn(|index| {
        Some(wgpu::RenderPassColorAttachment {
            view: &targets[index],
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("engine/environment"),
        color_attachments: &attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, item..item + 1);
}

fn sky(lighting: &LightingUniform, clouds: [f32; 4], air: [f32; 4]) -> [[f32; 4]; 7] {
    [
        lighting.sun,
        lighting.sun_color,
        lighting.zenith,
        lighting.horizon,
        lighting.ground,
        clouds,
        air,
    ]
}

fn sky_group(
    device: &wgpu::Device,
    layout: &crate::pipelines::GroupLayout,
    lighting: &wgpu::Buffer,
    clouds: crate::clouds::DomeBinding<'_>,
    air: crate::sky::AirBindings<'_>,
) -> wgpu::BindGroup {
    crate::pipelines::group(
        device,
        "engine/environment-sky-group",
        &ENVIRONMENT_SKY,
        layout,
        &[
            ("lighting", lighting.as_entire_binding()),
            crate::buffers::entry("sky_air", air.0),
            ("sky_view_table", crate::pipelines::view(air.1)),
            ("sky_table_sampler", crate::pipelines::sampler(air.3)),
            ("sky_clouds", crate::pipelines::view(clouds.new)),
            ("sky_clouds_old", crate::pipelines::view(clouds.old)),
            (
                "sky_cloud_sampler",
                crate::pipelines::sampler(clouds.sampler),
            ),
            crate::buffers::entry("sky_domes", clouds.uniform),
        ],
    )
}
