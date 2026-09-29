//! Depth pyramids: the farthest depth of each 2x2 block of the depth buffer,
//! then of each 2x2 block of the level below, which GPU presentation's
//! occlusion test (decision 040) compares candidates' projected bounds with
//! after phase one draws; or the nearest depths, which screen-space
//! reflections (decision 025) walk after the opaque pass.

use crate::gpu_data::{DEPTH_PYRAMID_SHADER, mip_levels};

const GROUP: u32 = 8;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

struct Levels {
    // Keeps the texture alive beside its views.
    _texture: wgpu::Texture,
    full: wgpu::TextureView,
    // Per level: the reduction's bind group and the level's size.
    passes: Vec<(wgpu::BindGroup, (u32, u32))>,
}

/// Which depth each pyramid texel keeps of the block it covers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reduction {
    /// The farthest, for occlusion culling.
    Farthest,
    /// The nearest, for reflection rays.
    Nearest,
}

pub(crate) struct DepthPyramid {
    first: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    levels: Option<(u64, Levels)>,
    // Advances with every rebuilt texture, for the groups that bind it.
    generation: u64,
}

impl DepthPyramid {
    pub(crate) fn new(device: &wgpu::Device, reduction: Reduction) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/depth-pyramid.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DEPTH_PYRAMID_SHADER.into()),
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (first, down) = match reduction {
            Reduction::Farthest => ("pyramid_first", "pyramid_down"),
            Reduction::Nearest => ("pyramid_first_nearest", "pyramid_down_nearest"),
        };
        Self {
            first: pipeline(first),
            down: pipeline(down),
            levels: None,
            generation: 0,
        }
    }

    /// Releases the levels; the next `prepare` recreates them.
    pub(crate) fn release(&mut self) {
        self.levels = None;
    }

    /// Prepares the pyramid of the `size` depth buffer `depth` of generation
    /// `depth_generation`; returns its level count.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        depth: &wgpu::TextureView,
        depth_generation: u64,
        (width, height): (u32, u32),
    ) -> u32 {
        if let Some((built, levels)) = &self.levels
            && *built == depth_generation
        {
            return levels.passes.len() as u32;
        }
        // Power-of-two levels halve exactly: an odd level's last row or
        // column would otherwise fall outside every texel of the next.
        // Texels beyond the depth buffer repeat its edge, which cannot
        // change a block's farthest or nearest depth.
        let base = (
            width.div_ceil(2).max(1).next_power_of_two(),
            height.div_ceil(2).max(1).next_power_of_two(),
        );
        let count = mip_levels(base.0, base.1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("engine/depth-pyramid"),
            size: wgpu::Extent3d {
                width: base.0,
                height: base.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let level_view = |level: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let views: Vec<wgpu::TextureView> = (0..count).map(level_view).collect();
        let passes = (0..count)
            .map(|level| {
                let size = ((base.0 >> level).max(1), (base.1 >> level).max(1));
                let group = if level == 0 {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("engine/depth-pyramid-first"),
                        layout: &self.first.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(depth),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&views[0]),
                            },
                        ],
                    })
                } else {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("engine/depth-pyramid-down"),
                        layout: &self.down.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &views[level as usize - 1],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(
                                    &views[level as usize],
                                ),
                            },
                        ],
                    })
                };
                (group, size)
            })
            .collect();
        let full = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.levels = Some((
            depth_generation,
            Levels {
                _texture: texture,
                full,
                passes,
            },
        ));
        self.generation += 1;
        count
    }

    /// Records every level's reduction, level zero from the depth buffer.
    /// With `timestamp` (a query set and index), the pass writes it when it
    /// begins.
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp: Option<(&wgpu::QuerySet, u32)>,
    ) {
        let Some((_, levels)) = &self.levels else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/depth-pyramid"),
            timestamp_writes: timestamp.map(|(query_set, index)| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(index),
                    end_of_pass_write_index: None,
                }
            }),
        });
        for (level, (group, (width, height))) in levels.passes.iter().enumerate() {
            pass.set_pipeline(if level == 0 { &self.first } else { &self.down });
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(width.div_ceil(GROUP), height.div_ceil(GROUP), 1);
        }
    }

    /// Every level, and its generation.
    pub(crate) fn view(&self) -> Option<(&wgpu::TextureView, u64)> {
        self.levels
            .as_ref()
            .map(|(_, levels)| (&levels.full, self.generation))
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.levels.as_ref().map_or(0, |(_, levels)| {
            levels
                .passes
                .iter()
                .map(|(_, (width, height))| u64::from(*width) * u64::from(*height) * 4)
                .sum()
        })
    }
}
