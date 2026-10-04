//! Depth pyramids: the farthest depth of each 2x2 block of the depth buffer,
//! then of each 2x2 block of the level below, which GPU presentation's
//! occlusion test (decision 040) compares candidates' projected bounds with
//! after phase one draws; or the nearest depths, which screen-space
//! reflections (decision 025) walk after the opaque pass.

use crate::bindings::{PYRAMID_DOWN, PYRAMID_FIRST};
use crate::cached::Cached;
use crate::gpu_data::{DEPTH_PYRAMID_SHADER, mip_levels};
use crate::profiling::Span;

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
    // The group layouts of the first level, which reads the depth buffer,
    // and of the levels after it, which read the level below.
    first_layout: wgpu::BindGroupLayout,
    down_layout: wgpu::BindGroupLayout,
    // The levels, by the depth buffer generation they reduce.
    levels: Cached<u64, Levels>,
    // Advances with every rebuilt texture, for the groups that bind it.
    generation: u64,
}

impl DepthPyramid {
    pub(crate) fn new(device: &wgpu::Device, reduction: Reduction) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/depth-pyramid.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DEPTH_PYRAMID_SHADER.into()),
        });
        // A level's source and the level written: both kinds of pass map to
        // the same descriptors, and so share one root signature in Direct3D
        // 12 (decision 049).
        let first_layout = crate::pipelines::layout(device, &PYRAMID_FIRST);
        let down_layout = crate::pipelines::layout(device, &PYRAMID_DOWN);
        let pipeline = |entry: &str, group: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: &[Some(group)],
                immediate_size: 0,
            });
            crate::pipelines::compute_entry(device, &layout, &module, entry)
        };
        let (first, down) = match reduction {
            Reduction::Farthest => ("pyramid_first", "pyramid_down"),
            Reduction::Nearest => ("pyramid_first_nearest", "pyramid_down_nearest"),
        };
        Self {
            first: pipeline(first, &first_layout),
            down: pipeline(down, &down_layout),
            first_layout,
            down_layout,
            levels: Cached::default(),
            generation: 0,
        }
    }

    /// Releases the levels; the next `prepare` recreates them.
    pub(crate) fn release(&mut self) {
        self.levels.clear();
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
        if !self.levels.stale(&depth_generation) {
            return self
                .levels
                .get()
                .map_or(0, |levels| levels.passes.len() as u32);
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
        let texture = crate::targets::texture_2d(
            device,
            "engine/depth-pyramid",
            (base.0, base.1),
            1,
            count,
            FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
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
                    crate::pipelines::group(
                        device,
                        "engine/depth-pyramid-first",
                        &PYRAMID_FIRST,
                        &self.first_layout,
                        &[
                            ("source_depth", crate::pipelines::view(depth)),
                            ("target_level", crate::pipelines::view(&views[0])),
                        ],
                    )
                } else {
                    crate::pipelines::group(
                        device,
                        "engine/depth-pyramid-down",
                        &PYRAMID_DOWN,
                        &self.down_layout,
                        &[
                            (
                                "source_level",
                                crate::pipelines::view(&views[level as usize - 1]),
                            ),
                            (
                                "target_level",
                                crate::pipelines::view(&views[level as usize]),
                            ),
                        ],
                    )
                };
                (group, size)
            })
            .collect();
        let full = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.levels.set(
            depth_generation,
            Levels {
                _texture: texture,
                full,
                passes,
            },
        );
        self.generation += 1;
        count
    }

    /// Records every level's reduction, level zero from the depth buffer.
    /// With `span`, the pass begins it.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, span: Option<Span<'_>>) {
        let Some(levels) = self.levels.get() else {
            return;
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/depth-pyramid"),
            timestamp_writes: span.map(|span| span.compute_part(true, false)),
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
            .get()
            .map(|levels| (&levels.full, self.generation))
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.levels.get().map_or(0, |levels| {
            levels
                .passes
                .iter()
                .map(|(_, (width, height))| u64::from(*width) * u64::from(*height) * 4)
                .sum()
        })
    }
}
