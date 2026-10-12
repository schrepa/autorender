//! Depth pyramids: the farthest depth of each 2x2 block of the depth buffer,
//! then of each 2x2 block of the level below, which GPU presentation's
//! occlusion test compares candidates' projected bounds with
//! after phase one draws; or the nearest depths, which screen-space
//! reflections walk after the opaque pass.

use crate::bindings::{PYRAMID_DOWN, PYRAMID_EXTENT_BYTES, PYRAMID_FIRST};
use crate::cached::{Cached, Versioned};
use crate::gpu_data::{DEPTH_PYRAMID_SHADER, mip_levels};
use crate::gpu_layouts::constants::PYRAMID_GROUP as GROUP;
use crate::profiling::Span;
use crate::shaders::entry::depth_pyramid;
use crate::targets::FrameSize;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

struct Levels {
    // Keeps the texture and the levels' extents alive beside their views.
    _texture: wgpu::Texture,
    _extents: wgpu::Buffer,
    full: Versioned<wgpu::TextureView>,
    passes: Vec<Level>,
}

// One level's reduction: its bind group, the texels it writes and the
// level's size.
struct Level {
    group: wgpu::BindGroup,
    written: (u32, u32),
    size: (u32, u32),
}

// Each level's extent in its own uniform binding offset.
const EXTENT_STRIDE: u64 = 256;

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
    first_layout: crate::pipelines::GroupLayout,
    down_layout: crate::pipelines::GroupLayout,
    // The levels, by the depth buffer generation they reduce.
    levels: Cached<u64, Levels>,
}

impl DepthPyramid {
    pub(crate) fn new(device: &wgpu::Device, reduction: Reduction) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/depth-pyramid.wgsl"),
            source: wgpu::ShaderSource::Wgsl(DEPTH_PYRAMID_SHADER.into()),
        });
        // A level's source and the level written: both kinds of pass map to
        // the same descriptors, and so share one root signature in Direct3D
        // 12.
        let first_layout = crate::pipelines::layout(device, &PYRAMID_FIRST);
        let down_layout = crate::pipelines::layout(device, &PYRAMID_DOWN);
        let pipeline = |entry: crate::shaders::Entry,
                        (declared, group): (
            &crate::pipeline_layouts::PipelineLayout,
            &crate::pipelines::GroupLayout,
        )| {
            let layout = crate::pipelines::pipeline_layout(device, declared, &[group]);
            crate::pipelines::compute_entry(device, &layout, &module, entry)
        };
        let (first, down) = match reduction {
            Reduction::Farthest => (depth_pyramid::pyramid_first, depth_pyramid::pyramid_down),
            Reduction::Nearest => (
                depth_pyramid::pyramid_first_nearest,
                depth_pyramid::pyramid_down_nearest,
            ),
        };
        Self {
            first: pipeline(
                first,
                (
                    &crate::pipeline_layouts::PYRAMID_FIRST_PIPELINES,
                    &first_layout,
                ),
            ),
            down: pipeline(
                down,
                (
                    &crate::pipeline_layouts::PYRAMID_DOWN_PIPELINES,
                    &down_layout,
                ),
            ),
            first_layout,
            down_layout,
            levels: Cached::default(),
        }
    }

    /// Releases the levels; the next `prepare` recreates them.
    pub(crate) fn release(&mut self) {
        self.levels.clear();
    }

    /// Prepares the pyramid of the `size` depth buffer `depth`, rebuilt
    /// when the buffer is replaced; returns its level count.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        depth: &Versioned<wgpu::TextureView>,
        FrameSize { width, height }: FrameSize,
    ) -> u32 {
        let depth_generation = depth.generation();
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
        // Each level writes only the texels covering the depth buffer: the
        // ones the occlusion test and reflection rays read.
        let mut source = (width.max(1), height.max(1));
        let extents: Vec<[u32; 4]> = (0..count)
            .map(|_| {
                let written = (source.0.div_ceil(2), source.1.div_ceil(2));
                let extent = [written.0, written.1, source.0 - 1, source.1 - 1];
                source = written;
                extent
            })
            .collect();
        let mut contents = vec![0_u8; count as usize * EXTENT_STRIDE as usize];
        for (level, extent) in extents.iter().enumerate() {
            let at = level * EXTENT_STRIDE as usize;
            contents[at..at + PYRAMID_EXTENT_BYTES].copy_from_slice(bytemuck::cast_slice(extent));
        }
        let extents_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("engine/depth-pyramid-extents"),
                contents: &contents,
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let extent_binding = |level: u32| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &extents_buffer,
                offset: u64::from(level) * EXTENT_STRIDE,
                size: wgpu::BufferSize::new(PYRAMID_EXTENT_BYTES as u64),
            })
        };
        let passes = (0..count)
            .map(|level| {
                let size = ((base.0 >> level).max(1), (base.1 >> level).max(1));
                let [written_width, written_height, ..] = extents[level as usize];
                let group = if level == 0 {
                    crate::pipelines::group(
                        device,
                        "engine/depth-pyramid-first",
                        &PYRAMID_FIRST,
                        &self.first_layout,
                        &[
                            ("source_depth", crate::pipelines::view(depth)),
                            ("target_level", crate::pipelines::view(&views[0])),
                            ("extent", extent_binding(0)),
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
                            ("extent", extent_binding(level)),
                        ],
                    )
                };
                Level {
                    group,
                    written: (written_width, written_height),
                    size,
                }
            })
            .collect();
        let full = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.levels.set(
            depth_generation,
            Levels {
                _texture: texture,
                _extents: extents_buffer,
                full: Versioned::new(full),
                passes,
            },
        );
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
        for (index, level) in levels.passes.iter().enumerate() {
            let (width, height) = level.written;
            if index < 2 {
                pass.set_pipeline(if index == 0 { &self.first } else { &self.down });
            }
            pass.set_bind_group(0, &level.group, &[]);
            pass.dispatch_workgroups(width.div_ceil(GROUP), height.div_ceil(GROUP), 1);
        }
    }

    /// Every level, a view each pyramid rebuilt anew.
    pub(crate) fn view(&self) -> Option<&Versioned<wgpu::TextureView>> {
        self.levels.get().map(|levels| &levels.full)
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.levels.get().map_or(0, |levels| {
            levels
                .passes
                .iter()
                .map(|level| u64::from(level.size.0) * u64::from(level.size.1) * 4)
                .sum()
        })
    }
}
