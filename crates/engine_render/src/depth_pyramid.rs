use crate::bindings::{PYRAMID_DOWN, PYRAMID_EXTENT_BYTES, PYRAMID_FIRST};
use crate::cached::Versioned;
use crate::gpu_data::mip_levels;
use crate::gpu_layouts::constants::PYRAMID_GROUP as GROUP;
use crate::profiling::Span;
use crate::shaders::entry::depth_pyramid;
use crate::targets::{FrameRegion, FrameSize};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

struct Levels {
    texture: wgpu::Texture,
    extents: wgpu::Buffer,
    full: Versioned<wgpu::TextureView>,
    passes: Vec<Level>,
    drawn: FrameSize,
}

impl Drop for Levels {
    fn drop(&mut self) {
        crate::memory::release_any(&self.texture);
        crate::memory::release_buffer(&self.extents);
    }
}

struct Level {
    group: wgpu::BindGroup,
    written: (u32, u32),
    size: (u32, u32),
}

const EXTENT_STRIDE: u64 = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    Farthest,
    Nearest,
}

pub struct DepthPyramid {
    first: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    first_layout: crate::pipelines::GroupLayout,
    down_layout: crate::pipelines::GroupLayout,
    levels: Option<(u64, Levels)>,
}

impl DepthPyramid {
    pub fn new(device: &wgpu::Device, reduction: Reduction) -> Self {
        let first_layout = crate::pipelines::layout(device, &PYRAMID_FIRST);
        let down_layout = crate::pipelines::layout(device, &PYRAMID_DOWN);
        let pipeline =
            |entry: crate::shaders::Entry,
             declared: &'static crate::pipeline_layouts::PipelineLayout| {
                let layout = crate::pipelines::layout_of(device, declared);
                crate::pipelines::compute_entry(device, &layout, entry)
            };
        let (first, down) = match reduction {
            Reduction::Farthest => (depth_pyramid::pyramid_first, depth_pyramid::pyramid_down),
            Reduction::Nearest => (
                depth_pyramid::pyramid_first_nearest,
                depth_pyramid::pyramid_down_nearest,
            ),
        };
        Self {
            first: pipeline(first, &crate::pipeline_layouts::PYRAMID_FIRST_PIPELINES),
            down: pipeline(down, &crate::pipeline_layouts::PYRAMID_DOWN_PIPELINES),
            first_layout,
            down_layout,
            levels: None,
        }
    }

    pub fn release(&mut self) {
        self.levels = None;
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
        depth: &Versioned<wgpu::TextureView>,
        region: FrameRegion,
    ) -> u32 {
        let generation = depth.generation();
        if self
            .levels
            .as_ref()
            .is_none_or(|(built, _)| *built != generation)
        {
            let built = self.build(device, depth, region);
            self.levels = Some((generation, built));
        }
        let Some((_, levels)) = &mut self.levels else {
            return 0;
        };
        if levels.drawn != region.render {
            levels.reduce(queue, region.render);
        }
        levels.passes.len() as u32
    }

    fn build(
        &self,
        device: &wgpu::Device,
        depth: &Versioned<wgpu::TextureView>,
        region: FrameRegion,
    ) -> Levels {
        let FrameSize { width, height } = region.display;
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
        let reduced = extents(region.render, count);
        let extents_buffer = crate::memory::buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("engine/depth-pyramid-extents"),
                contents: &extent_bytes(&reduced),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
                let [written_width, written_height, ..] = reduced[level as usize];
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
        Levels {
            texture,
            extents: extents_buffer,
            full: Versioned::new(full),
            passes,
            drawn: region.render,
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, span: Option<Span<'_>>) {
        let Some((_, levels)) = &self.levels else {
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

    pub fn view(&self) -> Option<&Versioned<wgpu::TextureView>> {
        self.levels.as_ref().map(|(_, levels)| &levels.full)
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.levels.as_ref().map_or(0, |(_, levels)| {
            levels
                .passes
                .iter()
                .map(|level| u64::from(level.size.0) * u64::from(level.size.1) * 4)
                .sum()
        })
    }
}

impl Levels {
    fn reduce(&mut self, queue: &crate::gpu_calls::Queue, drawn: FrameSize) {
        let reduced = extents(drawn, self.passes.len() as u32);
        queue.write_buffer(&self.extents, 0, &extent_bytes(&reduced));
        for (level, extent) in self.passes.iter_mut().zip(&reduced) {
            level.written = (extent[0], extent[1]);
        }
        self.full.replace(
            self.texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
        self.drawn = drawn;
    }
}

fn extents(drawn: FrameSize, count: u32) -> Vec<[u32; 4]> {
    let mut source = (drawn.width.max(1), drawn.height.max(1));
    (0..count)
        .map(|_| {
            let written = (source.0.div_ceil(2), source.1.div_ceil(2));
            let extent = [written.0, written.1, source.0 - 1, source.1 - 1];
            source = written;
            extent
        })
        .collect()
}

fn extent_bytes(extents: &[[u32; 4]]) -> Vec<u8> {
    let mut contents = vec![0_u8; extents.len() * EXTENT_STRIDE as usize];
    for (level, extent) in extents.iter().enumerate() {
        let at = level * EXTENT_STRIDE as usize;
        contents[at..at + PYRAMID_EXTENT_BYTES].copy_from_slice(bytemuck::cast_slice(extent));
    }
    contents
}
