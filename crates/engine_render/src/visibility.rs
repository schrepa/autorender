//! The visibility buffer (decision 045): the camera's virtual geometry
//! clusters (decision 044) rasterize which triangle covers each pixel
//! (`meshes.wgsl`'s `vs_visibility`) into an ID target; compute lists the
//! 8x8 tiles holding each material bin's pixels (`visibility.wgsl`); and
//! each bin's resolve (`fs_resolve*`) shades its pixels once, however many
//! triangles overlapped them or however small they were.

use crate::gpu_data::VISIBILITY_SHADER;

/// Each pixel's visible entry plus one (0: none) and its triangle | bin << 7.
pub(crate) const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
/// Pixels across a tile's side.
const TILE: u32 = 8;
/// Bytes between draws' uniforms: the uniform offset alignment every device
/// supports.
pub(crate) const DRAW_STRIDE: u64 = 256;
/// Storage buffers the resolve's fragment stage binds: poses, colors,
/// previous poses, the arena's meshes, clusters and pages, the selected
/// clusters and the camera's reflection probe grid; lit frames add the local
/// lights' three.
pub(crate) const RESOLVE_STORAGE_BUFFERS: u32 = 8;
pub(crate) const RESOLVE_LIT_STORAGE_BUFFERS: u32 = 11;

/// The ID target, the tile lists and each draw's uniform, sized to the
/// viewport and the frame's bins.
pub(crate) struct VisibilityBuffer {
    layout: wgpu::BindGroupLayout,
    classify: wgpu::ComputePipeline,
    args: wgpu::ComputePipeline,
    ids: wgpu::Texture,
    ids_view: wgpu::TextureView,
    // Tiles across and down the viewport.
    tiles: (u32, u32),
    // The classification's uniform, each bin's tile count, the tile lists
    // (each bin's from `bin * capacity`), each bin's resolve arguments, and
    // the draws' uniforms (phase one's bins, then phase two's).
    params: wgpu::Buffer,
    counts: wgpu::Buffer,
    lists: wgpu::Buffer,
    draw_args: wgpu::Buffer,
    draws: wgpu::Buffer,
    // The bins and tile grid the buffers hold.
    held: (usize, (u32, u32)),
    // Advances whenever the target or a buffer is replaced.
    generation: u64,
    group: Option<(u64, wgpu::BindGroup)>,
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

fn uniform(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn ids_target(device: &wgpu::Device, (width, height): (u32, u32)) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("engine/visibility-ids"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ID_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

impl VisibilityBuffer {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/visibility.wgsl"),
            source: wgpu::ShaderSource::Wgsl(VISIBILITY_SHADER.into()),
        });
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let read_write = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("engine/visibility-classify-layout"),
            entries: &[
                buffer(0, wgpu::BufferBindingType::Uniform),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                buffer(2, read_write),
                buffer(3, read_write),
                buffer(4, read_write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("engine/visibility-classify-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline =
            |entry: &str| crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry);
        let ids = ids_target(device, (1, 1));
        let none = wgpu::BufferUsages::empty();
        Self {
            classify: pipeline("classify_tiles"),
            args: pipeline("resolve_args"),
            layout,
            ids_view: ids.create_view(&Default::default()),
            ids,
            tiles: (0, 0),
            params: uniform(device, "engine/visibility-classify", 32),
            counts: storage(device, "engine/visibility-counts", 16, none),
            lists: storage(device, "engine/visibility-tiles", 16, none),
            draw_args: storage(
                device,
                "engine/visibility-args",
                16,
                wgpu::BufferUsages::INDIRECT,
            ),
            draws: uniform(device, "engine/visibility-draws", DRAW_STRIDE),
            held: (0, (0, 0)),
            generation: 0,
            group: None,
        }
    }

    /// Readies the target and buffers for a viewport of `size` and the bins
    /// whose first selected cluster in each phase `visible_bases` holds
    /// (phase one's, then phase two's), and writes the draws' uniforms.
    /// False when the tile lists would exceed the device's storage binding
    /// size: the frame draws its bins forward instead.
    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        visible_bases: [&[u32]; 2],
    ) -> bool {
        let (width, height) = (size.0.max(1), size.1.max(1));
        let bins = visible_bases[0].len();
        self.tiles = (width.div_ceil(TILE), height.div_ceil(TILE));
        let capacity = u64::from(self.tiles.0) * u64::from(self.tiles.1);
        let limits = device.limits();
        let binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        if bins as u64 * capacity * 4 > binding {
            return false;
        }
        let extent = self.ids.size();
        if (extent.width, extent.height) != (width, height) {
            self.ids = ids_target(device, (width, height));
            self.ids_view = self.ids.create_view(&Default::default());
            self.generation += 1;
        }
        if bins > self.held.0 || self.tiles != self.held.1 {
            let none = wgpu::BufferUsages::empty();
            self.counts = storage(device, "engine/visibility-counts", bins as u64 * 4, none);
            self.lists = storage(
                device,
                "engine/visibility-tiles",
                bins as u64 * capacity * 4,
                none,
            );
            self.draw_args = storage(
                device,
                "engine/visibility-args",
                bins as u64 * 16,
                wgpu::BufferUsages::INDIRECT,
            );
            self.draws = uniform(
                device,
                "engine/visibility-draws",
                2 * bins.max(1) as u64 * DRAW_STRIDE,
            );
            self.held = (bins, self.tiles);
            self.generation += 1;
        }
        if self
            .group
            .as_ref()
            .is_none_or(|(built, _)| *built != self.generation)
        {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("engine/visibility-classify-group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.ids_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.lists.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.draw_args.as_entire_binding(),
                    },
                ],
            });
            self.group = Some((self.generation, group));
        }
        let capacity = capacity as u32;
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::cast_slice(&[width, height, self.tiles.0, capacity, bins as u32, 0, 0, 0]),
        );
        let stride = DRAW_STRIDE as usize / 4;
        let mut words = vec![0u32; 2 * bins * stride];
        for (phase, bases) in visible_bases.iter().enumerate() {
            for (bin, &base) in bases.iter().enumerate() {
                let slot = (phase * bins + bin) * stride;
                words[slot..slot + 4].copy_from_slice(&[
                    base,
                    bin as u32,
                    bin as u32 * capacity,
                    self.tiles.0,
                ]);
            }
        }
        if !words.is_empty() {
            queue.write_buffer(&self.draws, 0, bytemuck::cast_slice(&words));
        }
        true
    }

    /// Records the tile lists of the first `bins` bins from the ID target,
    /// and their resolve arguments.
    pub(crate) fn encode_classify(&self, encoder: &mut wgpu::CommandEncoder, bins: usize) {
        let Some((_, group)) = &self.group else {
            return;
        };
        encoder.clear_buffer(&self.counts, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/visibility-classify"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, group, &[]);
        pass.set_pipeline(&self.classify);
        pass.dispatch_workgroups(self.tiles.0, self.tiles.1, 1);
        pass.set_pipeline(&self.args);
        pass.dispatch_workgroups((bins as u32).div_ceil(64).max(1), 1, 1);
    }

    /// The dynamic offset of the uniform of bin `bin`'s draws in phase
    /// `phase` of `bins` bins.
    pub(crate) fn draw_offset(phase: usize, bins: usize, bin: usize) -> u32 {
        ((phase * bins + bin) as u64 * DRAW_STRIDE) as u32
    }

    pub(crate) fn ids_view(&self) -> &wgpu::TextureView {
        &self.ids_view
    }

    /// The draws' uniforms, the tile lists and the resolve arguments.
    pub(crate) fn buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer) {
        (&self.draws, &self.lists, &self.draw_args)
    }

    /// Advances whenever the ID target or a buffer is replaced.
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        let extent = self.ids.size();
        u64::from(extent.width) * u64::from(extent.height) * 8
            + self.counts.size()
            + self.lists.size()
            + self.draw_args.size()
            + self.draws.size()
    }
}
