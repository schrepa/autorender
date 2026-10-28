#![expect(
    clippy::expect_used,
    reason = "drawing reads the targets its preparation sized"
)]

use crate::bindings::VISIBILITY_CLASSIFY;
use crate::buffers::{storage, uniform};
use crate::cached::{Bound, GroupCache, Versioned};
use crate::gpu_data::VISIBILITY_SHADER;
use crate::gpu_layouts::constants::RESOLVE_ARGS_GROUP;
use crate::gpu_layouts::constants::VISIBILITY_TILE as TILE;
use crate::shaders::entry::visibility;
use crate::targets::{FrameRegion, FrameSize, ScreenTarget};

pub(crate) const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;
pub(crate) const DRAW_STRIDE: u64 = 256;

pub(crate) struct VisibilityBuffer {
    layout: crate::pipelines::GroupLayout,
    classify: wgpu::ComputePipeline,
    args: wgpu::ComputePipeline,
    ids: ScreenTarget<wgpu::TextureView>,
    tiles: (u32, u32),
    params: wgpu::Buffer,
    counts: Versioned<wgpu::Buffer>,
    lists: Versioned<wgpu::Buffer>,
    draw_args: Versioned<wgpu::Buffer>,
    draws: Versioned<wgpu::Buffer>,
    held: (usize, (u32, u32)),
    group: GroupCache<5>,
}

fn ids_target(device: &wgpu::Device, (width, height): (u32, u32)) -> wgpu::Texture {
    crate::targets::texture_2d(
        device,
        "engine/visibility-ids",
        (width, height),
        1,
        1,
        ID_FORMAT,
        crate::targets::RENDER_TARGET,
    )
}

impl VisibilityBuffer {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/visibility.wgsl"),
            source: wgpu::ShaderSource::Wgsl(VISIBILITY_SHADER.into()),
        });
        let layout = crate::pipelines::layout(device, &VISIBILITY_CLASSIFY);
        let pipeline_layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::VISIBILITY_CLASSIFY_PIPELINES,
            &[&layout],
        );
        let pipeline = |entry: crate::shaders::Entry| {
            crate::pipelines::compute_entry(device, &pipeline_layout, &module, entry)
        };
        let none = wgpu::BufferUsages::empty();
        Self {
            classify: pipeline(visibility::classify_tiles),
            args: pipeline(visibility::resolve_args),
            layout,
            ids: ScreenTarget::default(),
            tiles: (0, 0),
            params: uniform(device, "engine/visibility-classify", 32),
            counts: Versioned::new(storage(device, "engine/visibility-counts", 16, none)),
            lists: Versioned::new(storage(device, "engine/visibility-tiles", 16, none)),
            draw_args: Versioned::new(storage(
                device,
                "engine/visibility-args",
                16,
                wgpu::BufferUsages::INDIRECT,
            )),
            draws: Versioned::new(uniform(device, "engine/visibility-draws", DRAW_STRIDE)),
            held: (0, (0, 0)),
            group: GroupCache::default(),
        }
    }

    pub(crate) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        region: FrameRegion,
        visible_bases: [&[u32]; 2],
    ) -> bool {
        let target = FrameSize {
            width: region.display.width.max(1),
            height: region.display.height.max(1),
        };
        let (width, height) = (region.render.width.max(1), region.render.height.max(1));
        let bins = visible_bases[0].len();
        self.tiles = (width.div_ceil(TILE), height.div_ceil(TILE));
        let room = (target.width.div_ceil(TILE), target.height.div_ceil(TILE));
        let capacity = u64::from(room.0) * u64::from(room.1);
        let limits = device.limits();
        let binding = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        if bins as u64 * capacity * 4 > binding {
            return false;
        }
        if self.ids.resizes(target) {
            let view = ids_target(device, target.pixels()).create_view(&Default::default());
            self.ids.set(target, view);
        }
        if bins > self.held.0 || room != self.held.1 {
            let none = wgpu::BufferUsages::empty();
            self.counts.replace(storage(
                device,
                "engine/visibility-counts",
                bins as u64 * 4,
                none,
            ));
            self.lists.replace(storage(
                device,
                "engine/visibility-tiles",
                bins as u64 * capacity * 4,
                none,
            ));
            self.draw_args.replace(storage(
                device,
                "engine/visibility-args",
                bins as u64 * 16,
                wgpu::BufferUsages::INDIRECT,
            ));
            self.draws.replace(uniform(
                device,
                "engine/visibility-draws",
                2 * bins.max(1) as u64 * DRAW_STRIDE,
            ));
            self.held = (bins, room);
        }
        let ids = self.ids.get().expect("sized above");
        self.group.bind(
            device,
            "engine/visibility-classify-group",
            (&VISIBILITY_CLASSIFY, &self.layout),
            [
                ("classify", Bound::fixed(self.params.as_entire_binding())),
                ("ids", ids.bound()),
                ("counts", self.counts.bound()),
                ("tiles", self.lists.bound()),
                ("draw_args", self.draw_args.bound()),
            ],
        );
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

    pub(crate) fn encode_classify(&self, encoder: &mut wgpu::CommandEncoder, bins: usize) {
        let Some(group) = self.group.get() else {
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
        pass.dispatch_workgroups((bins as u32).div_ceil(RESOLVE_ARGS_GROUP).max(1), 1, 1);
    }

    pub(crate) fn draw_offset(phase: usize, bins: usize, bin: usize) -> u32 {
        ((phase * bins + bin) as u64 * DRAW_STRIDE) as u32
    }

    pub(crate) fn ids_view(&self) -> &Versioned<wgpu::TextureView> {
        self.ids.get().expect("prepared before drawing")
    }

    pub(crate) fn buffers(
        &self,
    ) -> (
        &Versioned<wgpu::Buffer>,
        &Versioned<wgpu::Buffer>,
        &Versioned<wgpu::Buffer>,
    ) {
        (&self.draws, &self.lists, &self.draw_args)
    }

    pub(crate) fn capacity_bytes(&self) -> u64 {
        self.ids
            .size()
            .map_or(0, |size| u64::from(size.width) * u64::from(size.height) * 8)
            + self.counts.size()
            + self.lists.size()
            + self.draw_args.size()
            + self.draws.size()
    }
}
