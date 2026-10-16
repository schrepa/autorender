//! Development-only, requested copies. Never part of the default render path.

use std::cell::RefCell;

use super::*;
use engine_core::{DepthLightEntry, DepthOccupancyPage};

const PAGE_ENTRIES: usize = 128;
const WORDS: usize = 47 + PAGE_ENTRIES * 6;
const BYTES: u64 = (WORDS as u64 * 4).div_ceil(16) * 16;
type Sample = (u64, [u32; WORDS]);

pub(super) struct Diagnostic {
    uniform: wgpu::Buffer,
    output: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    key: [u32; 3],
    ring: ReadbackRing<u64>,
    sample: Rc<RefCell<Option<Sample>>>,
}

impl Diagnostic {
    fn new(device: &wgpu::Device) -> Self {
        let uniform = crate::buffers::uniform(device, "engine/light-depth-request", 16);
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("engine/light-depth-page"),
            size: BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let filter_layout = crate::pipelines::layout(device, &LIGHT_DEPTH_FILTER);
        let table = &crate::bindings::LIGHT_DEPTH_DIAGNOSTIC;
        let request_layout = crate::pipelines::layout(device, table);
        let group = crate::pipelines::group(
            device,
            "engine/light-depth-page",
            table,
            &request_layout,
            &[
                entry("depth_request", &uniform),
                entry("depth_page", &output),
            ],
        );
        let layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::LIGHT_DEPTH_DIAGNOSTIC_PIPELINES,
            &[&filter_layout, &request_layout],
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/light-depth-diagnostic"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::LIGHT_DEPTH_SHADER.into()),
        });
        let pipeline = crate::pipelines::compute_entry(
            device,
            &layout,
            &module,
            light_depth::inspect_light_depth,
        );
        Self {
            uniform,
            output,
            group,
            pipeline,
            key: [u32::MAX; 3],
            ring: ReadbackRing::new("engine/light-depth-page"),
            sample: Rc::new(RefCell::new(None)),
        }
    }

    fn request(&mut self, queue: &wgpu::Queue, key: [u32; 3]) {
        if self.key == key {
            return;
        }
        self.key = key;
        self.ring.clear();
        self.sample = Rc::new(RefCell::new(None));
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::cast_slice(&[key[1], key[2], 0, 0]),
        );
    }

    pub(super) fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        phase: usize,
        group: &wgpu::BindGroup,
        frame: u64,
    ) {
        if phase as u32 != self.key[0] {
            return;
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/light-depth-page"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.set_bind_group(1, &self.group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.ring
            .record(device, encoder, &self.output, BYTES, frame);
    }

    pub(super) fn after_submit(&self) {
        let sample = Rc::clone(&self.sample);
        self.ring.after_submit(move |words, frame| {
            let mut latest = sample.borrow_mut();
            if latest.as_ref().is_none_or(|(old, _)| *old <= frame) {
                *latest = Some((frame, std::array::from_fn(|index| words.get(index))));
            }
        });
    }

    pub(super) fn bytes(&self) -> u64 {
        self.uniform.size() + self.output.size() + self.ring.capacity_bytes()
    }
}

impl DepthPass {
    pub(crate) fn page(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        key: [u32; 3],
        allowed_bytes: Option<u64>,
    ) -> Option<DepthOccupancyPage> {
        if key[0] > 1 || u64::from(key[1]) >= CLUSTERS {
            return None;
        }
        let (view, uniform) = self.context?;
        let mut diagnostic = self.diagnostic.borrow_mut();
        let diagnostic = diagnostic.get_or_insert_with(|| Diagnostic::new(device));
        diagnostic.request(queue, key);
        let (frame, words) = (*diagnostic.sample.borrow())?;
        let original = (0..PAGE_ENTRIES)
            .filter_map(|lane| {
                let start = 47 + lane * 6;
                (words[start] != u32::MAX).then(|| DepthLightEntry {
                    id: words[start],
                    position_bits: [words[start + 2], words[start + 3], words[start + 4]],
                    range_bits: words[start + 5],
                })
            })
            .collect();
        let filtered = (0..PAGE_ENTRIES)
            .map(|lane| words[48 + lane * 6])
            .filter(|&id| id != u32::MAX)
            .collect();
        let limit = device
            .limits()
            .max_storage_buffer_binding_size
            .min(device.limits().max_buffer_size);
        let allowed_bytes = allowed_bytes.unwrap_or(limit).min(limit);
        Some(DepthOccupancyPage {
            full_routes: [
                "transparent",
                "terrain",
                "decals",
                "impostors",
                "ground_cover",
                "visibility_resolve",
                "probes",
                "water",
                "particles",
            ]
            .into_iter()
            .map(|consumer| engine_core::DepthFullRoute {
                consumer,
                occupancy_override: false,
                generation: uniform.lights[1],
            })
            .collect(),
            view,
            phase: words[1],
            generation: words[0],
            readback_age: self.frame.saturating_sub(frame),
            cluster: words[2],
            mask: words[3],
            threshold_bits: words[16..47].to_vec(),
            original_length: words[5],
            filtered_length: words[7],
            offset: words[8],
            original,
            filtered,
            full_generation: uniform.lights[1],
            index_capacity: uniform.lights[2],
            arena_bytes: self.arena.size(),
            allowed_bytes,
            allocation_fits: arena_bytes(uniform.lights[2], allowed_bytes).is_some(),
        })
    }
}
