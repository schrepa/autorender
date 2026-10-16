//! Optional current-phase depth occupancy. The full binner owns the arena's
//! prefix; collection and stable filtering own its tail, in separate passes.

use std::{cell::Cell, rc::Rc};

use engine_core::DepthOccupancyStats;

use crate::bindings::{LIGHT_DEPTH_COLLECTION, LIGHT_DEPTH_FILTER, LIGHT_SHADING};
use crate::buffers::{ReadbackRing, entry};
use crate::lights::{ClusterUniform, SLICES, TILES_X, TILES_Y};
use crate::pipelines::GroupLayout;
use crate::shaders::entry::light_depth;
use crate::targets::FrameSize;

#[cfg(feature = "dev")]
mod diagnostic;

pub(crate) const CLUSTERS: u64 = (TILES_X * TILES_Y * SLICES) as u64;
const THRESHOLDS: usize = 31 * SLICES as usize;
pub(crate) const THRESHOLD_OFFSET: u64 = 5 * CLUSTERS * 4;
pub(crate) const HEADER_OFFSET: u64 = THRESHOLD_OFFSET + THRESHOLDS as u64 * 4;
const HEADER_BYTES: u64 = 8 * 4;

/// Additional storage must coexist with the unchanged full index capacity.
/// `limit` is the device's smaller buffer/binding limit, also used by dev probes.
pub(crate) fn arena_bytes(capacity: u32, limit: u64) -> Option<u64> {
    let bytes = HEADER_OFFSET
        .checked_add(HEADER_BYTES)?
        .checked_add(u64::from(capacity).checked_mul(4)?)?
        .checked_add(15)?
        & !15;
    (bytes <= limit).then_some(bytes)
}

fn domain(value: f32, bound: f32) -> bool {
    (value == 0.0 || value.is_normal()) && value.abs() <= bound
}

/// Slice bounds only choose useful thresholds; open-ended extreme bins keep
/// correctness independent of the original logarithmic index's rounding.
pub(crate) fn thresholds(uniform: &ClusterUniform) -> Option<[f32; THRESHOLDS]> {
    if uniform.counts[..3] != [TILES_X, TILES_Y, SLICES]
        || !uniform.depth[..3].iter().all(|&value| domain(value, 2.0))
        || !domain(uniform.depth[3], 2_097_152.0)
        || !uniform.grid.iter().all(|value| value.is_finite())
        || uniform.grid[2] <= 0.0
    {
        return None;
    }
    let mut table = [0.0; THRESHOLDS];
    for slice in 0..SLICES as usize {
        let near = if slice == 0 {
            0.0
        } else {
            ((slice as f32 - uniform.grid[3]) / uniform.grid[2]).exp2()
        };
        let far = ((slice as f32 + 1.0 - uniform.grid[3]) / uniform.grid[2]).exp2();
        let row = &mut table[slice * 31..(slice + 1) * 31];
        for (bin, value) in row.iter_mut().enumerate() {
            *value = near + (far - near) * ((bin + 1) as f32 / 32.0);
        }
        if !row.iter().all(|&value| domain(value, f32::MAX))
            || row.windows(2).any(|pair| pair[0] > pair[1])
        {
            return None;
        }
    }
    Some(table)
}

pub(crate) struct DepthBuffers<'a> {
    pub(crate) arena: &'a wgpu::Buffer,
    pub(crate) lights: &'a wgpu::Buffer,
    pub(crate) indices: &'a wgpu::Buffer,
    pub(crate) shading: &'a GroupLayout,
}

#[derive(Clone, Copy)]
struct CounterSample {
    frame: u64,
    words: [u32; 8],
}

struct DepthPhase {
    uniform: wgpu::Buffer,
    collection: wgpu::BindGroup,
    filter: wgpu::BindGroup,
    shading: wgpu::BindGroup,
    readbacks: ReadbackRing<u64>,
    counters: Rc<Cell<Option<CounterSample>>>,
}

impl DepthPhase {
    fn new(
        device: &wgpu::Device,
        (collection, filter): (&GroupLayout, &GroupLayout),
        buffers: &DepthBuffers<'_>,
    ) -> Self {
        let uniform = crate::buffers::uniform(
            device,
            "engine/light-depth-phase",
            size_of::<ClusterUniform>() as u64,
        );
        let group = |label, table, layout, entries: &[_]| {
            crate::pipelines::group(device, label, table, layout, entries)
        };
        Self {
            collection: group(
                "engine/light-depth-collection-group",
                &LIGHT_DEPTH_COLLECTION,
                collection,
                &[
                    entry("clusters", &uniform),
                    entry("light_depth_arena", buffers.arena),
                ],
            ),
            filter: group(
                "engine/light-depth-filter-group",
                &LIGHT_DEPTH_FILTER,
                filter,
                &[
                    entry("clusters", &uniform),
                    entry("lights", buffers.lights),
                    entry("full_indices", buffers.indices),
                    entry("light_depth_arena", buffers.arena),
                ],
            ),
            shading: group(
                "engine/light-depth-shading-group",
                &LIGHT_SHADING,
                buffers.shading,
                &[
                    entry("clusters", &uniform),
                    entry("lights", buffers.lights),
                    entry("cluster_ranges", buffers.arena),
                    entry("cluster_lights", buffers.indices),
                ],
            ),
            uniform,
            readbacks: ReadbackRing::new("engine/light-depth-counters"),
            counters: Rc::new(Cell::new(None)),
        }
    }

    fn invalidate(&mut self) {
        self.readbacks.clear();
        // An old map's callback keeps the old cell, never this view's cell.
        self.counters = Rc::new(Cell::new(None));
    }
}

pub(crate) struct DepthPass {
    pub(crate) storage_generation: u64,
    pub(crate) size: FrameSize,
    pub(crate) collection_layout: GroupLayout,
    reset: wgpu::ComputePipeline,
    filter: wgpu::ComputePipeline,
    phases: [DepthPhase; 2],
    scratch: wgpu::Texture,
    pub(crate) scratch_view: wgpu::TextureView,
    arena: wgpu::Buffer,
    context: Option<(u64, ClusterUniform)>,
    table: Option<[f32; THRESHOLDS]>,
    frame: u64,
    eligible_draws: u32,
    #[cfg(feature = "dev")]
    diagnostic: std::cell::RefCell<Option<diagnostic::Diagnostic>>,
}

impl DepthPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        size: FrameSize,
        storage_generation: u64,
        buffers: DepthBuffers<'_>,
    ) -> Self {
        let collection_layout = crate::pipelines::layout(device, &LIGHT_DEPTH_COLLECTION);
        let filter_layout = crate::pipelines::layout(device, &LIGHT_DEPTH_FILTER);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("engine/light-depth-filter"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::LIGHT_DEPTH_SHADER.into()),
        });
        let layout = crate::pipelines::pipeline_layout(
            device,
            &crate::pipeline_layouts::LIGHT_DEPTH_FILTER_PIPELINES,
            &[&filter_layout],
        );
        let reset = crate::pipelines::compute_entry(
            device,
            &layout,
            &module,
            light_depth::reset_light_depth,
        );
        let filter = crate::pipelines::compute_entry(
            device,
            &layout,
            &module,
            light_depth::filter_light_depth,
        );
        let phases = std::array::from_fn(|_| {
            DepthPhase::new(device, (&collection_layout, &filter_layout), &buffers)
        });
        let scratch = crate::targets::texture_2d(
            device,
            "engine/light-depth-scratch",
            size.pixels(),
            1,
            1,
            crate::renderer::DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let scratch_view = scratch.create_view(&Default::default());
        Self {
            storage_generation,
            size,
            collection_layout,
            reset,
            filter,
            phases,
            scratch,
            scratch_view,
            arena: buffers.arena.clone(),
            context: None,
            table: None,
            frame: 0,
            eligible_draws: 0,
            #[cfg(feature = "dev")]
            diagnostic: std::cell::RefCell::new(None),
        }
    }

    pub(crate) fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        context: (u64, ClusterUniform),
        table: [f32; THRESHOLDS],
        tags: [u32; 2],
        eligible_draws: u32,
    ) {
        if self.context != Some(context) || tags.contains(&1) {
            for phase in &mut self.phases {
                phase.invalidate();
            }
            #[cfg(feature = "dev")]
            {
                *self.diagnostic.borrow_mut() = None;
            }
        }
        self.context = Some(context);
        self.frame += 1;
        self.eligible_draws = eligible_draws;
        if self.table != Some(table) {
            queue.write_buffer(&self.arena, THRESHOLD_OFFSET, bytemuck::cast_slice(&table));
            self.table = Some(table);
        }
        for (index, phase) in self.phases.iter().enumerate() {
            let mut uniform = context.1;
            uniform.lights[1] = tags[index];
            uniform.lights[3] = index as u32 + 1;
            queue.write_buffer(&phase.uniform, 0, bytemuck::bytes_of(&uniform));
        }
    }

    pub(crate) fn collection_group(&self, phase: usize) -> &wgpu::BindGroup {
        &self.phases[phase].collection
    }

    pub(crate) fn shading_group(&self, phase: usize) -> &wgpu::BindGroup {
        &self.phases[phase].shading
    }

    pub(crate) fn reset(&self, encoder: &mut wgpu::CommandEncoder, phase: usize) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("engine/light-depth-reset"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.reset);
        pass.set_bind_group(0, &self.phases[phase].filter, &[]);
        pass.dispatch_workgroups((CLUSTERS as u32).div_ceil(64), 1, 1);
    }

    pub(crate) fn filter(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        phase: usize,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/light-depth-filter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.filter);
            pass.set_bind_group(0, &self.phases[phase].filter, &[]);
            pass.dispatch_workgroups(CLUSTERS as u32, 1, 1);
        }
        self.phases[phase].readbacks.record_at(
            device,
            encoder,
            (&self.arena, HEADER_OFFSET),
            HEADER_BYTES,
            self.frame,
        );
        #[cfg(feature = "dev")]
        if let Some(diagnostic) = self.diagnostic.borrow().as_ref() {
            diagnostic.record(
                device,
                encoder,
                phase,
                &self.phases[phase].filter,
                self.frame,
            );
        }
    }

    pub(crate) fn after_submit(&self) {
        #[cfg(feature = "dev")]
        if let Some(diagnostic) = self.diagnostic.borrow().as_ref() {
            diagnostic.after_submit();
        }
        for phase in &self.phases {
            let counters = Rc::clone(&phase.counters);
            phase.readbacks.after_submit(move |words, frame| {
                let sample = CounterSample {
                    frame,
                    words: std::array::from_fn(|index| words.get(index)),
                };
                if counters.get().is_none_or(|old| old.frame <= frame) {
                    counters.set(Some(sample));
                }
            });
        }
    }

    /// Bytes besides the ranges arena, which the full light pass counts.
    pub(crate) fn extra_bytes(&self) -> u64 {
        #[cfg(feature = "dev")]
        let diagnostic_bytes = self
            .diagnostic
            .borrow()
            .as_ref()
            .map_or(0, diagnostic::Diagnostic::bytes);
        #[cfg(not(feature = "dev"))]
        let diagnostic_bytes = 0;
        u64::from(self.scratch.width()) * u64::from(self.scratch.height()) * 4
            + diagnostic_bytes
            + self
                .phases
                .iter()
                .map(|phase| phase.uniform.size() + phase.readbacks.capacity_bytes())
                .sum::<u64>()
    }

    pub(crate) fn stats(&self) -> DepthOccupancyStats {
        let sample = self
            .phases
            .iter()
            .filter_map(|phase| phase.counters.get())
            .max_by_key(|sample| (sample.frame, sample.words[1]));
        let mut stats = DepthOccupancyStats {
            phases: std::array::from_fn(|index| {
                self.phases[index].counters.get().map(|sample| {
                    let [
                        generation,
                        _,
                        occupied_clusters,
                        occupied_bits,
                        original_entries,
                        kept_entries,
                        rejected_entries,
                        empty_clusters,
                    ] = sample.words;
                    engine_core::DepthOccupancyPhaseStats {
                        generation,
                        readback_age: self.frame.saturating_sub(sample.frame),
                        occupied_clusters,
                        occupied_bits,
                        original_entries,
                        kept_entries,
                        rejected_entries,
                        empty_clusters,
                    }
                })
            }),
            enabled: true,
            active: true,
            fallback: sample.is_none().then_some("readback_pending"),
            view: self.context.map_or(0, |(view, _)| view),
            eligible_draws: self.eligible_draws,
            allocated_bytes: self.arena.size() - CLUSTERS * 8 + self.extra_bytes(),
            ..Default::default()
        };
        if let Some(sample) = sample {
            stats.generation = sample.words[0];
            stats.phase = sample.words[1];
            stats.readback_age = self.frame.saturating_sub(sample.frame);
            for phase in self
                .phases
                .iter()
                .filter_map(|phase| phase.counters.get())
                .filter(|phase| phase.frame == sample.frame)
            {
                stats.occupied_clusters = stats.occupied_clusters.saturating_add(phase.words[2]);
                stats.occupied_bits = stats.occupied_bits.saturating_add(phase.words[3]);
                stats.original_entries = stats.original_entries.saturating_add(phase.words[4]);
                stats.kept_entries = stats.kept_entries.saturating_add(phase.words[5]);
                stats.rejected_entries = stats.rejected_entries.saturating_add(phase.words[6]);
                stats.empty_clusters = stats.empty_clusters.saturating_add(phase.words[7]);
            }
        }
        stats
    }
}
