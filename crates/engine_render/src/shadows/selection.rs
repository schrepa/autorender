//! Caster selection (decision 020): one compute pass selects each map's
//! casters from every candidate, visible or culled, into indirect draws per
//! caster group, from the draw arguments' template. On GPU-presented frames
//! it also keeps each slot's stillness and marks the kept map's dirty
//! tiles, read back a few frames later with the caster totals.

use std::{cell::Cell, rc::Rc};

use super::casters::CasterGroups;
use super::pipelines::ShadowPipelines;
use super::{GROUP, KEPT_MAP, MAPS, Presented, SKINNED_GROUP, ShadowFrame, Source};
use crate::bindings::{SHADOW_SELECT, SHADOW_SKINNED};
use crate::browser::FramePoses;
use crate::buffers::{ReadbackRing, entry};
use crate::cached::Cached;
use crate::gpu_data::{BatchData, CandidateData, DRAW_ARGS_BYTES};
use crate::profiling::{GpuProfiler, Span};
use crate::shadow_fit::TILE_WORDS;

// Bindings need at least one element; exceeds every record stride.
const MIN_STORAGE_BYTES: u64 = 256;
// Dirty tiles, then the three caster totals and a pad word, read back
// together.
const READBACK_BYTES: u64 = (TILE_WORDS as u64 + 4) * 4;

#[derive(Clone, Copy, PartialEq, Eq)]
struct SelectKey {
    source: Source,
    buffers: u64,
    presented: u64,
    poses: u64,
    pose_index: usize,
}

/// What a template update changed for the kept map: the levels of detail
/// it draws (it must be redrawn whole), and the dirty-tile buffer (every
/// tile must be redrawn).
pub(super) struct TemplateUpdate {
    pub(super) kept_levels: bool,
    pub(super) dirty_replaced: bool,
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    crate::buffers::storage(device, label, bytes.max(MIN_STORAGE_BYTES), extra)
}

pub(super) struct Selection {
    // Per slot: pose key, stillness count and kept-map tiles.
    stillness: wgpu::Buffer,
    stillness_capacity: usize,
    // Each map's selected caster IDs (a region per group) and indirect
    // arguments, and the arguments' template.
    pub(super) ids: wgpu::Buffer,
    pub(super) id_capacity: usize,
    pub(super) args: wgpu::Buffer,
    args_template: wgpu::Buffer,
    arg_capacity: usize,
    // Dirty tiles and caster totals, then every group's region start.
    dirty: wgpu::Buffer,
    starts_capacity: usize,
    // Stands in for presentation flags where frames have none.
    no_flags: wgpu::Buffer,
    // Casters of CPU-prepared frames and one group record per mesh.
    listed: wgpu::Buffer,
    listed_groups: wgpu::Buffer,
    listed_capacity: usize,
    // The level of detail each map draws each group at (decision 047), as
    // the template holds them, the mesh arena layout it was written for,
    // and whether it must be written again.
    template_levels: Vec<u8>,
    mesh_layout: u64,
    template_stale: bool,
    // Whether the device draws indirectly from nonzero first instances, and
    // whether the template's first instances name the draws' ID regions.
    first_instance: bool,
    pub(super) regions_in_args: bool,
    // Advances when this part replaces a buffer.
    generation: u64,
    // Selection bindings, and skinned selection's (decision 036), by the
    // inputs they were built for.
    // Per pose buffer parity (temporal frames alternate them), and the
    // parity this frame binds.
    select_groups: [Cached<SelectKey, wgpu::BindGroup>; 2],
    select_parity: usize,
    skinned_group: Cached<(u64, u64), wgpu::BindGroup>,
    readbacks: ReadbackRing<()>,
    // Dirty tiles read back and not yet taken, and the latest caster totals
    // read back: near, moving and kept maps.
    read_back: Rc<Cell<[u32; TILE_WORDS]>>,
    casters: Rc<Cell<Option<[u32; 3]>>>,
}

impl Selection {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let empty = |label: &str, extra| storage(device, label, 0, extra);
        Self {
            stillness: empty("engine/shadow-stillness", wgpu::BufferUsages::empty()),
            stillness_capacity: 0,
            ids: empty("engine/shadow-ids", wgpu::BufferUsages::VERTEX),
            id_capacity: 0,
            args: empty("engine/shadow-args", wgpu::BufferUsages::INDIRECT),
            args_template: empty("engine/shadow-args-template", wgpu::BufferUsages::COPY_SRC),
            arg_capacity: 0,
            dirty: storage(
                device,
                "engine/shadow-dirty",
                READBACK_BYTES,
                wgpu::BufferUsages::COPY_SRC,
            ),
            starts_capacity: 0,
            no_flags: empty("engine/shadow-no-flags", wgpu::BufferUsages::empty()),
            listed: empty("engine/shadow-casters", wgpu::BufferUsages::empty()),
            listed_groups: empty("engine/shadow-caster-groups", wgpu::BufferUsages::empty()),
            listed_capacity: 0,
            template_levels: Vec::new(),
            mesh_layout: 0,
            template_stale: false,
            first_instance: device
                .features()
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
            regions_in_args: false,
            generation: 0,
            select_groups: Default::default(),
            select_parity: 0,
            skinned_group: Cached::default(),
            readbacks: ReadbackRing::new("engine/shadow-dirty-readback"),
            read_back: Rc::new(Cell::new([0; TILE_WORDS])),
            casters: Rc::new(Cell::new(None)),
        }
    }

    /// Advances when a selection buffer is replaced.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) fn capacity_bytes(&self) -> u64 {
        [
            &self.stillness,
            &self.ids,
            &self.args,
            &self.args_template,
            &self.listed,
            &self.listed_groups,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }

    /// The latest caster totals read back: near cascades together, the
    /// moving map and the kept map.
    pub(super) fn caster_counts(&self) -> Option<[u32; 3]> {
        self.casters.get()
    }

    /// Takes the dirty tiles read back since the last call.
    pub(super) fn take_dirty_tiles(&self) -> [u32; TILE_WORDS] {
        self.read_back.replace([0; TILE_WORDS])
    }

    /// The template must be written again: the caster groups changed.
    pub(super) fn groups_changed(&mut self) {
        self.template_stale = true;
    }

    /// Keeps the stillness of `slots` slots; returns whether it starts
    /// over, in a new buffer.
    pub(super) fn ensure_stillness(&mut self, device: &wgpu::Device, slots: usize) -> bool {
        if self.stillness_capacity >= slots {
            return false;
        }
        self.stillness_capacity = slots.next_power_of_two();
        self.stillness = storage(
            device,
            "engine/shadow-stillness",
            self.stillness_capacity as u64 * 8,
            wgpu::BufferUsages::empty(),
        );
        self.generation += 1;
        true
    }

    /// Sizes the selection's outputs: `casters` IDs per map and the
    /// arguments of `groups` groups.
    pub(super) fn ensure_outputs(&mut self, device: &wgpu::Device, groups: usize, casters: u32) {
        let ids = (casters as usize).max(1);
        if ids > self.id_capacity {
            self.id_capacity = ids.next_power_of_two();
            // Draw arguments name regions of the new capacity.
            self.template_stale = true;
            self.ids = storage(
                device,
                "engine/shadow-ids",
                (MAPS * self.id_capacity * 4) as u64,
                wgpu::BufferUsages::VERTEX,
            );
            self.generation += 1;
        }
        if groups > self.arg_capacity {
            self.arg_capacity = groups.next_power_of_two();
            let bytes = (MAPS * self.arg_capacity) as u64 * DRAW_ARGS_BYTES;
            self.args = storage(
                device,
                "engine/shadow-args",
                bytes,
                wgpu::BufferUsages::INDIRECT,
            );
            self.args_template = storage(
                device,
                "engine/shadow-args-template",
                bytes,
                wgpu::BufferUsages::COPY_SRC,
            );
            self.generation += 1;
            self.template_stale = true;
        }
    }

    /// Writes the draw arguments' template when the groups, `levels` (each
    /// map's level of detail per group, decision 047), the storage or the
    /// mesh arena `mesh_layout` changed, from each group's `ranges` (its
    /// index count, first index and base vertex at each level), with the
    /// group starts and CPU-prepared frames' casters.
    pub(super) fn update_template(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        casters: &CasterGroups,
        levels: Vec<u8>,
        ranges: impl Fn(usize, u8) -> (u32, u32, i32),
        mesh_layout: u64,
    ) -> TemplateUpdate {
        let mut update = TemplateUpdate {
            kept_levels: false,
            dirty_replaced: false,
        };
        let groups = casters.groups();
        self.template_stale |= self.mesh_layout != mesh_layout;
        self.mesh_layout = mesh_layout;
        if levels != self.template_levels {
            // Casters drawn at other levels than the kept map holds redraw
            // it whole, as a moved sun does (decision 047): its levels change
            // only with its texel and when levels of detail turn on or off.
            let kept = |levels: &[u8]| {
                levels
                    .get(KEPT_MAP * groups.len()..(KEPT_MAP + 1) * groups.len())
                    .map(<[u8]>::to_vec)
            };
            update.kept_levels = kept(&levels) != kept(&self.template_levels);
            self.template_stale = true;
        }
        if !self.template_stale {
            return update;
        }
        self.template_levels = levels;
        self.template_stale = false;
        if groups.len() > self.starts_capacity {
            // Counters first, then every group's region start; a new
            // buffer drops unread counters, which only delays tiles.
            self.starts_capacity = groups.len().next_power_of_two();
            self.dirty = storage(
                device,
                "engine/shadow-dirty",
                READBACK_BYTES + (self.starts_capacity * 4) as u64,
                wgpu::BufferUsages::COPY_SRC,
            );
            update.dirty_replaced = true;
            self.generation += 1;
        }
        let starts: Vec<u32> = groups.iter().map(|group| group.start).collect();
        if !starts.is_empty() {
            queue.write_buffer(&self.dirty, READBACK_BYTES, bytemuck::cast_slice(&starts));
        }
        // Each map's draws name their ID regions as first instances
        // where the device draws indirectly from them.
        self.regions_in_args =
            self.first_instance && u32::try_from(MAPS * self.id_capacity).is_ok();
        let mut template: Vec<u32> = Vec::with_capacity(MAPS * groups.len() * 5);
        for map in 0..MAPS {
            for (index, group) in groups.iter().enumerate() {
                let (count, first_index, base_vertex) =
                    ranges(index, self.template_levels[map * groups.len() + index]);
                let first_instance = if self.regions_in_args {
                    (map * self.id_capacity) as u32 + group.start
                } else {
                    0
                };
                template.extend([count, 0, first_index, base_vertex as u32, first_instance]);
            }
        }
        if !template.is_empty() {
            queue.write_buffer(&self.args_template, 0, bytemuck::cast_slice(&template));
        }
        if casters.source() == Source::Listed {
            let listed = casters.listed.len().max(1);
            let group_bytes = (groups.len().max(1) * size_of::<BatchData>()) as u64;
            if listed > self.listed_capacity || group_bytes > self.listed_groups.size() {
                self.listed_capacity = listed.next_power_of_two();
                self.listed = storage(
                    device,
                    "engine/shadow-casters",
                    (self.listed_capacity * size_of::<CandidateData>()) as u64,
                    wgpu::BufferUsages::empty(),
                );
                self.listed_groups = storage(
                    device,
                    "engine/shadow-caster-groups",
                    group_bytes.next_power_of_two(),
                    wgpu::BufferUsages::empty(),
                );
                self.generation += 1;
            }
            if !casters.listed.is_empty() {
                queue.write_buffer(&self.listed, 0, bytemuck::cast_slice(&casters.listed));
            }
            queue.write_buffer(
                &self.listed_groups,
                0,
                bytemuck::cast_slice(&casters.records),
            );
        }
        update
    }

    /// Binds the selection of a frame of `source` over `poses` (pose buffer
    /// `pose_index`), with GPU presentation's buffers (`presented`) and
    /// `skinned` skinned slots on GPU-presented frames.
    pub(super) fn bind(
        &mut self,
        device: &wgpu::Device,
        pipelines: &ShadowPipelines,
        source: Source,
        presented: Option<&Presented<'_>>,
        poses: &FramePoses<'_>,
        skinned: u32,
    ) {
        let key = SelectKey {
            source,
            buffers: self.generation,
            presented: presented.map_or(0, |presented| presented.generation),
            poses: poses.generation,
            pose_index: poses.parity,
        };
        self.select_parity = poses.parity;
        if self.select_groups[poses.parity].stale(&key) {
            let pose_buffer = poses.buffers[poses.parity].expect("the frame's pose buffer");
            let (candidate_buffer, batch_buffer, flag_buffer) = match presented {
                Some(presented) if source == Source::Presented => {
                    (presented.candidates, presented.batches, presented.flags)
                }
                _ => (&self.listed, &self.listed_groups, &self.no_flags),
            };
            let group = crate::pipelines::group(
                device,
                "engine/shadow-select-group",
                &SHADOW_SELECT,
                &pipelines.select_layout,
                &[
                    entry("cull", &pipelines.cull),
                    entry("candidates", candidate_buffer),
                    entry("batches", batch_buffer),
                    entry("poses", pose_buffer),
                    entry("flags", flag_buffer),
                    entry("stillness", &self.stillness),
                    entry("ids", &self.ids),
                    entry("args", &self.args),
                    entry("counters", &self.dirty),
                ],
            );
            self.select_groups[poses.parity].set(key, group);
        }
        if let Some(presented) = presented.filter(|_| skinned != 0) {
            let key = (self.generation, presented.generation);
            if self.skinned_group.stale(&key) {
                let group = crate::pipelines::group(
                    device,
                    "engine/shadow-skinned-group",
                    &SHADOW_SKINNED,
                    &pipelines.skinned_layout,
                    &[
                        entry("cull", &pipelines.cull),
                        entry("batches", presented.batches),
                        entry("flags", presented.flags),
                        entry("ids", &self.ids),
                        entry("args", &self.args),
                        entry("counters", &self.dirty),
                        entry("skinned", presented.skinned),
                        entry("skin_bounds", presented.skin_bounds),
                    ],
                );
                self.skinned_group.set(key, group);
            }
        }
    }

    /// Records caster selection of `frame` with `groups` caster groups,
    /// timed by `profiler` (`shadow_select`).
    pub(super) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &ShadowPipelines,
        frame: &ShadowFrame,
        groups: usize,
        profiler: &GpuProfiler,
    ) {
        // Instance counts start at zero every frame.
        encoder.copy_buffer_to_buffer(
            &self.args_template,
            0,
            &self.args,
            0,
            MAPS as u64 * groups as u64 * DRAW_ARGS_BYTES,
        );
        if let Some(group) = self.select_groups[self.select_parity].get()
            && frame.candidates != 0
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("engine/shadow-select"),
                timestamp_writes: profiler.span("shadow_select").map(Span::compute),
            });
            pass.set_pipeline(&pipelines.select);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(frame.candidates.div_ceil(GROUP), 1, 1);
            // Skinned casters by their palette bounds (decision 036).
            if frame.skinned != 0
                && let Some(skinned) = self.skinned_group.get()
            {
                pass.set_pipeline(&pipelines.select_skinned);
                pass.set_bind_group(0, skinned, &[]);
                pass.dispatch_workgroups(frame.skinned.div_ceil(SKINNED_GROUP), 1, 1);
                pass.set_bind_group(0, group, &[]);
            }
            pass.set_pipeline(&pipelines.count);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // Unread tiles stay until a readback copies them.
        if self
            .readbacks
            .record(device, encoder, &self.dirty, READBACK_BYTES, ())
        {
            encoder.clear_buffer(&self.dirty, 0, Some(TILE_WORDS as u64 * 4));
        }
    }

    /// Requests maps of dirty tiles recorded in the frame just submitted.
    pub(super) fn after_submit(&mut self) {
        let (read_back, casters) = (Rc::clone(&self.read_back), Rc::clone(&self.casters));
        self.readbacks.after_submit(move |words, ()| {
            let mut tiles = read_back.get();
            for (index, tile) in tiles.iter_mut().enumerate() {
                *tile |= words.get(index);
            }
            read_back.set(tiles);
            casters.set(Some([
                words.get(TILE_WORDS),
                words.get(TILE_WORDS + 1),
                words.get(TILE_WORDS + 2),
            ]));
        });
    }
}
