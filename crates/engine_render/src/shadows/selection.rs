//! Caster selection: one compute pass selects each map's casters from every
//! candidate, visible or culled, into indirect draws per caster group, from
//! the draw arguments' template. On GPU-presented frames it also keeps
//! each slot's stillness, marks the drawn tiles whose still casters changed
//! and those moving casters cross, read back a few frames later with the
//! caster totals.

#![expect(clippy::expect_used, reason = "the frame binds a pose buffer")]

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use super::casters::CasterGroups;
use super::pipelines::ShadowPipelines;
use super::{GROUP, MAPS, MOVING_MAP, Presented, SKINNED_GROUP, ShadowFrame, Source};
use crate::bindings::{SHADOW_SELECT, SHADOW_SKINNED};
use crate::buffers::{ReadbackRing, entry};
use crate::cached::Cached;
use crate::gpu_data::{BatchData, CandidateData, DRAW_ARGS_BYTES};
use crate::gpu_layouts::constants::{MOVING_PAIRS, SHADOW_TOTALS};
use crate::profiling::{GpuProfiler, Span};
use crate::renderer::FramePoses;

// Bindings need at least one element; exceeds every record stride.
const MIN_STORAGE_BYTES: u64 = 256;
// Bytes of a slot's stillness: its pose key, its still-frame count, and
// its footprint while kept.
const STILLNESS_BYTES: u64 = 16;
/// Capacities of the instance IDs of every map: each region's and the sky
/// map's one, the moving map's MOVING_PAIRS.
const ID_CAPACITIES: usize = MAPS - 1 + MOVING_PAIRS as usize;

/// Where map `map`'s instance IDs start among the maps' IDs, `capacity` a
/// map; the moving map holds MOVING_PAIRS times as many as the others.
pub(super) fn map_start(map: usize, capacity: usize) -> usize {
    if map <= MOVING_MAP {
        map * capacity
    } else {
        (map + MOVING_PAIRS as usize - 1) * capacity
    }
}

/// How many IDs map `map` holds for each of a group's casters.
pub(super) fn map_factor(map: usize) -> u32 {
    if map == MOVING_MAP { MOVING_PAIRS } else { 1 }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SelectKey {
    source: Source,
    buffers: u64,
    presented: u64,
    poses: u64,
    pose_index: usize,
    tiles: u64,
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    extra: wgpu::BufferUsages,
) -> wgpu::Buffer {
    crate::buffers::storage(device, label, bytes.max(MIN_STORAGE_BYTES), extra)
}

/// What a frame read back of the selection: the masks of every level's
/// tiles whose still casters changed and that moving casters crossed,
/// merged since last taken, and the clipmap epoch they were recorded in.
#[derive(Default)]
struct MasksRead {
    epoch: u64,
    dirty: Vec<u32>,
    moving: Vec<u32>,
}

pub(super) struct Selection {
    // Per slot: pose key, stillness count and footprint.
    stillness: wgpu::Buffer,
    stillness_capacity: usize,
    // Each map's selected caster IDs (a region per group) and indirect
    // arguments, and the arguments' template.
    pub(super) ids: wgpu::Buffer,
    pub(super) id_capacity: usize,
    pub(super) args: wgpu::Buffer,
    args_template: wgpu::Buffer,
    arg_capacity: usize,
    // The totals and masks read back, then every group's region start, and
    // the words of a mask and the groups it is sized for.
    counters: wgpu::Buffer,
    mask_words: usize,
    starts_capacity: usize,
    // Stands in for presentation flags where frames have none.
    no_flags: wgpu::Buffer,
    // Casters of CPU-prepared frames and one group record per mesh.
    listed: wgpu::Buffer,
    listed_groups: wgpu::Buffer,
    listed_capacity: usize,
    // The level of detail each map draws each group at, as
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
    // Selection bindings, and skinned selection's, per pose buffer parity
    // (temporal frames alternate them), and the parity this frame binds.
    select_groups: [Cached<SelectKey, wgpu::BindGroup>; 2],
    select_parity: usize,
    skinned_group: Cached<(u64, u64, u64), wgpu::BindGroup>,
    readbacks: ReadbackRing<u64>,
    // The masks read back and not yet taken, and the latest caster totals
    // read back: the regions', the moving map's and those it left out.
    read_back: Rc<RefCell<MasksRead>>,
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
            counters: empty("engine/shadow-counters", wgpu::BufferUsages::COPY_SRC),
            mask_words: 0,
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
            readbacks: ReadbackRing::new("engine/shadow-counters-readback"),
            read_back: Rc::new(RefCell::new(MasksRead::default())),
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
            &self.counters,
            &self.listed,
            &self.listed_groups,
        ]
        .iter()
        .map(|buffer| buffer.size())
        .sum()
    }

    /// Words read back: the totals, then the masks of the dirty and the
    /// moving tiles.
    fn readback_words(&self) -> usize {
        SHADOW_TOTALS + 2 * self.mask_words
    }

    /// The latest caster totals read back: the casters the regions drew,
    /// the moving map's draws and those it left out.
    pub(super) fn caster_counts(&self) -> Option<[u32; 3]> {
        self.casters.get()
    }

    /// Takes the masks read back since the last call, recorded in epoch
    /// `epoch` of the clipmap (others name cells that no longer hold what
    /// they did): the tiles whose still casters changed, then those moving
    /// casters crossed.
    pub(super) fn take_masks(&self, epoch: u64) -> (Vec<u32>, Vec<u32>) {
        let mut read = self.read_back.borrow_mut();
        let (dirty, moving) = (
            std::mem::take(&mut read.dirty),
            std::mem::take(&mut read.moving),
        );
        if read.epoch == epoch {
            (dirty, moving)
        } else {
            (Vec::new(), Vec::new())
        }
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
            self.stillness_capacity as u64 * STILLNESS_BYTES,
            wgpu::BufferUsages::empty(),
        );
        self.generation += 1;
        true
    }

    /// Sizes the selection's outputs: `casters` IDs per map and the
    /// arguments of `groups` groups; and the counters for masks of
    /// `mask_words` words; returns whether the counters were made anew,
    /// dropping what they had not read back.
    pub(super) fn ensure_outputs(
        &mut self,
        device: &wgpu::Device,
        (groups, casters): (usize, u32),
        mask_words: usize,
    ) -> bool {
        let ids = (casters as usize).max(1);
        if ids > self.id_capacity {
            self.id_capacity = ids.next_power_of_two();
            // Draw arguments name regions of the new capacity.
            self.template_stale = true;
            self.ids = storage(
                device,
                "engine/shadow-ids",
                (ID_CAPACITIES * self.id_capacity * 4) as u64,
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
        if mask_words == self.mask_words && groups <= self.starts_capacity {
            return false;
        }
        // The totals and masks first, then every group's region start.
        self.mask_words = mask_words;
        self.starts_capacity = self.starts_capacity.max(groups.next_power_of_two());
        let words = self.readback_words() + self.starts_capacity;
        self.counters = storage(
            device,
            "engine/shadow-counters",
            (words * 4) as u64,
            wgpu::BufferUsages::COPY_SRC,
        );
        self.readbacks.clear();
        self.generation += 1;
        self.template_stale = true;
        true
    }

    /// Writes the draw arguments' template when the groups, `levels` (each
    /// map's level of detail per group), the storage or the
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
    ) {
        let groups = casters.groups();
        self.template_stale |= self.mesh_layout != mesh_layout || levels != self.template_levels;
        self.mesh_layout = mesh_layout;
        if !self.template_stale {
            return;
        }
        self.template_levels = levels;
        self.template_stale = false;
        let starts: Vec<u32> = groups.iter().map(|group| group.start).collect();
        if !starts.is_empty() {
            queue.write_buffer(
                &self.counters,
                (self.readback_words() * 4) as u64,
                bytemuck::cast_slice(&starts),
            );
        }
        // Each map's draws name their ID regions as first instances
        // where the device draws indirectly from them.
        self.regions_in_args =
            self.first_instance && u32::try_from(ID_CAPACITIES * self.id_capacity).is_ok();
        let mut template: Vec<u32> = Vec::with_capacity(MAPS * groups.len() * 5);
        for map in 0..MAPS {
            let start = map_start(map, self.id_capacity) as u32;
            for (index, group) in groups.iter().enumerate() {
                let (count, first_index, base_vertex) =
                    ranges(index, self.template_levels[map * groups.len() + index]);
                let first_instance = if self.regions_in_args {
                    start + map_factor(map) * group.start
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
    }

    /// Binds the selection of a frame of `source` over `poses` (pose buffer
    /// `pose_index`), with GPU presentation's buffers (`presented`) and
    /// `skinned` skinned slots on GPU-presented frames, and the table of
    /// the clipmap's slots `tiles` of generation `tiles_key`.
    #[expect(clippy::too_many_arguments, reason = "a frame's disjoint inputs")]
    pub(super) fn bind(
        &mut self,
        device: &wgpu::Device,
        pipelines: &ShadowPipelines,
        source: Source,
        presented: Option<&Presented<'_>>,
        poses: &FramePoses<'_>,
        skinned: u32,
        (tiles, tiles_key): (&wgpu::TextureView, u64),
    ) {
        let key = SelectKey {
            source,
            buffers: self.generation,
            presented: presented.map_or(0, |presented| presented.generation),
            poses: poses.generation(),
            pose_index: poses.parity,
            tiles: tiles_key,
        };
        self.select_parity = poses.parity;
        if self.select_groups[poses.parity].stale(&key) {
            let pose_buffer = poses.buffer(poses.parity).expect("the frame's pose buffer");
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
                    entry("counters", &self.counters),
                    ("tile_slots", crate::pipelines::view(tiles)),
                ],
            );
            self.select_groups[poses.parity].set(key, group);
        }
        if let Some(presented) = presented.filter(|_| skinned != 0) {
            let key = (self.generation, presented.generation, tiles_key);
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
                        entry("counters", &self.counters),
                        entry("skinned", presented.skinned),
                        entry("skin_bounds", presented.skin_bounds),
                        ("tile_slots", crate::pipelines::view(tiles)),
                    ],
                );
                self.skinned_group.set(key, group);
            }
        }
    }

    /// Records caster selection of `frame` with `groups` caster groups,
    /// timed by `profiler` (`shadow_select`), and the read-back of what it
    /// counted and marked, tagged with the clipmap's `epoch`.
    pub(super) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &ShadowPipelines,
        (frame, groups): (&ShadowFrame, usize),
        epoch: u64,
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
            // Skinned casters by their palette bounds.
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
        // Unread masks stay until a read-back copies them.
        let bytes = (self.readback_words() * 4) as u64;
        if self
            .readbacks
            .record(device, encoder, &self.counters, bytes, epoch)
        {
            encoder.clear_buffer(&self.counters, 0, Some(bytes));
        }
    }

    /// Requests maps of what the frame just submitted read back.
    pub(super) fn after_submit(&mut self) {
        let (read_back, casters) = (Rc::clone(&self.read_back), Rc::clone(&self.casters));
        let mask_words = self.mask_words;
        self.readbacks.after_submit(move |words, epoch| {
            let mut read = read_back.borrow_mut();
            if read.epoch != epoch {
                *read = MasksRead {
                    epoch,
                    dirty: vec![0; mask_words],
                    moving: vec![0; mask_words],
                };
            }
            read.dirty.resize(mask_words, 0);
            read.moving.resize(mask_words, 0);
            for index in 0..mask_words {
                read.dirty[index] |= words.get(SHADOW_TOTALS + index);
                read.moving[index] |= words.get(SHADOW_TOTALS + mask_words + index);
            }
            casters.set(Some([words.get(0), words.get(1), words.get(2)]));
        });
    }
}
