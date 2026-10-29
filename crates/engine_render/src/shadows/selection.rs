#![expect(clippy::expect_used, reason = "the frame binds a pose buffer")]

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use super::casters::CasterGroups;
use super::pipelines::ShadowPipelines;
use super::{GROUP, MAPS, MOVING_MAP, Presented, ShadowFrame, Source};
use crate::bindings::SHADOW_SELECT;
use crate::buffers::{ReadbackRing, entry};
use crate::cached::Cached;
use crate::gpu_data::{BatchData, CandidateData, DRAW_ARGS_BYTES};
use crate::gpu_layouts::constants::{MOVING_PAIRS, SHADOW_TOTALS};
use crate::profiling::{GpuProfiler, Span};
use crate::renderer::FramePoses;

const MIN_STORAGE_BYTES: u64 = 256;
const STILLNESS_BYTES: u64 = 16;
const ID_CAPACITIES: usize = MAPS - 1 + MOVING_PAIRS as usize;

pub(super) fn map_start(map: usize, capacity: usize) -> usize {
    if map <= MOVING_MAP {
        map * capacity
    } else {
        (map + MOVING_PAIRS as usize - 1) * capacity
    }
}

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

#[derive(Default)]
struct MasksRead {
    epoch: u64,
    dirty: Vec<u32>,
    moving: Vec<u32>,
}

pub(super) struct Selection {
    stillness: wgpu::Buffer,
    stillness_capacity: usize,
    pub(super) ids: wgpu::Buffer,
    pub(super) id_capacity: usize,
    pub(super) args: wgpu::Buffer,
    args_template: wgpu::Buffer,
    arg_capacity: usize,
    counters: wgpu::Buffer,
    mask_words: usize,
    starts_capacity: usize,
    no_flags: wgpu::Buffer,
    listed: wgpu::Buffer,
    listed_groups: wgpu::Buffer,
    listed_capacity: usize,
    template_levels: Vec<u8>,
    mesh_layout: u64,
    template_stale: bool,
    first_instance: bool,
    pub(super) regions_in_args: bool,
    generation: u64,
    select_groups: [Cached<SelectKey, wgpu::BindGroup>; 2],
    select_parity: usize,
    readbacks: ReadbackRing<u64>,
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
            readbacks: ReadbackRing::new("engine/shadow-counters-readback"),
            read_back: Rc::new(RefCell::new(MasksRead::default())),
            casters: Rc::new(Cell::new(None)),
        }
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

    fn readback_words(&self) -> usize {
        SHADOW_TOTALS + 2 * self.mask_words
    }

    pub(super) fn caster_counts(&self) -> Option<[u32; 3]> {
        self.casters.get()
    }

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

    pub(super) fn groups_changed(&mut self) {
        self.template_stale = true;
    }

    pub(super) fn ensure_stillness(&mut self, device: &wgpu::Device, slots: usize) -> bool {
        if self.stillness_capacity >= slots {
            return false;
        }
        self.stillness_capacity = slots.next_power_of_two();
        crate::memory::release_buffer(&self.stillness);
        self.stillness = storage(
            device,
            "engine/shadow-stillness",
            self.stillness_capacity as u64 * STILLNESS_BYTES,
            wgpu::BufferUsages::empty(),
        );
        self.generation += 1;
        true
    }

    pub(super) fn ensure_outputs(
        &mut self,
        device: &wgpu::Device,
        (groups, casters): (usize, u32),
        mask_words: usize,
    ) -> bool {
        let ids = (casters as usize).max(1);
        if ids > self.id_capacity {
            self.id_capacity = ids.next_power_of_two();
            self.template_stale = true;
            crate::memory::release_buffer(&self.ids);
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
            crate::memory::release_buffer(&self.args);
            self.args = storage(
                device,
                "engine/shadow-args",
                bytes,
                wgpu::BufferUsages::INDIRECT,
            );
            crate::memory::release_buffer(&self.args_template);
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
        self.mask_words = mask_words;
        self.starts_capacity = self.starts_capacity.max(groups.next_power_of_two());
        let words = self.readback_words() + self.starts_capacity;
        crate::memory::release_buffer(&self.counters);
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

    pub(super) fn update_template(
        &mut self,
        device: &wgpu::Device,
        queue: &crate::gpu_calls::Queue,
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
                crate::memory::release_buffer(&self.listed);
                self.listed = storage(
                    device,
                    "engine/shadow-casters",
                    (self.listed_capacity * size_of::<CandidateData>()) as u64,
                    wgpu::BufferUsages::empty(),
                );
                crate::memory::release_buffer(&self.listed_groups);
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

    pub(super) fn bind(
        &mut self,
        device: &wgpu::Device,
        pipelines: &ShadowPipelines,
        source: Source,
        presented: Option<&Presented<'_>>,
        poses: &FramePoses<'_>,
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
    }

    pub(super) fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &ShadowPipelines,
        (frame, groups): (&ShadowFrame, usize),
        epoch: u64,
        profiler: &GpuProfiler,
    ) {
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
            pass.set_pipeline(&pipelines.count);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let bytes = (self.readback_words() * 4) as u64;
        if self
            .readbacks
            .record(device, encoder, &self.counters, bytes, epoch)
        {
            encoder.clear_buffer(&self.counters, 0, Some(bytes));
        }
    }

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
