use super::*;
use crate::hierarchy::Link;
use crate::retained_render::{PoseRows, pose_rows};

pub type NodePose = [f32; 10];

pub type NodeInfo = [u32; 2];

pub const NO_PARENT: u32 = u32::MAX;
pub const NODE_VALID: u32 = 1;
pub const NODE_BLEND: u32 = 2;
pub const NODE_ACTIVE: u32 = 4;
pub const NODE_PARENT_ACTIVE: u32 = 8;

const SNAP: u8 = 1;
const ACTIVE: u8 = 2;
const AUTHORITY: u8 = 4;
const NEVER: u32 = u32::MAX;

static NEXT_INPUT_STREAM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

#[derive(Clone, Copy)]
pub struct GpuPresentation<'a> {
    pub alpha: f32,
    pub identity: u64,
    pub sequence: u64,
    pub shift: bool,
    pub shift_sequence: u64,
    pub current: &'a [NodePose],
    pub current_stamps: &'a [u64],
    pub previous: &'a [NodePose],
    pub previous_stamps: &'a [u64],
    pub info: &'a [NodeInfo],
    pub info_stamps: &'a [u64],
    pub authority: &'a [PoseRows],
    pub authority_stamps: &'a [u64],
    pub parents: &'a [u32],
    pub levels: &'a [u32],
    pub stamped: [usize; 4],
    pub stamped_nodes: [Option<&'a [u32]>; 4],
    pub current_version: (u64, u64),
    pub hold: bool,
}

#[derive(Clone, Copy)]
pub struct GpuCurrent<'a> {
    pub poses: &'a [NodePose],
    pub version: (u64, u64),
    pub stamped: usize,
}

#[derive(Clone)]
pub(crate) struct GpuInputs {
    identity: u64,
    sequence: u64,
    alpha: f32,
    gathered: Option<(u64, u64, u64)>,
    current_stamps: Vec<u64>,
    previous: Vec<NodePose>,
    previous_stamps: Vec<u64>,
    epochs: Vec<[u32; 2]>,
    info: Vec<NodeInfo>,
    info_stamps: Vec<u64>,
    authority: Vec<PoseRows>,
    authority_stamps: Vec<u64>,
    authority_revisions: Vec<u64>,
    parents: Vec<u32>,
    levels: Vec<u32>,
    state: Vec<u8>,
    depth_counts: Vec<u32>,
    parent_scratch: Vec<(u8, u32)>,
    last_epoch: Option<u64>,
    last_revision: u64,
    tick_begin_revision: Option<u64>,
    shift: bool,
    shift_sequence: u64,
    stamped: [usize; 4],
    stamped_lists: [Vec<u32>; 4],
    current_version: (u64, u64),
    hold: bool,
    unpresented: Option<u64>,
    active: Vec<u32>,
    next_active: Vec<u32>,
    active_listed: bool,
    queued: Vec<u32>,
    queue_generation: u32,
    buckets: Vec<Vec<u32>>,
    visited: usize,
}

impl Default for GpuInputs {
    fn default() -> Self {
        Self {
            identity: 0,
            sequence: 0,
            alpha: 1.0,
            gathered: None,
            current_stamps: Vec::new(),
            previous: Vec::new(),
            previous_stamps: Vec::new(),
            epochs: Vec::new(),
            info: Vec::new(),
            info_stamps: Vec::new(),
            authority: Vec::new(),
            authority_stamps: Vec::new(),
            authority_revisions: Vec::new(),
            parents: Vec::new(),
            levels: Vec::new(),
            state: Vec::new(),
            depth_counts: Vec::new(),
            parent_scratch: Vec::new(),
            last_epoch: None,
            last_revision: 0,
            tick_begin_revision: None,
            shift: false,
            shift_sequence: 0,
            stamped: [0; 4],
            stamped_lists: Default::default(),
            current_version: (0, 0),
            hold: false,
            unpresented: None,
            active: Vec::new(),
            next_active: Vec::new(),
            active_listed: false,
            queued: Vec::new(),
            queue_generation: 0,
            buckets: Vec::new(),
            visited: 0,
        }
    }
}

#[inline]
fn forget_published(epochs: &mut [u32; 2]) {
    if *epochs != [NEVER; 2] {
        *epochs = [NEVER; 2];
    }
}

#[inline]
fn bit<T: Default>(on: bool, value: T) -> T {
    if on { value } else { T::default() }
}

impl GpuInputs {
    pub(crate) fn view<'a>(&'a self, hierarchy: &'a HierarchyScratch) -> GpuPresentation<'a> {
        GpuPresentation {
            alpha: self.alpha,
            identity: self.identity,
            sequence: self.sequence,
            shift: self.shift,
            shift_sequence: self.shift_sequence,
            current: hierarchy.locals(),
            current_stamps: &self.current_stamps,
            previous: &self.previous,
            previous_stamps: &self.previous_stamps,
            info: &self.info,
            info_stamps: &self.info_stamps,
            authority: &self.authority,
            authority_stamps: &self.authority_stamps,
            parents: &self.parents,
            levels: &self.levels,
            stamped: self.stamped,
            stamped_nodes: std::array::from_fn(|stream| {
                let list = &self.stamped_lists[stream];
                (list.len() == self.stamped[stream]).then_some(list.as_slice())
            }),
            current_version: self.current_version,
            hold: self.hold,
        }
    }

    pub(crate) fn capacity_bytes(&self) -> usize {
        self.previous.capacity() * size_of::<NodePose>()
            + self.epochs.capacity() * size_of::<[u32; 2]>()
            + self.info.capacity() * size_of::<NodeInfo>()
            + self.authority.capacity() * size_of::<PoseRows>()
            + (self.current_stamps.capacity()
                + self.previous_stamps.capacity()
                + self.info_stamps.capacity()
                + self.authority_stamps.capacity()
                + self.authority_revisions.capacity())
                * size_of::<u64>()
            + (self.parents.capacity()
                + self.levels.capacity()
                + self.depth_counts.capacity()
                + self.active.capacity()
                + self.next_active.capacity()
                + self.queued.capacity()
                + self
                    .stamped_lists
                    .iter()
                    .map(|list| list.capacity())
                    .sum::<usize>()
                + self
                    .buckets
                    .iter()
                    .map(|bucket| bucket.capacity())
                    .sum::<usize>())
                * size_of::<u32>()
            + self.state.capacity()
            + self.parent_scratch.capacity() * size_of::<(u8, u32)>()
    }

    pub(super) fn begin_tick(&mut self, hierarchy_revision: Option<u64>) {
        self.tick_begin_revision = hierarchy_revision;
    }

    pub(super) fn set_alpha(&mut self, alpha: f32, hold: bool) {
        self.alpha = alpha;
        self.hold = hold;
        if !hold {
            self.presented();
        }
    }

    pub(super) fn presented(&mut self) {
        self.unpresented = None;
    }

    pub(super) fn gathered_ahead(&mut self, prior: u64) {
        self.unpresented = Some(prior);
    }

    pub(super) fn sequence(&self) -> u64 {
        self.sequence
    }

    pub(super) fn shift_sequence(&self) -> u64 {
        self.shift_sequence
    }

    pub(super) fn alpha(&self) -> f32 {
        self.alpha
    }

    fn reindex(&mut self, nodes: usize) {
        self.identity = NEXT_INPUT_STREAM.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        fn reset<T: Copy>(values: &mut Vec<T>, nodes: usize, value: T) {
            values.clear();
            values.reserve_exact(nodes);
            values.resize(nodes, value);
        }
        reset(&mut self.current_stamps, nodes, 0);
        reset(&mut self.previous_stamps, nodes, 0);
        reset(&mut self.info_stamps, nodes, 0);
        reset(&mut self.authority_stamps, nodes, 0);
        reset(&mut self.authority_revisions, nodes, u64::MAX);
        reset(&mut self.previous, nodes, [0.0; 10]);
        reset(&mut self.epochs, nodes, [NEVER; 2]);
        reset(&mut self.info, nodes, [u32::MAX; 2]);
        reset(&mut self.authority, nodes, [[0.0; 4]; 3]);
        reset(&mut self.state, nodes, 0);
        reset(&mut self.queued, nodes, 0);
        self.queue_generation = 0;
        self.buckets.resize_with(MAX_HIERARCHY_DEPTH + 1, Vec::new);
        self.active.clear();
        self.active_listed = false;
        self.last_epoch = None;
    }
}

impl InterpolationState {
    pub(crate) fn gathers_gpu(&self) -> bool {
        self.gpu.gathered.is_some()
    }

    pub(crate) fn gpu_current<'a>(
        &self,
        hierarchy: &'a HierarchyScratch,
    ) -> Option<GpuCurrent<'a>> {
        if !self.gathers_gpu() {
            return None;
        }
        let version = hierarchy.stamp();
        let stamped = if self.gpu.current_version == version {
            self.gpu.stamped[0]
        } else {
            hierarchy.locals().len()
        };
        Some(GpuCurrent {
            poses: hierarchy.locals(),
            version,
            stamped,
        })
    }

    pub(super) fn gathered_topology(&self, hierarchy: &HierarchyScratch) -> bool {
        self.gpu
            .gathered
            .is_some_and(|(topology, ..)| topology == hierarchy.topology_revision())
            && self.gpu.info.len() == hierarchy.links().len()
    }

    pub(super) fn gather_gpu(&mut self, hierarchy: &mut HierarchyScratch, alpha: f32) {
        self.gpu.alpha = alpha;
        self.gpu.hold = false;
        let key = (
            hierarchy.topology_revision(),
            hierarchy.revision(),
            self.revision,
        );
        if self.gpu.gathered == Some(key) {
            engine_profile::counter("interpolation.gpu_gathered", 0.0);
            return;
        }
        let _profile = engine_profile::span("interpolation", "gather_gpu_inputs");
        let changes = hierarchy.take_changes();
        self.gather_nodes(hierarchy, key, changes.as_deref());
        hierarchy.restore_changes(changes);
    }

    fn gather_nodes(
        &mut self,
        hierarchy: &HierarchyScratch,
        key: (u64, u64, u64),
        changes: Option<&[u32]>,
    ) {
        let nodes = hierarchy.links().len();
        let reindexed = self
            .gpu
            .gathered
            .is_none_or(|(topology, ..)| topology != hierarchy.topology_revision())
            || self.gpu.info.len() != nodes;
        if reindexed {
            self.gpu.reindex(nodes);
        }
        self.gpu.gathered = Some(key);
        self.gpu.current_version = (key.0, key.1);
        self.gpu.sequence += 1;
        let sequence = self.gpu.sequence;
        let epoch = self.epoch;
        let first_of_epoch = self.gpu.last_epoch != Some(epoch);
        let superseded = self.gpu.unpresented.take();
        let shift = superseded.is_none()
            && first_of_epoch
            && self
                .gpu
                .last_epoch
                .is_some_and(|last| last.wrapping_add(1) == epoch)
            && self.gpu.tick_begin_revision == Some(self.gpu.last_revision);
        self.gpu.shift = shift;
        if shift {
            self.gpu.shift_sequence = sequence;
        } else if let Some(prior) = superseded {
            self.gpu.shift_sequence = prior;
        }
        let pass = Pass {
            sequence,
            epoch,
            active_tick: self.active,
            this_epoch: epoch as u32,
            shift,
            last_revision: self.gpu.last_revision,
        };
        let records = Records {
            history_nodes: &self.history_nodes,
            intervals: &self.intervals,
            previous: &self.previous,
            indices: &self.indices,
        };
        let sparse = changes.filter(|changes| {
            !reindexed
                && self.gpu.active_listed
                && changes
                    .len()
                    .max(self.gpu.active.len())
                    .max(records.history_nodes.len())
                    <= nodes / SPARSE_SHARE
        });
        let changed = walk(&mut self.gpu, &records, hierarchy, pass, sparse);
        engine_profile::counter("interpolation.gpu_visited", self.gpu.visited as f64);
        let gpu = &mut self.gpu;
        gpu.last_epoch = Some(epoch);
        gpu.last_revision = hierarchy.revision();
        gpu.stamped = changed;
        gpu.depth_counts.clear();
        for &(depth, _) in &gpu.parent_scratch {
            let depth = usize::from(depth);
            if gpu.depth_counts.len() <= depth {
                gpu.depth_counts.resize(depth + 1, 0);
            }
            gpu.depth_counts[depth] += 1;
        }
        gpu.levels.clear();
        let mut end = 0;
        for count in &mut gpu.depth_counts {
            let start = end;
            end += *count;
            gpu.levels.push(end);
            *count = start;
        }
        gpu.parents.clear();
        gpu.parents.resize(gpu.parent_scratch.len(), 0);
        for &(depth, index) in &gpu.parent_scratch {
            let cursor = &mut gpu.depth_counts[usize::from(depth)];
            gpu.parents[*cursor as usize] = index;
            *cursor += 1;
        }
        engine_profile::counter("interpolation.gpu_gathered", 1.0);
        engine_profile::counter("interpolation.gpu_shift", f64::from(u8::from(shift)));
        engine_profile::counter("interpolation.gpu_current_changed", changed[0] as f64);
        engine_profile::counter("interpolation.gpu_previous_changed", changed[1] as f64);
        engine_profile::counter("interpolation.gpu_info_changed", changed[2] as f64);
        engine_profile::counter("interpolation.gpu_authority_changed", changed[3] as f64);
        engine_profile::counter("interpolation.gpu_parents", gpu.parents.len() as f64);
        engine_profile::counter("memory.gpu_input_bytes", gpu.capacity_bytes() as f64);
    }
}

const SPARSE_SHARE: usize = 3;

#[derive(Clone, Copy)]
struct Pass {
    sequence: u64,
    epoch: u64,
    active_tick: bool,
    this_epoch: u32,
    shift: bool,
    last_revision: u64,
}

struct Records<'a> {
    history_nodes: &'a [u32],
    intervals: &'a [Interval],
    previous: &'a [NodePose],
    indices: &'a EntityIndices,
}

struct Walk<'a> {
    pass: Pass,
    hierarchy: &'a HierarchyScratch,
    links: &'a [Link],
    locals: &'a [NodePose],
    local_revisions: &'a [u64],
    history_hints: &'a [u32],
    history_nodes: &'a [u32],
    intervals: &'a [Interval],
    record_previous: &'a [NodePose],
    indices: &'a EntityIndices,
    current_stamps: &'a mut [u64],
    previous: &'a mut [NodePose],
    previous_stamps: &'a mut [u64],
    epochs: &'a mut [[u32; 2]],
    infos: &'a mut [NodeInfo],
    info_stamps: &'a mut [u64],
    authority: &'a mut [PoseRows],
    authority_stamps: &'a mut [u64],
    authority_revisions: &'a mut [u64],
    state: &'a mut [u8],
    parent_scratch: &'a mut Vec<(u8, u32)>,
    changed: [usize; 4],
    lists: &'a mut [Vec<u32>; 4],
    list_limit: usize,
}

impl Walk<'_> {
    #[inline(always)]
    fn stamp(&mut self, stream: usize, index: usize) {
        self.changed[stream] += 1;
        let list = &mut self.lists[stream];
        if list.len() < self.list_limit {
            list.push(index as u32);
        }
    }

    #[inline(always)]
    fn node(&mut self, index: usize) -> (u8, u8) {
        let pass = self.pass;
        let link = &self.links[index];
        let before = self.state[index];
        if !link.topology_valid() {
            let after = before & AUTHORITY;
            self.state[index] = after;
            forget_published(&mut self.epochs[index]);
            let info = [NO_PARENT, 0];
            if self.infos[index] != info {
                self.infos[index] = info;
                self.info_stamps[index] = pass.sequence;
                self.stamp(2, index);
            }
            return (before, after);
        }
        let parent = link.parent();
        let inherited = parent.map_or(0, |parent| self.state[parent]);
        let valid = link.has_global();
        let current = &self.locals[index];
        let hint = self.history_hints[index];
        let record = if hint == u32::MAX {
            None
        } else if self.history_nodes.get(hint as usize) == Some(&(index as u32)) {
            Some(hint as usize)
        } else {
            self.indices
                .get(&self.hierarchy.sources()[index].entity)
                .copied()
        };
        let (own_snap, blends) = match record {
            Some(record) => {
                let interval = &self.intervals[record];
                debug_assert_eq!(
                    interval.scales_compatible,
                    compatible_scales(&self.record_previous[record], current),
                    "an interval's cached scale compatibility is current"
                );
                let blends = interval.blends(pass.epoch, pass.active_tick);
                (!blends, blends)
            }
            None => (false, false),
        };
        let snap = own_snap || inherited & SNAP != 0;
        let parent_active = valid && inherited & ACTIVE != 0;
        let blend = valid && blends && !snap;
        let moved = self.local_revisions[index] > pass.last_revision;
        let active = valid && (blend || parent_active || moved);
        let flags = bit(valid, NODE_VALID)
            | bit(blend, NODE_BLEND)
            | bit(active, NODE_ACTIVE)
            | bit(parent_active, NODE_PARENT_ACTIVE);
        let info = [parent.map_or(NO_PARENT, |parent| parent as u32), flags];
        if self.infos[index] != info {
            self.infos[index] = info;
            self.info_stamps[index] = pass.sequence;
            self.stamp(2, index);
        }
        let mut authority_bit = before & AUTHORITY;
        if active {
            let [published, mut previous_valid] = self.epochs[index];
            let published_last = published != NEVER && published == pass.this_epoch.wrapping_sub(1);
            if pass.shift && published_last {
                previous_valid = pass.this_epoch;
            }
            if blend && previous_valid != pass.this_epoch {
                let record = record.expect("blending nodes have a history record");
                self.previous[index] = *previous_pose(
                    &self.record_previous[record],
                    &self.intervals[record],
                    pass.epoch,
                    current,
                );
                self.previous_stamps[index] = pass.sequence;
                previous_valid = pass.this_epoch;
                self.stamp(1, index);
            }
            if published == NEVER || moved {
                self.current_stamps[index] = pass.sequence;
                self.stamp(0, index);
            }
            self.epochs[index] = [pass.this_epoch, previous_valid];
            if authority_bit != 0 {
                self.authority_revisions[index] = u64::MAX;
                authority_bit = 0;
            }
            if link.has_children() {
                self.parent_scratch.push((link.depth(), index as u32));
            }
        } else {
            forget_published(&mut self.epochs[index]);
            if valid {
                let revision = self.hierarchy.revision_at(index);
                if self.authority_revisions[index] != revision || revision == u64::MAX {
                    let matrix = self
                        .hierarchy
                        .global(index)
                        .expect("valid node has a global");
                    self.authority[index] = pose_rows(matrix);
                    self.authority_revisions[index] = revision;
                    self.authority_stamps[index] = pass.sequence;
                    self.stamp(3, index);
                }
                authority_bit = AUTHORITY;
            }
        }
        let after = bit(snap, SNAP) | bit(active, ACTIVE) | authority_bit;
        self.state[index] = after;
        (before, after)
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "longer than the line ceiling; it may only shrink"
)]
fn walk(
    gpu: &mut GpuInputs,
    records: &Records<'_>,
    hierarchy: &HierarchyScratch,
    pass: Pass,
    sparse: Option<&[u32]>,
) -> [usize; 4] {
    let nodes = hierarchy.links().len();
    let GpuInputs {
        current_stamps,
        previous,
        previous_stamps,
        epochs,
        info: infos,
        info_stamps,
        authority,
        authority_stamps,
        authority_revisions,
        state,
        parent_scratch,
        active,
        next_active,
        active_listed,
        queued,
        queue_generation,
        buckets,
        visited,
        stamped_lists,
        ..
    } = gpu;
    parent_scratch.clear();
    next_active.clear();
    for list in stamped_lists.iter_mut() {
        list.clear();
    }
    let record_count = records.history_nodes.len();
    let mut gather = Walk {
        pass,
        hierarchy,
        links: &hierarchy.links()[..nodes],
        locals: &hierarchy.locals()[..nodes],
        local_revisions: &hierarchy.local_revisions()[..nodes],
        history_hints: &hierarchy.history_hints()[..nodes],
        history_nodes: records.history_nodes,
        intervals: &records.intervals[..record_count],
        record_previous: &records.previous[..record_count],
        indices: records.indices,
        current_stamps: &mut current_stamps[..nodes],
        previous: &mut previous[..nodes],
        previous_stamps: &mut previous_stamps[..nodes],
        epochs: &mut epochs[..nodes],
        infos: &mut infos[..nodes],
        info_stamps: &mut info_stamps[..nodes],
        authority: &mut authority[..nodes],
        authority_stamps: &mut authority_stamps[..nodes],
        authority_revisions: &mut authority_revisions[..nodes],
        state: &mut state[..nodes],
        parent_scratch,
        changed: [0; 4],
        lists: stamped_lists,
        list_limit: nodes / 4,
    };
    let limit = nodes / SPARSE_SHARE;
    let Some(changes) = sparse else {
        return walk_all(
            &mut gather,
            hierarchy,
            active,
            next_active,
            active_listed,
            visited,
        );
    };
    *queue_generation = queue_generation.wrapping_add(1);
    if *queue_generation == 0 {
        queued.fill(0);
        *queue_generation = 1;
    }
    let generation = *queue_generation;
    let links = &hierarchy.links()[..nodes];
    let queued = &mut queued[..nodes];
    let mut enqueue = |buckets: &mut [Vec<u32>], index: usize| {
        if queued[index] != generation {
            queued[index] = generation;
            buckets[usize::from(links[index].depth())].push(index as u32);
            true
        } else {
            false
        }
    };
    let mut seeds = 0;
    for &index in active.iter().chain(changes) {
        seeds += usize::from(enqueue(&mut buckets[..], index as usize));
    }
    for &index in records.history_nodes {
        if (index as usize) < nodes {
            seeds += usize::from(enqueue(&mut buckets[..], index as usize));
        }
    }
    if seeds > limit {
        for bucket in buckets.iter_mut() {
            bucket.clear();
        }
        return walk_all(
            &mut gather,
            hierarchy,
            active,
            next_active,
            active_listed,
            visited,
        );
    }
    let mut count = 0;
    for depth in 0..buckets.len() {
        let bucket = std::mem::take(&mut buckets[depth]);
        count += bucket.len();
        for &index in &bucket {
            let index = index as usize;
            let (before, after) = gather.node(index);
            if after & ACTIVE != 0 {
                next_active.push(index as u32);
            }
            if (before ^ after) & (SNAP | ACTIVE) != 0 {
                for child in hierarchy.children(index) {
                    enqueue(&mut buckets[..], child);
                }
            }
        }
        let mut bucket = bucket;
        bucket.clear();
        buckets[depth] = bucket;
    }
    std::mem::swap(active, next_active);
    *active_listed = true;
    *visited = count;
    gather.changed
}

fn walk_all(
    gather: &mut Walk<'_>,
    hierarchy: &HierarchyScratch,
    active: &mut Vec<u32>,
    next_active: &mut Vec<u32>,
    active_listed: &mut bool,
    visited: &mut usize,
) -> [usize; 4] {
    let nodes = hierarchy.links().len();
    let limit = nodes / SPARSE_SHARE;
    let mut listed = true;
    for &index in hierarchy.order() {
        let (_, after) = gather.node(index);
        if after & ACTIVE != 0 {
            if next_active.len() < limit {
                next_active.push(index as u32);
            } else {
                listed = false;
            }
        }
    }
    std::mem::swap(active, next_active);
    *active_listed = listed;
    *visited = nodes;
    gather.changed
}
