//! Interpolation inputs for GPU presentation (decision 014).
//!
//! Ordinary interpolated browser frames blend and compose displayed poses on
//! the GPU. This gathers exactly the inputs the CPU presentation pass reads —
//! current and previous local poses, effective blend and snap decisions, parent
//! links, and authoritative matrices of nodes whose display does not depend on
//! the fraction — in upload layout, stamping each record that changed. It runs
//! only after hierarchy or interpolation state changed (normally once per fixed
//! tick), never merely because another frame is displayed. The CPU pass stays
//! the exact reference for captures, previews and inspection.
//!
//! Previous poses are usually not sent: after exactly one fixed tick with no
//! other local change since the last gather, every previous pose equals the
//! current pose that gather published, so the consumer copies its current
//! poses into their previous slots on the GPU (`GpuPresentation::shift`) and
//! only current poses upload. Other gathers publish explicit previous poses.

use super::*;
use crate::hierarchy::Link;
use crate::retained_render::{PoseRows, pose_rows};

/// Local pose of one node, 40 bytes:
/// `[t.x, t.y, t.z, s.x, r.x, r.y, r.z, r.w, s.y, s.z]`.
pub type NodePose = [f32; 10];

/// Parent node index (`NO_PARENT` for roots) and `NODE_*` flags.
pub type NodeInfo = [u32; 2];

/// Parent index of roots in [`NodeInfo`].
pub const NO_PARENT: u32 = u32::MAX;
/// Admitted topology and a valid authoritative world pose.
pub const NODE_VALID: u32 = 1;
/// Blends its previous and current local pose by the frame fraction.
pub const NODE_BLEND: u32 = 2;
/// Its displayed pose depends on the fraction: blended here or inherited
/// from an active parent. Inactive valid nodes display their authority rows.
pub const NODE_ACTIVE: u32 = 4;
/// Composes with the parent's GPU-composed pose instead of its authority rows.
pub const NODE_PARENT_ACTIVE: u32 = 8;

// Per-node traversal bits of the gather's parent-first pass, and whether the
// node's authority revision may hold a written revision (otherwise it is
// `u64::MAX`), so active nodes skip reading it.
const SNAP: u8 = 1;
const ACTIVE: u8 = 2;
const AUTHORITY: u8 = 4;
// Epoch (low 32 bits) of a pose never published.
const NEVER: u32 = u32::MAX;

static NEXT_INPUT_STREAM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Borrowed GPU presentation inputs of the latest GPU-prepared frame.
///
/// Node arrays share the hierarchy's index space (`RenderFrame::slot_nodes`
/// maps render slots into it). A record is current when its stamp is at most
/// `sequence`; consumers upload records whose stamp is newer than the sequence
/// they last uploaded, and everything after an `identity` change (topology
/// rebuilds reassign node indices). When `shift` is set, a consumer that
/// applied the gather just before this one first copies every current pose
/// into its previous slot, then applies this gather's stamped records. A
/// consumer that missed that copy must not blend until a later shift it applies.
/// `current` is the hierarchy's cached local of every node, stamped for
/// `NODE_ACTIVE` nodes when it changes; previous poses are maintained for
/// `NODE_BLEND` nodes and authority rows for valid inactive nodes.
#[derive(Clone, Copy)]
pub struct GpuPresentation<'a> {
    /// Fraction of the latest fixed interval to display, in `0..=1`.
    pub alpha: f32,
    pub identity: u64,
    pub sequence: u64,
    /// Copy current poses into previous slots before applying this gather.
    pub shift: bool,
    /// Sequence of the latest gather that set `shift`. A held gather that a
    /// later gather superseded before any view presented it does not count:
    /// the later gather then sends explicit previous poses instead.
    pub shift_sequence: u64,
    pub current: &'a [NodePose],
    pub current_stamps: &'a [u64],
    pub previous: &'a [NodePose],
    pub previous_stamps: &'a [u64],
    pub info: &'a [NodeInfo],
    pub info_stamps: &'a [u64],
    pub authority: &'a [PoseRows],
    pub authority_stamps: &'a [u64],
    /// Active nodes with children, ordered by depth; compose these first.
    pub parents: &'a [u32],
    /// End offset in `parents` of each depth level, shallowest first.
    pub levels: &'a [u32],
    /// Records stamped with `sequence` in the current, previous, info and
    /// authority streams. A consumer that applied the gather just before this
    /// one can skip a stream with no stamped records and copy a mostly stamped
    /// stream whole instead of scanning its stamps.
    pub stamped: [usize; 4],
    /// Per stream, the node of every record stamped with `sequence`, in no
    /// particular order, when the gather listed them all (at most a quarter
    /// of the nodes). A consumer one gather behind can upload these records
    /// without scanning stamps.
    pub stamped_nodes: [Option<&'a [u32]>; 4],
    /// Topology and hierarchy revision whose locals `current` holds: equal
    /// versions certify equal current poses (see `Engine::gpu_current`).
    pub current_version: (u64, u64),
    /// This gather describes a tick completed ahead of its due frame: keep
    /// presenting the inputs applied last, at `alpha`, and apply this gather
    /// at the first view without `hold`. A consumer that applied nothing yet
    /// in this `identity` applies it now instead.
    pub hold: bool,
}

/// GPU presentation's current poses between frames (see
/// `Engine::gpu_current`), for a renderer to stage before the frame that
/// uploads them.
#[derive(Clone, Copy)]
pub struct GpuCurrent<'a> {
    /// The hierarchy's cached local of every node, in upload layout.
    pub poses: &'a [NodePose],
    /// [`GpuPresentation::current_version`] of `poses`: equal versions
    /// certify equal poses.
    pub version: (u64, u64),
    /// Current poses the latest gather stamped when it has `version`, and
    /// every pose otherwise. A gather stamping most of them uploads the
    /// whole stream; other gathers upload only the stamped records.
    pub stamped: usize,
}

#[derive(Clone)]
pub(crate) struct GpuInputs {
    identity: u64,
    sequence: u64,
    alpha: f32,
    // Revisions of the state the last gather read; any change regathers.
    gathered: Option<(u64, u64, u64)>,
    current_stamps: Vec<u64>,
    previous: Vec<NodePose>,
    previous_stamps: Vec<u64>,
    // Per node: epochs (low 32 bits) at which the published current pose was
    // last confirmed, and for which the consumer's previous pose is valid.
    epochs: Vec<[u32; 2]>,
    info: Vec<NodeInfo>,
    info_stamps: Vec<u64>,
    authority: Vec<PoseRows>,
    authority_stamps: Vec<u64>,
    // Hierarchy output revision each authority row was written from.
    authority_revisions: Vec<u64>,
    parents: Vec<u32>,
    levels: Vec<u32>,
    // Parent-first traversal state and a depth-bucket scratch for `parents`.
    state: Vec<u8>,
    depth_counts: Vec<u32>,
    parent_scratch: Vec<(u8, u32)>,
    // Interpolation epoch and hierarchy revision of the last gather, and the
    // hierarchy revision when the latest fixed tick began.
    last_epoch: Option<u64>,
    last_revision: u64,
    tick_begin_revision: Option<u64>,
    shift: bool,
    shift_sequence: u64,
    // Records the latest gather stamped in each stream, and their nodes while
    // the lists stay within their bound.
    stamped: [usize; 4],
    stamped_lists: [Vec<u32>; 4],
    current_version: (u64, u64),
    // Views hold consumers at their last applied gather (see
    // `GpuPresentation::hold`).
    hold: bool,
    // Set while the latest gather was made ahead and no view without `hold`
    // presented it yet: the shift sequence before it, restored if a later
    // gather supersedes it before consumers apply it.
    unpresented: Option<u64>,
    // Nodes the last gather made `NODE_ACTIVE`, when it listed every one
    // (`active_listed`): a sparse gather revisits them all.
    active: Vec<u32>,
    next_active: Vec<u32>,
    active_listed: bool,
    // Sparse gathers: per-node queue marks (a node is queued when its mark
    // equals the generation) and one queue per depth.
    queued: Vec<u32>,
    queue_generation: u32,
    buckets: Vec<Vec<u32>>,
    // Nodes the latest gather visited.
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

/// Marks a node's published poses as not maintained, writing only on change.
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
    /// Current poses are the hierarchy's cached locals.
    pub(crate) fn view<'a>(&'a self, hierarchy: &'a HierarchyScratch) -> GpuPresentation<'a> {
        GpuPresentation {
            alpha: self.alpha,
            identity: self.identity,
            sequence: self.sequence,
            shift: self.shift,
            shift_sequence: self.shift_sequence,
            current: &hierarchy.locals,
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

    /// Records the hierarchy revision at the start of a fixed tick.
    pub(super) fn begin_tick(&mut self, hierarchy_revision: Option<u64>) {
        self.tick_begin_revision = hierarchy_revision;
    }

    /// Displays another fraction of the gathered inputs without regathering,
    /// or, with `hold`, of the inputs consumers applied before them.
    pub(super) fn set_alpha(&mut self, alpha: f32, hold: bool) {
        self.alpha = alpha;
        self.hold = hold;
        if !hold {
            self.presented();
        }
    }

    /// A view without `hold` presents the latest gather to consumers.
    pub(super) fn presented(&mut self) {
        self.unpresented = None;
    }

    /// Marks the latest gather as made ahead of its due frame; `prior` is the
    /// shift sequence before it.
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

    /// Resets per-node storage for a new node index space.
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
        // Never matches a gathered record, so every node is stamped once.
        reset(&mut self.info, nodes, [u32::MAX; 2]);
        reset(&mut self.authority, nodes, [[0.0; 4]; 3]);
        reset(&mut self.state, nodes, 0);
        reset(&mut self.queued, nodes, 0);
        self.queue_generation = 0;
        self.buckets.resize_with(MAX_HIERARCHY_DEPTH + 1, Vec::new);
        self.active.clear();
        self.active_listed = false;
        // The new stream starts without previous poses to shift.
        self.last_epoch = None;
    }

    /// Every output a gather writes, compared bit for bit (parents and the
    /// active set as sets: their order does not reach consumers).
    #[cfg(test)]
    fn assert_same_outputs(&self, actual: &Self, stamped: [usize; 4]) {
        let poses = |poses: &[NodePose]| -> Vec<[u32; 10]> {
            poses.iter().map(|pose| pose.map(f32::to_bits)).collect()
        };
        let rows = |rows: &[PoseRows]| -> Vec<[[u32; 4]; 3]> {
            rows.iter()
                .map(|rows| rows.map(|row| row.map(f32::to_bits)))
                .collect()
        };
        assert_eq!(self.current_stamps, actual.current_stamps, "current stamps");
        assert_eq!(poses(&self.previous), poses(&actual.previous), "previous");
        assert_eq!(
            self.previous_stamps, actual.previous_stamps,
            "previous stamps"
        );
        assert_eq!(self.epochs, actual.epochs, "epochs");
        assert_eq!(self.info, actual.info, "info");
        assert_eq!(self.info_stamps, actual.info_stamps, "info stamps");
        assert_eq!(rows(&self.authority), rows(&actual.authority), "authority");
        assert_eq!(
            self.authority_stamps, actual.authority_stamps,
            "authority stamps"
        );
        assert_eq!(
            self.authority_revisions, actual.authority_revisions,
            "authority revisions"
        );
        assert_eq!(self.state, actual.state, "traversal state");
        let sorted = |values: &[(u8, u32)]| {
            let mut values = values.to_vec();
            values.sort_unstable();
            values
        };
        assert_eq!(
            sorted(&self.parent_scratch),
            sorted(&actual.parent_scratch),
            "active parents"
        );
        assert!(actual.active_listed);
        let set = |nodes: &[u32]| {
            let mut nodes = nodes.to_vec();
            nodes.sort_unstable();
            nodes
        };
        if self.active_listed {
            assert_eq!(set(&self.active), set(&actual.active), "active set");
        }
        for (stream, (expected, actual)) in self
            .stamped_lists
            .iter()
            .zip(&actual.stamped_lists)
            .enumerate()
        {
            // Lists hold every stamped record within their bound.
            let complete = expected.len() == stamped[stream];
            assert_eq!(complete, actual.len() == stamped[stream]);
            if complete {
                assert_eq!(
                    set(expected),
                    set(actual),
                    "stamped nodes of stream {stream}"
                );
            }
        }
    }
}

impl InterpolationState {
    /// Whether GPU presentation has gathered inputs.
    pub(crate) fn gathers_gpu(&self) -> bool {
        self.gpu.gathered.is_some()
    }

    /// Current poses now, with the latest gather's stamp count when it
    /// gathered them; `None` before GPU presentation first gathered.
    pub(crate) fn gpu_current<'a>(
        &self,
        hierarchy: &'a HierarchyScratch,
    ) -> Option<GpuCurrent<'a>> {
        if !self.gathers_gpu() {
            return None;
        }
        let version = (hierarchy.topology_revision, hierarchy.revision);
        let stamped = if self.gpu.current_version == version {
            self.gpu.stamped[0]
        } else {
            hierarchy.locals.len()
        };
        Some(GpuCurrent {
            poses: &hierarchy.locals,
            version,
            stamped,
        })
    }

    /// Whether the last gather used the hierarchy's current node space.
    pub(super) fn gathered_topology(&self, hierarchy: &HierarchyScratch) -> bool {
        self.gpu
            .gathered
            .is_some_and(|(topology, ..)| topology == hierarchy.topology_revision)
            && self.gpu.info.len() == hierarchy.links.len()
    }

    /// Whether the last gather still describes the current inputs.
    pub(super) fn gathered_current(&self, hierarchy: &HierarchyScratch) -> bool {
        self.gpu.gathered
            == Some((
                hierarchy.topology_revision,
                hierarchy.revision,
                self.revision,
            ))
    }

    /// Gathers GPU inputs for displaying `alpha` of the latest fixed interval.
    /// Mirrors the dense CPU pass's decisions in parent-first order; the GPU
    /// applies the fraction. Skipped when nothing it reads changed. Drains the
    /// hierarchy's change log: when few nodes changed, only the nodes whose
    /// inputs can differ are visited (see [`walk`]).
    pub(super) fn gather_gpu(&mut self, hierarchy: &mut HierarchyScratch, alpha: f32) {
        self.gpu.alpha = alpha;
        self.gpu.hold = false;
        let key = (
            hierarchy.topology_revision,
            hierarchy.revision,
            self.revision,
        );
        if self.gpu.gathered == Some(key) {
            engine_profile::counter("interpolation.gpu_gathered", 0.0);
            return;
        }
        let _profile = engine_profile::span("interpolation", "gather_gpu_inputs");
        let changes = hierarchy.changes.take();
        self.gather_nodes(hierarchy, key, changes.as_deref());
        hierarchy.changes.restore(changes);
    }

    fn gather_nodes(
        &mut self,
        hierarchy: &HierarchyScratch,
        key: (u64, u64, u64),
        changes: Option<&[u32]>,
    ) {
        let nodes = hierarchy.links.len();
        let reindexed = self
            .gpu
            .gathered
            .is_none_or(|(topology, ..)| topology != hierarchy.topology_revision)
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
        // Exactly one tick since the last gather, and no boundary changed a
        // local between that gather and the tick: every interval's previous
        // pose is the current pose the last gather published.
        // A gather made ahead and superseded before any view presented it
        // never reached consumers: they still hold the gather before it, so
        // this one sends explicit previous poses and the skipped gather's
        // shift does not count as one they missed.
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
            // Epoch comparisons use the low 32 bits; a wrap needs 2^32 fixed ticks.
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
        // Sparse when the nodes it must visit are a small share of all nodes;
        // a new node space, an overflowed log or an unlisted active set walks
        // every node.
        let sparse = changes.filter(|changes| {
            !reindexed
                && self.gpu.active_listed
                && changes.len() + self.gpu.active.len() + records.history_nodes.len()
                    <= nodes / SPARSE_SHARE
        });
        #[cfg(test)]
        let reference = sparse.is_some().then(|| self.gpu.clone());
        let changed = walk(&mut self.gpu, &records, hierarchy, pass, sparse);
        #[cfg(test)]
        if let Some(mut reference) = reference {
            // Every sparse gather reproduces the full walk exactly.
            let expected = walk(&mut reference, &records, hierarchy, pass, None);
            assert_eq!(changed, expected, "stamped records per stream");
            reference.assert_same_outputs(&self.gpu, expected);
        }
        engine_profile::counter(
            "interpolation.gpu_visited",
            sparse.map_or(nodes, |_| self.gpu.visited) as f64,
        );
        let gpu = &mut self.gpu;
        gpu.last_epoch = Some(epoch);
        gpu.last_revision = hierarchy.revision;
        gpu.stamped = changed;
        // Active parents by depth: every level composes after its ancestors.
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

/// A sparse gather visits at most this share of the nodes (one in N): more
/// visits cost about as much as streaming every node in parent-first order.
const SPARSE_SHARE: usize = 3;

/// Decisions one gather shares across every node.
#[derive(Clone, Copy)]
struct Pass {
    sequence: u64,
    epoch: u64,
    active_tick: bool,
    this_epoch: u32,
    shift: bool,
    last_revision: u64,
}

/// Interpolation records a gather reads.
struct Records<'a> {
    history_nodes: &'a [u32],
    intervals: &'a [Interval],
    previous: &'a [NodePose],
    indices: &'a HashMap<Entity, usize>,
}

/// Split borrow of the per-node arrays a gather reads and writes, each viewed
/// at the node count so one checked access covers the others.
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
    indices: &'a HashMap<Entity, usize>,
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
    // Records stamped per stream: current, previous, info, authority; and
    // their nodes up to `list_limit` each.
    changed: [usize; 4],
    lists: &'a mut [Vec<u32>; 4],
    list_limit: usize,
}

impl Walk<'_> {
    /// Counts a record stamped in `stream`, and lists it within the bound.
    #[inline(always)]
    fn stamp(&mut self, stream: usize, index: usize) {
        self.changed[stream] += 1;
        let list = &mut self.lists[stream];
        if list.len() < self.list_limit {
            list.push(index as u32);
        }
    }

    /// Gathers one node after its parent, returning its traversal bits before
    /// and after. A node's inputs are its topology, validity, cached local
    /// and output revisions, its interpolation record and its parent's bits.
    #[inline(always)]
    fn node(&mut self, index: usize) -> (u8, u8) {
        let pass = self.pass;
        let link = &self.links[index];
        // The node's state before this gather: its authority bit is current
        // for the rest of the gather.
        let before = self.state[index];
        if !link.topology_valid() {
            // Invalid ancestry never renders or composes; children are
            // invalid too, so their inherited state is never read.
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
                .get(&self.hierarchy.sources[index].entity)
                .copied()
        };
        let (own_snap, blends) = match record {
            Some(record) => {
                let interval = &self.intervals[record];
                debug_assert_eq!(
                    interval.scales_compatible,
                    compatible_scales(&self.record_previous[record], current)
                );
                let blends = interval.blends(pass.epoch, pass.active_tick);
                (!blends, blends)
            }
            None => (false, false),
        };
        let snap = own_snap || inherited & SNAP != 0;
        let parent_active = valid && inherited & ACTIVE != 0;
        let blend = valid && blends && !snap;
        // A node whose local changed since the last gather also composes
        // on the GPU, from its current local without blending: uploading
        // that local beats composing and uploading its authority rows.
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
            // The consumer's shift made the last published pose previous.
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
            // The consumer holds this node's local as of the last gather
            // unless the node was not maintained then; upload it when it
            // changed since, without reading the pose itself.
            if published == NEVER || moved {
                self.current_stamps[index] = pass.sequence;
                self.stamp(0, index);
            }
            self.epochs[index] = [pass.this_epoch, previous_valid];
            // Active authority rows are never read; rewrite them when the
            // node next becomes inactive.
            if authority_bit != 0 {
                self.authority_revisions[index] = u64::MAX;
                authority_bit = 0;
            }
            if link.has_children() {
                self.parent_scratch.push((link.depth(), index as u32));
            }
        } else {
            // Inactive poses are not maintained, so none stays published.
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

/// Gathers every node in parent-first order, or with `sparse` (the nodes
/// whose local or output changed since the last gather) only the nodes whose
/// inputs can differ from the last gather, by depth: the nodes active then,
/// every interpolation record's node (blend decisions follow the tick), the
/// changed nodes, and the children of every visited node whose snap or active
/// bit changed. Any other node has unchanged inputs and outputs: it was not
/// active, has no record, and its parent's bits are unchanged, so it stays
/// inactive with unchanged authority rows. Returns the stamps per stream.
fn walk(
    gpu: &mut GpuInputs,
    records: &Records<'_>,
    hierarchy: &HierarchyScratch,
    pass: Pass,
    sparse: Option<&[u32]>,
) -> [usize; 4] {
    let nodes = hierarchy.links.len();
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
        links: &hierarchy.links[..nodes],
        locals: &hierarchy.locals[..nodes],
        local_revisions: &hierarchy.local_revisions[..nodes],
        history_hints: &hierarchy.history_hints[..nodes],
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
    let Some(changes) = sparse else {
        // A later sparse gather needs the complete active set; beyond its
        // bound none can be sparse, so stop listing.
        let limit = nodes / SPARSE_SHARE;
        let mut listed = true;
        for &index in &hierarchy.order {
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
        return gather.changed;
    };
    *queue_generation = queue_generation.wrapping_add(1);
    if *queue_generation == 0 {
        queued.fill(0);
        *queue_generation = 1;
    }
    let generation = *queue_generation;
    let links = &hierarchy.links[..nodes];
    let queued = &mut queued[..nodes];
    // Each node is queued once, in the bucket of its depth: parents always
    // precede their children, which sit one level deeper.
    let mut enqueue = |buckets: &mut [Vec<u32>], index: usize| {
        if queued[index] != generation {
            queued[index] = generation;
            buckets[usize::from(links[index].depth())].push(index as u32);
        }
    };
    for &index in active.iter().chain(changes) {
        enqueue(&mut buckets[..], index as usize);
    }
    for &index in records.history_nodes {
        // Records beyond the node budget have no node.
        if (index as usize) < nodes {
            enqueue(&mut buckets[..], index as usize);
        }
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

#[cfg(test)]
mod tests;
