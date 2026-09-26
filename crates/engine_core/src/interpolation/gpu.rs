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
    /// Topology and hierarchy revision whose locals `current` holds: equal
    /// versions certify equal current poses (see `Engine::gpu_current`).
    pub current_version: (u64, u64),
    /// This gather describes a tick completed ahead of its due frame: keep
    /// presenting the inputs applied last, at `alpha`, and apply this gather
    /// at the first view without `hold`. A consumer that applied nothing yet
    /// in this `identity` applies it now instead.
    pub hold: bool,
}

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
    // Records the latest gather stamped in each stream.
    stamped: [usize; 4],
    current_version: (u64, u64),
    // Views hold consumers at their last applied gather (see
    // `GpuPresentation::hold`).
    hold: bool,
    // Set while the latest gather was made ahead and no view without `hold`
    // presented it yet: the shift sequence before it, restored if a later
    // gather supersedes it before consumers apply it.
    unpresented: Option<u64>,
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
            current_version: (0, 0),
            hold: false,
            unpresented: None,
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
            + (self.parents.capacity() + self.levels.capacity() + self.depth_counts.capacity())
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
        // The new stream starts without previous poses to shift.
        self.last_epoch = None;
    }
}

impl InterpolationState {
    /// Whether GPU presentation has gathered inputs.
    pub(crate) fn gathers_gpu(&self) -> bool {
        self.gpu.gathered.is_some()
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
    /// Mirrors the dense CPU pass's decisions in the same parent-first order;
    /// the GPU applies the fraction. Skipped when nothing it reads changed.
    pub(super) fn gather_gpu(&mut self, hierarchy: &HierarchyScratch, alpha: f32) {
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
        let nodes = hierarchy.links.len();
        if self
            .gpu
            .gathered
            .is_none_or(|(topology, ..)| topology != hierarchy.topology_revision)
            || self.gpu.info.len() != nodes
        {
            self.gpu.reindex(nodes);
        }
        self.gpu.gathered = Some(key);
        self.gpu.current_version = (key.0, key.1);
        self.gpu.sequence += 1;
        let sequence = self.gpu.sequence;
        let (epoch, active_tick) = (self.epoch, self.active);
        // Epoch comparisons use the low 32 bits; a wrap needs 2^32 fixed ticks.
        let this_epoch = epoch as u32;
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
        let mut changed = [0_usize; 4];
        let last_revision = self.gpu.last_revision;
        // Split borrows: the per-node loop keeps every array's pointer and
        // length in locals instead of reloading them after each write.
        let Self {
            gpu,
            history_nodes,
            intervals,
            previous: record_previous,
            indices,
            ..
        } = self;
        gpu.parent_scratch.clear();
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
            ..
        } = gpu;
        // Per-node and per-record arrays each share one length, so one checked
        // access at an index covers the rest.
        let records = history_nodes.len();
        let (current_stamps, previous, previous_stamps, epochs) = (
            &mut current_stamps[..nodes],
            &mut previous[..nodes],
            &mut previous_stamps[..nodes],
            &mut epochs[..nodes],
        );
        let (infos, info_stamps, authority, authority_stamps, authority_revisions, state) = (
            &mut infos[..nodes],
            &mut info_stamps[..nodes],
            &mut authority[..nodes],
            &mut authority_stamps[..nodes],
            &mut authority_revisions[..nodes],
            &mut state[..nodes],
        );
        let (history_nodes, intervals, record_previous) = (
            &history_nodes[..records],
            &intervals[..records],
            &record_previous[..records],
        );
        let (links, locals, local_revisions, history_hints) = (
            &hierarchy.links[..nodes],
            &hierarchy.locals[..nodes],
            &hierarchy.local_revisions[..nodes],
            &hierarchy.history_hints[..nodes],
        );
        for &index in &hierarchy.order {
            let link = &links[index];
            // The node's state before this gather: its authority bit is
            // current for the rest of the gather.
            let before = state[index];
            if !link.topology_valid() {
                // Invalid ancestry never renders or composes; children are
                // invalid too, so their inherited state is never read.
                state[index] = before & AUTHORITY;
                forget_published(&mut epochs[index]);
                let info = [NO_PARENT, 0];
                if infos[index] != info {
                    infos[index] = info;
                    info_stamps[index] = sequence;
                    changed[2] += 1;
                }
                continue;
            }
            let parent = link.parent();
            let inherited = parent.map_or(0, |parent| state[parent]);
            let valid = link.has_global();
            let current = &locals[index];
            let hint = history_hints[index];
            let record = if hint == u32::MAX {
                None
            } else if history_nodes.get(hint as usize) == Some(&(index as u32)) {
                Some(hint as usize)
            } else {
                indices.get(&hierarchy.sources[index].entity).copied()
            };
            let (own_snap, blends) = match record {
                Some(record) => {
                    let interval = &intervals[record];
                    debug_assert_eq!(
                        interval.scales_compatible,
                        compatible_scales(&record_previous[record], current)
                    );
                    let blends = interval.blends(epoch, active_tick);
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
            let moved = local_revisions[index] > last_revision;
            let active = valid && (blend || parent_active || moved);
            let flags = bit(valid, NODE_VALID)
                | bit(blend, NODE_BLEND)
                | bit(active, NODE_ACTIVE)
                | bit(parent_active, NODE_PARENT_ACTIVE);
            let info = [parent.map_or(NO_PARENT, |parent| parent as u32), flags];
            if infos[index] != info {
                infos[index] = info;
                info_stamps[index] = sequence;
                changed[2] += 1;
            }
            let mut authority_bit = before & AUTHORITY;
            if active {
                let [published, mut previous_valid] = epochs[index];
                let published_last = published != NEVER && published == this_epoch.wrapping_sub(1);
                // The consumer's shift made the last published pose previous.
                if shift && published_last {
                    previous_valid = this_epoch;
                }
                if blend && previous_valid != this_epoch {
                    let record = record.expect("blending nodes have a history record");
                    previous[index] = *previous_pose(
                        &record_previous[record],
                        &intervals[record],
                        epoch,
                        current,
                    );
                    previous_stamps[index] = sequence;
                    previous_valid = this_epoch;
                    changed[1] += 1;
                }
                // The consumer holds this node's local as of the last gather
                // unless the node was not maintained then; upload it when it
                // changed since, without reading the pose itself.
                if published == NEVER || local_revisions[index] > last_revision {
                    current_stamps[index] = sequence;
                    changed[0] += 1;
                }
                epochs[index] = [this_epoch, previous_valid];
                // Active authority rows are never read; rewrite them when the
                // node next becomes inactive.
                if authority_bit != 0 {
                    authority_revisions[index] = u64::MAX;
                    authority_bit = 0;
                }
                if link.has_children() {
                    parent_scratch.push((link.depth(), index as u32));
                }
            } else {
                // Inactive poses are not maintained, so none stays published.
                forget_published(&mut epochs[index]);
                if valid {
                    let revision = hierarchy.revision_at(index);
                    if authority_revisions[index] != revision || revision == u64::MAX {
                        let matrix = hierarchy.global(index).expect("valid node has a global");
                        authority[index] = pose_rows(matrix);
                        authority_revisions[index] = revision;
                        authority_stamps[index] = sequence;
                        changed[3] += 1;
                    }
                    authority_bit = AUTHORITY;
                }
            }
            state[index] = bit(snap, SNAP) | bit(active, ACTIVE) | authority_bit;
        }
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

#[cfg(test)]
mod tests;
