mod block;
mod gpu;
mod presentation;

pub use gpu::{
    GpuCurrent, GpuPresentation, NO_PARENT, NODE_ACTIVE, NODE_BLEND, NODE_PARENT_ACTIVE,
    NODE_VALID, NodeInfo, NodePose,
};

use presentation::Display;
use std::collections::HashSet;

use bevy_ecs::prelude::*;
use glam::{Mat4, Quat};

use crate::entity_indices::EntityIndices;

use crate::{
    FixedTime, FrameTime, GlobalTransform, MAX_HIERARCHY_DEPTH, MAX_TRANSFORM_NODES, Parent,
    Transform,
    hierarchy::{
        HierarchyScratch, Source, TransformIndex, ValueIds, same_local_bits, same_pose_bits,
    },
    valid_global_matrix,
};

/// Opts a local transform into interpolation between its latest two fixed poses.
/// Descendants inherit the displayed parent matrix even without this marker.
#[derive(Component, Debug, Default, Clone, Copy)]
#[component(
    on_insert = crate::hierarchy::tracking::topology_changed,
    on_remove = crate::hierarchy::tracking::topology_changed
)]
pub struct InterpolateTransform;

/// Read-only last prepared display pose from [`PresentedTransforms`].
/// Refreshing authoritative transforms never replaces this retained value.
#[derive(Debug, Default, Clone, Copy)]
pub struct PresentedTransform(GlobalTransform);

impl PresentedTransform {
    pub fn matrix(&self) -> Option<Mat4> {
        self.0.matrix()
    }
    /// Output revision; u64::MAX must never be reused by a consumer cache.
    pub fn revision(&self) -> u64 {
        self.0.revision()
    }
}

/// Selects exact current poses or a bounded fraction of the latest fixed interval.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum PresentationMode {
    #[default]
    Current,
    Interpolated {
        alpha: f32,
    },
}

impl PresentationMode {
    pub fn validate(self) -> Result<(), InterpolationError> {
        if let Self::Interpolated { alpha } = self
            && (!alpha.is_finite() || !(0.0..=1.0).contains(&alpha))
        {
            return Err(InterpolationError { alpha });
        }
        Ok(())
    }
}

/// An invalid presentation fraction, rejected before changing engine state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterpolationError {
    pub alpha: f32,
}

impl std::fmt::Display for InterpolationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "interpolation alpha must be finite and within 0..=1 (received {})",
            self.alpha
        )
    }
}
impl std::error::Error for InterpolationError {}

/// Last extraction identity and bounded interpolation storage observations.
///
/// History counters describe live opt-in records and retained Vec payload;
/// component bytes are zero: derived poses are not ECS components. Storage counters
/// exclude allocator, HashMap, query metadata and Bevy archetype storage overhead.
#[derive(Resource, Debug, Default)]
pub struct PresentationState {
    mode: PresentationMode,
    frame: Option<u64>,
    tick: Option<u64>,
    history_nodes: usize,
    history_capacity: usize,
    history_bytes: usize,
    presented_nodes: usize,
    presented_store_bytes: usize,
    presentation_scratch_bytes: usize,
    gpu_pending: bool,
}

impl PresentationState {
    /// Whether the last extraction left blending and composition to the GPU
    /// and [`PresentedTransforms`] still hold earlier
    /// CPU-prepared values. [`crate::Engine::prepare_presented_transforms`]
    /// materializes that frame while its inputs are unchanged.
    pub fn gpu_pending(&self) -> bool {
        self.gpu_pending
    }
    pub fn mode(&self) -> PresentationMode {
        self.mode
    }
    pub fn frame(&self) -> Option<u64> {
        self.frame
    }
    pub fn tick(&self) -> Option<u64> {
        self.tick
    }
    pub fn history_nodes(&self) -> usize {
        self.history_nodes
    }
    /// Sum of both reusable history Vec capacities, in records.
    pub fn history_capacity(&self) -> usize {
        self.history_capacity
    }
    pub fn history_bytes(&self) -> usize {
        self.history_bytes
    }
    pub fn presented_nodes(&self) -> usize {
        self.presented_nodes
    }
    /// Reserved canonical Display and Entity Vec payload, excluding index-map
    /// and allocator overhead; derived poses have no ECS component allocation.
    pub fn presented_store_bytes(&self) -> usize {
        self.presented_store_bytes
    }
    pub fn presentation_scratch_bytes(&self) -> usize {
        self.presentation_scratch_bytes
    }
}

/// Cold state of one opted-in record: identity, read by structural
/// reconciliation, identity changes, validity transitions and resets.
/// Presentation decisions use the parallel [`Interval`] only, and the
/// interval's previous pose lives in a parallel dense array: written when an
/// interval starts, read only to blend it and to check scale compatibility.
#[derive(Clone, Copy)]
struct History {
    entity: Entity,
    identity: Identity,
    interval_identity: Identity,
    invalidated: bool,
}

/// Hot interval state of the same record in 16 bytes: written by every motion
/// observation and read by every presentation decision.
#[derive(Clone, Copy)]
struct Interval {
    epoch: u64,
    valid: bool,
    previous_valid: bool,
    cancelled: bool,
    // Cached `interval_identity == identity` of the cold record. Every change
    // to either identity refreshes it.
    stable: bool,
    // Cached `compatible_scales(previous, current local)`. Every local change
    // reaches `observe_source`, which refreshes it, as do interval starts.
    scales_compatible: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Identity {
    marker_added: u32,
    transform_added: u32,
    parent: Option<Entity>,
}

impl Interval {
    #[inline]
    fn blends(&self, epoch: u64, active: bool) -> bool {
        !active
            && epoch != 0
            && self.valid
            && (self.epoch != epoch
                || (!self.cancelled
                    && self.previous_valid
                    && self.stable
                    && self.scales_compatible))
    }

    fn cancel(&mut self, epoch: u64) {
        // Cancelled endpoints are never consumed. The next epoch freezes its
        // previous pose from the authoritative source only if it changes.
        self.epoch = epoch;
        self.cancelled = true;
    }
}

impl History {
    fn identify(&mut self, interval: &mut Interval, identity: Identity) {
        self.identity = identity;
        interval.stable = self.interval_identity == identity;
    }
}

/// Starts the interval of `epoch` from `current` once; the interval identity
/// becomes the current identity. The cold record is touched only when the
/// interval identity was unstable.
#[inline(always)]
fn begin_interval(
    history: &mut History,
    interval: &mut Interval,
    previous: &mut NodePose,
    epoch: u64,
    current: &NodePose,
) {
    if interval.epoch == epoch {
        return;
    }
    interval.epoch = epoch;
    *previous = *current;
    interval.previous_valid = interval.valid;
    interval.cancelled = false;
    if !std::mem::replace(&mut interval.stable, true) {
        history.interval_identity = history.identity;
    }
}

/// The pose `epoch`'s interval starts from: its captured previous pose, or
/// `current` when the interval did not start this epoch.
#[inline]
fn previous_pose<'a>(
    previous: &'a NodePose,
    interval: &Interval,
    epoch: u64,
    current: &'a NodePose,
) -> &'a NodePose {
    if interval.epoch == epoch {
        previous
    } else {
        current
    }
}

type HistoryQuery = QueryState<(
    Entity,
    Ref<'static, InterpolateTransform>,
    Ref<'static, Transform>,
    Option<&'static Parent>,
)>;

/// Canonical poses from the last prepared extraction, independent of later
/// authoritative refreshes. Also owns bounded local interpolation history.
#[derive(Resource)]
pub struct PresentedTransforms {
    history_query: HistoryQuery,
    // Parallel per-record arrays in history-query order.
    history: Vec<History>,
    intervals: Vec<Interval>,
    // Interval start poses in upload layout, like the hierarchy's cached
    // locals they are copied from.
    previous: Vec<NodePose>,
    // Hierarchy index of each record. With the hierarchy's per-node history
    // hint it forms a mutually validated pair; topology rebuilds rewrite both.
    history_nodes: Vec<u32>,
    next_history: Vec<History>,
    next_intervals: Vec<Interval>,
    next_previous: Vec<NodePose>,
    next_history_nodes: Vec<u32>,
    indices: EntityIndices,
    epoch: u64,
    active: bool,
    // Invalidity is consumed at the same observation boundary as presentation.
    // A temporarily invalid branch repaired before observation can retain its
    // interval. Duplicates are bounded and coalesced by the record's flag.
    invalidated: Vec<usize>,
    // Traversal flags, canonical presented matrices and their revisions are
    // parallel arrays: consumers of matrices never stream flags, and flags
    // never stream matrices.
    display: Vec<Display>,
    presented: Vec<Mat4>,
    presented_revisions: Vec<u64>,
    dense_display: bool,
    blended_locals: Vec<Transform>,
    touched_display: Vec<usize>,
    blended_roots: Vec<usize>,
    display_work: Vec<usize>,
    presented_entities: Vec<Entity>,
    presented_map: EntityIndices,
    prepared_topology: u64,
    value_ids: ValueIds,
    written_presented: usize,
    // Advances on every change to interval state or the tick boundary flags;
    // with the hierarchy revision it gates GPU input gathering.
    revision: u64,
    gpu: gpu::GpuInputs,
    // Fraction of the last GPU-presented frame, pending CPU preparation.
    gpu_pending: Option<f32>,
}

pub(crate) type InterpolationState = PresentedTransforms;

impl FromWorld for PresentedTransforms {
    fn from_world(world: &mut World) -> Self {
        Self {
            history_query: world.query(),
            history: Vec::new(),
            intervals: Vec::new(),
            previous: Vec::new(),
            history_nodes: Vec::new(),
            next_history: Vec::new(),
            next_intervals: Vec::new(),
            next_previous: Vec::new(),
            next_history_nodes: Vec::new(),
            indices: EntityIndices::default(),
            epoch: 0,
            active: false,
            invalidated: Vec::new(),
            display: Vec::new(),
            presented: Vec::new(),
            presented_revisions: Vec::new(),
            dense_display: false,
            blended_locals: Vec::new(),
            touched_display: Vec::new(),
            blended_roots: Vec::new(),
            display_work: Vec::new(),
            presented_entities: Vec::new(),
            presented_map: EntityIndices::default(),
            prepared_topology: 0,
            value_ids: ValueIds::default(),
            written_presented: 0,
            revision: 0,
            gpu: gpu::GpuInputs::default(),
            gpu_pending: None,
        }
    }
}

impl PresentedTransforms {
    /// Inputs of the latest GPU-prepared frame.
    pub(crate) fn gpu_presentation<'a>(
        &'a self,
        hierarchy: &'a HierarchyScratch,
    ) -> GpuPresentation<'a> {
        self.gpu.view(hierarchy)
    }

    /// Reads the last prepared snapshot. Absence means this entity was not
    /// admitted in that extraction (or none has run); an invalid admitted pose
    /// has matrix() == None. Later authoritative edits and removals do not alter
    /// this snapshot until another extraction prepares it.
    pub fn get(&self, entity: Entity) -> Option<PresentedTransform> {
        let index = *self.presented_map.get(&entity)?;
        Some(self.value_at(index))
    }

    #[inline]
    pub fn get_indexed(
        &self,
        entity: Entity,
        hint: &mut TransformIndex,
    ) -> Option<PresentedTransform> {
        self.get_indexed_with_revision(entity, hint)
            .map(|(value, _)| value)
    }

    /// Validates the full Entity before using a cached slot. Reindexing can
    /// conservatively change a revision even for an equal matrix; u64::MAX is
    /// an uncacheable exhaustion sentinel.
    #[inline]
    pub fn get_indexed_with_revision(
        &self,
        entity: Entity,
        hint: &mut TransformIndex,
    ) -> Option<(PresentedTransform, u64)> {
        let index = if self.presented_entities.get(hint.0) == Some(&entity) {
            hint.0
        } else {
            *self.presented_map.get(&entity)?
        };
        hint.0 = index;
        let value = self.value_at(index);
        Some((value, value.revision()))
    }

    /// Prepared matrix at a store index already validated for its entity during
    /// this preparation, e.g. by [`Self::get_indexed`]. Retained frames read
    /// poses through that mapping instead of copying every changed matrix.
    #[inline]
    pub(crate) fn prepared_matrix(&self, index: usize) -> Option<Mat4> {
        self.display
            .get(index)
            .filter(|display| display.valid())
            .map(|_| self.presented[index])
    }

    pub fn len(&self) -> usize {
        self.presented_entities.len()
    }
    pub fn is_empty(&self) -> bool {
        self.presented_entities.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Entity, PresentedTransform)> + '_ {
        self.presented_entities
            .iter()
            .take(self.display.len())
            .enumerate()
            .map(|(index, &entity)| (entity, self.value_at(index)))
    }

    #[inline]
    fn value_at(&self, index: usize) -> PresentedTransform {
        PresentedTransform(GlobalTransform::read_value(
            self.presented_matrix(index),
            self.presented_revisions[index],
        ))
    }

    #[inline]
    fn presented_matrix(&self, index: usize) -> Option<Mat4> {
        if self.display[index].valid() {
            Some(self.presented[index])
        } else {
            None
        }
    }

    /// Publishes one prepared value. Equal bits and validity keep the revision.
    #[inline]
    fn commit(&mut self, index: usize, value: Option<Mat4>) -> bool {
        let display = &mut self.display[index];
        let matrix = &mut self.presented[index];
        let revision = &mut self.presented_revisions[index];
        let changed = *revision == 0
            || display.valid() != value.is_some()
            || value
                .is_some_and(|value| !crate::hierarchy::same_finite_matrix_bits(*matrix, value));
        if changed {
            if let Some(value) = value {
                *matrix = value;
            }
            display.set_valid(value.is_some());
            *revision = self.value_ids.next();
        }
        changed
    }

    /// Sizes every per-node presentation array to the hierarchy. New slots
    /// start unpublished (revision 0) and are always written by the next commit.
    fn resize_display(&mut self, nodes: usize) {
        if self.display.len() < nodes {
            self.display.reserve_exact(nodes - self.display.len());
            self.presented.reserve_exact(nodes - self.presented.len());
            self.presented_revisions
                .reserve_exact(nodes - self.presented_revisions.len());
        }
        self.display.resize(nodes, Display::default());
        self.presented.resize(nodes, Mat4::IDENTITY);
        self.presented_revisions.resize(nodes, 0);
    }

    fn prepare_membership(&mut self, hierarchy: &HierarchyScratch) {
        if self.prepared_topology == hierarchy.topology_revision
            && self.prepared_topology != u64::MAX
        {
            return;
        }
        self.presented_entities.clear();
        self.presented_map.clear();
        self.presented_entities
            .reserve_exact(hierarchy.sources.len());
        self.presented_map.reserve(hierarchy.sources.len());
        for (index, source) in hierarchy.sources.iter().enumerate() {
            self.presented_entities.push(source.entity);
            self.presented_map.insert(source.entity, index);
        }
        self.resize_display(hierarchy.links.len());
        // A different source order invalidates sparse traversal flags, never
        // silently reuses one entity's scratch as another entity's ancestry.
        for display in &mut self.display {
            display.reset();
        }
        self.touched_display.clear();
        self.dense_display = false;
        self.prepared_topology = hierarchy.topology_revision;
    }

    /// Record of `node` whose hint was stale: resolves the full Entity and
    /// restores the mutual pair. Topology rebuilds make this rare.
    #[cold]
    fn resolve_record(&mut self, node: usize, entity: Entity) -> Option<usize> {
        resolve_record(&mut self.history_nodes, &self.indices, node, entity)
    }

    /// Borrows the record arrays observations update, for one boundary pass.
    #[inline]
    pub(crate) fn observer(&mut self) -> Observer<'_> {
        // Record arrays share one length: a record index checked against
        // `history_nodes` then needs no further bounds checks.
        let records = self.history_nodes.len();
        Observer {
            history: &mut self.history[..records],
            intervals: &mut self.intervals[..records],
            previous: &mut self.previous[..records],
            history_nodes: &mut self.history_nodes,
            indices: &self.indices,
            invalidated: &mut self.invalidated,
            revision: &mut self.revision,
            observed: false,
            epoch: self.epoch,
            active: self.active,
        }
    }

    /// Reconcile opted-in membership only after structural/lifetime changes.
    /// Ordinary fixed motion is observed by the hierarchy's existing source
    /// scan, so tick boundaries and presentation never query every history.
    /// Runs before the hierarchy reindexes: it reads the previous sources, then
    /// clears node hints for the rebuild to re-establish.
    pub(crate) fn reconcile(&mut self, world: &World, hierarchy: &HierarchyScratch) {
        self.revision += 1;
        let count = self.history_query.iter(world).len().min(hierarchy.limit);
        if count == 0 {
            self.history.clear();
            self.intervals.clear();
            self.previous.clear();
            self.history_nodes.clear();
            self.next_history.clear();
            self.next_intervals.clear();
            self.next_previous.clear();
            self.next_history_nodes.clear();
            self.indices.clear();
            self.invalidated.clear();
            return;
        }
        let _profile = engine_profile::span("interpolation", "update_history_in_place");
        let previous_count = self.history.len();
        let mut rebuilding = false;
        for (position, input) in self
            .history_query
            .iter(world)
            .take(hierarchy.limit)
            .enumerate()
        {
            if !rebuilding
                && self
                    .history
                    .get(position)
                    .is_some_and(|old| old.entity == input.0)
            {
                observe_record(
                    &mut self.history[position],
                    &mut self.intervals[position],
                    &mut self.previous[position],
                    self.history_nodes[position],
                    input,
                    self.epoch,
                    self.active,
                    hierarchy,
                );
                continue;
            }
            if !rebuilding {
                // Prefix records already consumed this observation. Remaining
                // query entities cannot be in that unique prefix.
                rebuilding = true;
                self.next_history.clear();
                self.next_intervals.clear();
                self.next_previous.clear();
                self.next_history_nodes.clear();
                self.next_history.reserve_exact(count);
                self.next_intervals.reserve_exact(count);
                self.next_previous.reserve_exact(count);
                self.next_history_nodes.reserve_exact(count);
                self.next_history
                    .extend_from_slice(&self.history[..position]);
                self.next_intervals
                    .extend_from_slice(&self.intervals[..position]);
                self.next_previous
                    .extend_from_slice(&self.previous[..position]);
                self.next_history_nodes
                    .extend_from_slice(&self.history_nodes[..position]);
            }
            let (mut history, mut interval, mut previous, node) = match self.indices.get(&input.0) {
                Some(&index) => (
                    self.history[index],
                    self.intervals[index],
                    self.previous[index],
                    self.history_nodes[index],
                ),
                None => initial_record(&input, self.epoch, hierarchy),
            };
            observe_record(
                &mut history,
                &mut interval,
                &mut previous,
                node,
                input,
                self.epoch,
                self.active,
                hierarchy,
            );
            self.next_history.push(history);
            self.next_intervals.push(interval);
            self.next_previous.push(previous);
            self.next_history_nodes.push(node);
        }
        if rebuilding {
            std::mem::swap(&mut self.history, &mut self.next_history);
            std::mem::swap(&mut self.intervals, &mut self.next_intervals);
            std::mem::swap(&mut self.previous, &mut self.next_previous);
            std::mem::swap(&mut self.history_nodes, &mut self.next_history_nodes);
            self.next_history.clear();
            self.next_intervals.clear();
            self.next_previous.clear();
            self.next_history_nodes.clear();
        } else {
            // Tail removal has no mismatching input row but changes membership.
            self.history.truncate(count);
            self.intervals.truncate(count);
            self.previous.truncate(count);
            self.history_nodes.truncate(count);
        }
        let index_changed = rebuilding || count != previous_count;
        engine_profile::counter(
            "interpolation.history_index_rebuilt",
            if index_changed { 1.0 } else { 0.0 },
        );
        if index_changed {
            let _profile = engine_profile::span("interpolation", "rebuild_history_index");
            self.indices.clear();
            self.indices.reserve(self.history.len());
            for (index, record) in self.history.iter().enumerate() {
                self.indices.insert(record.entity, index);
            }
        }
        // Structural work may reorder slots. Recreate this rare pending list
        // from current validity rather than retaining stale slot indices.
        self.invalidated.clear();
        for (index, (record, interval)) in self.history.iter_mut().zip(&self.intervals).enumerate()
        {
            record.invalidated = !interval.valid;
            if !interval.valid {
                push_invalid(&mut self.invalidated, index);
            }
        }
        // Node indices are about to be reassigned; the rebuild re-establishes
        // every admitted pair through `observe_source`.
        self.history_nodes.fill(u32::MAX);
    }

    /// Observe source values while hierarchy validation already has them hot:
    /// the local before and after this boundary, in upload layout.
    /// `hint` and the record's node index are a mutually validated pair.
    /// `local_changed` reuses the hierarchy's exact old/new bit comparison.
    /// `identity_changed` is false when the source keys that feed the record
    /// identity are unchanged, so only the hot interval can change.
    /// See [`Observer::observe_source`].
    pub(crate) fn observe_source(
        &mut self,
        hint: &mut u32,
        node: usize,
        source: &Source,
        locals: (&NodePose, &NodePose),
        local_changed: bool,
        identity_changed: bool,
    ) {
        self.observer()
            .observe_source(hint, node, source, locals, local_changed, identity_changed);
    }

    fn counters(&self, state: &mut PresentationState) {
        state.history_nodes = self.history.len();
        state.history_capacity = self.history.capacity() + self.next_history.capacity();
        state.history_bytes = (self.history.capacity() + self.next_history.capacity())
            * size_of::<History>()
            + (self.intervals.capacity() + self.next_intervals.capacity()) * size_of::<Interval>()
            + (self.previous.capacity() + self.next_previous.capacity()) * size_of::<NodePose>()
            + (self.history_nodes.capacity() + self.next_history_nodes.capacity())
                * size_of::<u32>();
        state.presented_store_bytes = self.display.capacity() * size_of::<Display>()
            + self.presented.capacity() * size_of::<Mat4>()
            + self.presented_revisions.capacity() * size_of::<u64>()
            + self.presented_entities.capacity() * size_of::<Entity>();
        state.presentation_scratch_bytes = self.blended_locals.capacity() * size_of::<Transform>()
            + (self.touched_display.capacity()
                + self.blended_roots.capacity()
                + self.display_work.capacity()
                + self.invalidated.capacity())
                * size_of::<usize>();
    }

    fn prepare(&mut self, world: &mut World, mode: PresentationMode) -> usize {
        self.observe_invalid();
        self.written_presented = 0;
        let interpolated = match mode {
            PresentationMode::Interpolated { alpha } if alpha < 1.0 && !self.history.is_empty() => {
                Some(alpha)
            }
            _ => None,
        };
        world.resource_scope(|_world, hierarchy: Mut<HierarchyScratch>| {
            self.prepare_membership(&hierarchy);
            if let Some(alpha) = interpolated {
                self.prepare_interpolated(&hierarchy, alpha)
            } else {
                self.apply_current(&hierarchy)
            }
        })
    }

    fn observe_invalid(&mut self) {
        if !self.invalidated.is_empty() {
            self.revision += 1;
        }
        for index in self.invalidated.drain(..) {
            if let Some(record) = self.history.get_mut(index) {
                record.invalidated = false;
                let interval = &mut self.intervals[index];
                if !interval.valid {
                    interval.cancel(self.epoch);
                }
            }
        }
    }

    fn apply_current(&mut self, hierarchy: &HierarchyScratch) -> usize {
        let _profile = engine_profile::span("interpolation", "prepare_current_values");
        self.dense_display = false;
        self.blended_locals.clear();
        self.touched_display.clear();
        self.blended_roots.clear();
        self.display_work.clear();
        let mut written = 0;
        for index in 0..self.display.len() {
            self.display[index].reset();
            written += usize::from(self.commit(index, hierarchy.global(index)));
        }
        engine_profile::counter("interpolation.current_reused", f64::from(written == 0));
        engine_profile::counter("interpolation.written_presented", written as f64);
        hierarchy.valid_count()
    }
}

/// Split borrow of the record arrays that boundary observations update. The
/// hierarchy's scans hold one for a whole pass, so per-record work keeps array
/// pointers and lengths in locals instead of reloading them from the store
/// after every write. The interpolation revision advances once, on drop, when
/// anything was observed: consumers compare revisions only for equality.
pub(crate) struct Observer<'a> {
    history: &'a mut [History],
    intervals: &'a mut [Interval],
    previous: &'a mut [NodePose],
    history_nodes: &'a mut [u32],
    indices: &'a EntityIndices,
    invalidated: &'a mut Vec<usize>,
    revision: &'a mut u64,
    observed: bool,
    epoch: u64,
    active: bool,
}

impl Drop for Observer<'_> {
    fn drop(&mut self) {
        if self.observed {
            *self.revision += 1;
        }
    }
}

impl Observer<'_> {
    /// Record index for a node's hint, validated by the mutual node index.
    #[inline]
    fn record_for(&mut self, hint: u32, node: usize, entity: Entity) -> Option<usize> {
        if self.history_nodes.get(hint as usize) == Some(&(node as u32)) {
            Some(hint as usize)
        } else {
            resolve_record(self.history_nodes, self.indices, node, entity)
        }
    }

    /// The common observation, a local change of an unchanged source identity,
    /// exactly as [`Self::observe_source`] performs it. Returns false without
    /// changing anything unless `hint` is valid for `node`; the caller then
    /// takes the general path.
    #[cfg_attr(not(feature = "attribution"), inline(always))]
    #[cfg_attr(feature = "attribution", inline(never))]
    pub(crate) fn observe_motion(
        &mut self,
        hint: u32,
        node: usize,
        previous_local: &NodePose,
        current: &NodePose,
    ) -> bool {
        let record = hint as usize;
        if self.history_nodes.get(record) != Some(&(node as u32)) {
            return false;
        }
        self.observed = true;
        let interval = &mut self.intervals[record];
        let previous = &mut self.previous[record];
        if self.active {
            begin_interval(
                &mut self.history[record],
                interval,
                previous,
                self.epoch,
                previous_local,
            );
        } else {
            interval.cancel(self.epoch);
        }
        interval.scales_compatible = compatible_scales(previous, current);
        true
    }

    /// Observe source values while hierarchy validation already has them hot:
    /// the local before and after this boundary, in upload layout.
    /// `hint` and the record's node index are a mutually validated pair.
    /// `local_changed` reuses the hierarchy's exact old/new bit comparison.
    /// `identity_changed` is false when the source keys that feed the record
    /// identity are unchanged, so only the hot interval can change.
    #[cfg_attr(feature = "attribution", inline(never))]
    pub(crate) fn observe_source(
        &mut self,
        hint: &mut u32,
        node: usize,
        source: &Source,
        (previous_local, current): (&NodePose, &NodePose),
        local_changed: bool,
        identity_changed: bool,
    ) {
        let Some(marker_added) = source.marker_added else {
            *hint = u32::MAX;
            return;
        };
        self.observed = true;
        let Some(index) = self.record_for(*hint, node, source.entity) else {
            *hint = u32::MAX;
            return;
        };
        *hint = index as u32;
        let history = &mut self.history[index];
        let interval = &mut self.intervals[index];
        let previous = &mut self.previous[index];
        let identity = Identity {
            marker_added,
            transform_added: source.transform_added,
            parent: source.parent,
        };
        if identity_changed {
            observe_local(
                history,
                interval,
                previous,
                previous_local,
                local_changed,
                identity,
                self.epoch,
                self.active,
            );
        } else if local_changed {
            debug_assert!(
                history.identity == identity,
                "unchanged source keys imply an unchanged record identity"
            );
            observe_motion(
                history,
                interval,
                previous,
                previous_local,
                self.epoch,
                self.active,
            );
        }
        interval.scales_compatible = compatible_scales(previous, current);
    }

    #[cfg_attr(feature = "attribution", inline(never))]
    pub(crate) fn observe_validity(
        &mut self,
        hint: u32,
        node: usize,
        local: &NodePose,
        valid: bool,
    ) {
        if self.history_nodes.get(hint as usize) != Some(&(node as u32)) {
            return;
        }
        let index = hint as usize;
        let history = &mut self.history[index];
        let interval = &mut self.intervals[index];
        let previous = &mut self.previous[index];
        if interval.valid == valid {
            return;
        }
        self.observed = true;
        // A repair after an invalid completed tick must not resurrect an old
        // valid interval, even if no local pose changed during that tick.
        begin_interval(history, interval, previous, self.epoch, local);
        interval.scales_compatible = compatible_scales(previous, local);
        if !valid && !history.invalidated {
            push_invalid(self.invalidated, index);
            history.invalidated = true;
        }
        interval.valid = valid;
    }
}

/// Record of `node` whose hint was stale: resolves the full Entity and
/// restores the mutual pair. Topology rebuilds make this rare.
#[cold]
fn resolve_record(
    history_nodes: &mut [u32],
    indices: &EntityIndices,
    node: usize,
    entity: Entity,
) -> Option<usize> {
    let index = *indices.get(&entity)?;
    history_nodes[index] = node as u32;
    Some(index)
}

type HistoryInput<'a> = (
    Entity,
    Ref<'a, InterpolateTransform>,
    Ref<'a, Transform>,
    Option<&'a Parent>,
);

fn push_invalid(indices: &mut Vec<usize>, index: usize) {
    if indices.len() == indices.capacity() {
        let capacity = (indices.capacity().max(16) * 2).min(MAX_TRANSFORM_NODES);
        indices.reserve_exact(capacity - indices.len());
    }
    indices.push(index);
}

fn initial_record(
    input: &HistoryInput<'_>,
    epoch: u64,
    hierarchy: &HierarchyScratch,
) -> (History, Interval, NodePose, u32) {
    let identity = Identity {
        marker_added: input.1.added().get(),
        transform_added: input.2.added().get(),
        parent: input.3.map(Parent::parent),
    };
    let history = History {
        entity: input.0,
        identity,
        interval_identity: identity,
        invalidated: false,
    };
    let previous = input.2.to_pose();
    let interval = Interval {
        epoch,
        valid: hierarchy
            .get(input.0)
            .and_then(|global| global.matrix())
            .is_some(),
        previous_valid: false,
        cancelled: true,
        stable: true,
        scales_compatible: compatible_scales(&previous, &previous),
    };
    (history, interval, previous, u32::MAX)
}

/// `node` is the record's hint into the hierarchy being replaced; it is
/// validated by full Entity against those previous sources.
#[allow(clippy::too_many_arguments)]
fn observe_record(
    history: &mut History,
    interval: &mut Interval,
    previous: &mut NodePose,
    node: u32,
    input: HistoryInput<'_>,
    epoch: u64,
    active: bool,
    hierarchy: &HierarchyScratch,
) {
    let (_, marker, transform, parent) = input;
    let identity = Identity {
        marker_added: marker.added().get(),
        transform_added: transform.added().get(),
        parent: parent.map(Parent::parent),
    };
    let cached = Some(node as usize)
        .filter(|&index| {
            hierarchy
                .sources
                .get(index)
                .is_some_and(|source| source.entity == history.entity)
        })
        .or_else(|| hierarchy.indices.get(&history.entity).copied())
        .map(|index| hierarchy.locals[index]);
    let current = transform.to_pose();
    if let Some(local) = &cached {
        observe_local(
            history,
            interval,
            previous,
            local,
            !same_pose_bits(local, &current),
            identity,
            epoch,
            active,
        );
    } else {
        // No authoritative old source exists for a new/unadmitted lifetime.
        // Start from current until a later valid fixed interval is available.
        history.identify(interval, identity);
        interval.cancel(epoch);
    }
    let valid = hierarchy
        .get(history.entity)
        .and_then(|global| global.matrix())
        .is_some();
    if interval.valid != valid {
        begin_interval(
            history,
            interval,
            previous,
            epoch,
            cached.as_ref().unwrap_or(&current),
        );
    }
    interval.valid = valid;
    // Rebuilding the hierarchy observes the admitted current local again.
    interval.scales_compatible = compatible_scales(previous, &current);
}

#[allow(clippy::too_many_arguments)]
fn observe_local(
    history: &mut History,
    interval: &mut Interval,
    previous: &mut NodePose,
    previous_local: &NodePose,
    local_changed: bool,
    identity: Identity,
    epoch: u64,
    active: bool,
) {
    if history.identity == identity && !local_changed {
        return;
    }
    observe_motion(history, interval, previous, previous_local, epoch, active);
    if history.identity != identity {
        history.identify(interval, identity);
    }
}

/// A local change with the record's current identity.
#[inline]
fn observe_motion(
    history: &mut History,
    interval: &mut Interval,
    previous: &mut NodePose,
    previous_local: &NodePose,
    epoch: u64,
    active: bool,
) {
    if active {
        begin_interval(history, interval, previous, epoch, previous_local);
    } else {
        interval.cancel(epoch);
    }
}

/// Scale components of two poses in upload layout are nonzero (NaN included,
/// as for `!= 0.0`) with equal sign bits. Branch-free over the three.
#[inline]
fn compatible_scales(previous: &NodePose, current: &NodePose) -> bool {
    [3, 8, 9].into_iter().fold(true, |compatible, component| {
        let (a, b) = (previous[component].to_bits(), current[component].to_bits());
        compatible & (a << 1 != 0) & (b << 1 != 0) & ((a ^ b) >> 31 == 0)
    })
}

// About 3.624 degrees of physical rotation. Normalized lerp removes the three
// sine evaluations on ordinary small fixed-step motion. The ideal angular
// deviation is below 1.1e-6 rad; tests allow 1e-5 rad including f32 rounding.
// This is an explicit interpolation policy, not bit-equivalent SLERP.
const LINEAR_ROTATION_DOT_THRESHOLD: f32 = 0.9995;

fn blend_rotation(previous: Quat, current: Quat, alpha: f32) -> Quat {
    let previous = previous.normalize();
    let current = current.normalize();
    if previous.dot(current).abs() >= LINEAR_ROTATION_DOT_THRESHOLD {
        // glam's quaternion lerp normalizes and selects the shortest hemisphere.
        previous.lerp(current, alpha)
    } else {
        previous.slerp(current, alpha).normalize()
    }
}

fn blend(previous: &Transform, current: &Transform, alpha: f32) -> Transform {
    if alpha == 0.0 {
        return *previous;
    }
    let inverse = 1.0 - alpha;
    Transform {
        translation: previous.translation * inverse + current.translation * alpha,
        rotation: blend_rotation(previous.rotation, current.rotation, alpha),
        scale: previous.scale * inverse + current.scale * alpha,
    }
}

pub(crate) fn begin_tick(world: &mut World) {
    let _profile = engine_profile::span("interpolation", "capture_tick_begin");
    world.resource_scope(|world, mut state: Mut<InterpolationState>| {
        state.epoch = state.epoch.wrapping_add(1);
        if state.epoch == 0 {
            // A bounded repair only after 2^64 fixed ticks keeps epoch equality
            // unambiguous without adding wrap checks to every history event.
            state.epoch = 1;
            for interval in &mut state.intervals {
                interval.epoch = 0;
            }
        }
        state.active = true;
        state.revision += 1;
        // GPU presentation keeps last gather's current poses as this tick's
        // previous poses when no boundary changed locals in between.
        let hierarchy = world
            .get_resource::<HierarchyScratch>()
            .map(|hierarchy| hierarchy.revision);
        state.gpu.begin_tick(hierarchy);
        state.counters(&mut world.resource_mut::<PresentationState>());
    });
}
pub(crate) fn end_tick(world: &mut World) {
    let _profile = engine_profile::span("interpolation", "capture_tick_end");
    world.resource_scope(|world, mut state: Mut<InterpolationState>| {
        // Usually empty. Only branches whose validity changed need an end-boundary
        // check; a repair before this point retains the valid endpoint pair.
        state.observe_invalid();
        state.active = false;
        state.revision += 1;
        state.counters(&mut world.resource_mut::<PresentationState>());
    });
}

pub(crate) fn prepare(world: &mut World, mode: PresentationMode) {
    let _profile = engine_profile::span("interpolation", "prepare_presented_transforms");
    world.resource_scope(|world, mut interpolation: Mut<InterpolationState>| {
        interpolation.gpu_pending = None;
        let count = interpolation.prepare(world, mode);
        let frame = world.get_resource::<FrameTime>().map(|time| time.frame);
        let tick = world.get_resource::<FixedTime>().map(|time| time.tick);
        let mut state = world.resource_mut::<PresentationState>();
        state.mode = mode;
        state.frame = frame;
        state.tick = tick;
        state.presented_nodes = count;
        state.gpu_pending = false;
        interpolation.counters(&mut state);
    });
}

/// Prepares inputs for GPU blending/composition of `alpha`
/// instead of CPU presented poses; the CPU store keeps its previous values
/// until [`prepare_pending`] or another CPU preparation.
pub(crate) fn prepare_gpu(world: &mut World, alpha: f32) {
    let _profile = engine_profile::span("interpolation", "prepare_gpu_presentation");
    world.resource_scope(|world, mut interpolation: Mut<InterpolationState>| {
        interpolation.observe_invalid();
        interpolation.gpu_pending = Some(alpha);
        world.resource_scope(|_world, mut hierarchy: Mut<HierarchyScratch>| {
            interpolation.gather_gpu(&mut hierarchy, alpha);
        });
        // This view carries no hold: consumers apply the latest gather now.
        interpolation.gpu.presented();
        let frame = world.get_resource::<FrameTime>().map(|time| time.frame);
        let tick = world.get_resource::<FixedTime>().map(|time| time.tick);
        let mut state = world.resource_mut::<PresentationState>();
        state.mode = PresentationMode::Interpolated { alpha };
        state.frame = frame;
        state.tick = tick;
        state.gpu_pending = true;
        interpolation.counters(&mut state);
    });
}

/// Gathers GPU inputs for a tick completed before its due frame, keeping the
/// displayed fraction, and returns whether it did. The due frame's
/// [`prepare_gpu`] then finds its inputs current and gathers nothing, unless a
/// later boundary changed them. Nothing is gathered before GPU presentation
/// first gathered or after the tick changed the node space: frames that hold
/// consumers at their applied inputs must keep that node space.
pub(crate) fn gather_ahead(world: &mut World) -> bool {
    world.resource_scope(|world, mut interpolation: Mut<InterpolationState>| {
        world.resource_scope(|_world, mut hierarchy: Mut<HierarchyScratch>| {
            if !interpolation.gathered_topology(&hierarchy) {
                return false;
            }
            interpolation.observe_invalid();
            let alpha = interpolation.gpu.alpha();
            let (sequence, prior) = (
                interpolation.gpu.sequence(),
                interpolation.gpu.shift_sequence(),
            );
            interpolation.gather_gpu(&mut hierarchy, alpha);
            if interpolation.gpu.sequence() != sequence {
                interpolation.gpu.gathered_ahead(prior);
            }
            true
        })
    })
}

/// Presents `alpha` of the last gathered GPU inputs while a fixed tick is in
/// flight: its partial changes are neither propagated nor gathered. With
/// `hold` the inputs were gathered ahead for a tick not yet due, and consumers
/// present the inputs they applied before at `alpha`. The presentation state
/// keeps the gathered frame's tick.
pub(crate) fn present_gpu_fraction(world: &mut World, alpha: f32, hold: bool) {
    let _profile = engine_profile::span("interpolation", "present_gpu_fraction");
    world.resource_scope(|world, mut interpolation: Mut<InterpolationState>| {
        interpolation.gpu.set_alpha(alpha, hold);
        interpolation.gpu_pending = Some(alpha);
        let frame = world.get_resource::<FrameTime>().map(|time| time.frame);
        let mut state = world.resource_mut::<PresentationState>();
        state.mode = PresentationMode::Interpolated { alpha };
        state.frame = frame;
        state.gpu_pending = true;
    });
}

/// CPU-prepares the fraction of the last GPU-presented frame: exactly what a CPU
/// extraction of that frame prepares. A no-op when the last extraction was
/// CPU-prepared, and refused once a tick, refresh or reset changed the inputs
/// that frame gathered. Called only on explicit request, never per frame.
pub(crate) fn prepare_pending(world: &mut World) -> bool {
    let Some(alpha) = world
        .get_resource::<InterpolationState>()
        .and_then(|state| {
            let hierarchy = world.get_resource::<HierarchyScratch>()?;
            state
                .gpu_pending
                .filter(|_| state.gathered_current(hierarchy))
        })
    else {
        return false;
    };
    let frame = world.resource::<PresentationState>().frame;
    let tick = world.resource::<PresentationState>().tick;
    prepare(world, PresentationMode::Interpolated { alpha });
    // Identity stays that of the GPU-presented frame.
    let mut state = world.resource_mut::<PresentationState>();
    state.frame = frame;
    state.tick = tick;
    true
}

/// Snaps tracked histories on this root and its current descendants.
///
/// Call after teleports. A reset during FixedSimulation suppresses that pending
/// interval; a later tick can interpolate normally. Returns reset record count.
/// This explicit control scans bounded ancestry and does not advance clocks or
/// change local, authoritative global, or last presented poses.
pub fn reset_interpolation(world: &mut World, root: Entity) -> usize {
    reset_roots(world, &[root])
}

/// Snaps every tracked history, including pending mid-tick intervals.
pub fn reset_all_interpolation(world: &mut World) -> usize {
    let Some(mut state) = world.get_resource_mut::<InterpolationState>() else {
        return 0;
    };
    let epoch = state.epoch;
    for interval in &mut state.intervals {
        interval.cancel(epoch);
    }
    state.revision += 1;
    state.history.len()
}

pub(crate) fn reset_changed_parents(world: &mut World, changes: &[(Entity, Option<Entity>)]) {
    if world
        .get_resource::<InterpolationState>()
        .is_none_or(|state| state.history.is_empty())
    {
        return;
    }
    let roots: Vec<_> = changes.iter().map(|(entity, _)| *entity).collect();
    reset_roots(world, &roots);
}

fn reset_roots(world: &mut World, roots: &[Entity]) -> usize {
    if world
        .get_resource::<InterpolationState>()
        .is_none_or(|state| state.history.is_empty())
    {
        return 0;
    }
    let roots: HashSet<_> = roots.iter().copied().collect();
    world.resource_scope(|world, mut state: Mut<InterpolationState>| {
        let mut count = 0;
        let state = &mut *state;
        state.revision += 1;
        let epoch = state.epoch;
        for (history, interval) in state.history.iter().zip(&mut state.intervals) {
            let mut cursor = Some(history.entity);
            for _ in 0..=MAX_HIERARCHY_DEPTH {
                let Some(entity) = cursor else {
                    break;
                };
                if roots.contains(&entity) {
                    interval.cancel(epoch);
                    count += 1;
                    break;
                }
                cursor = world.get::<Parent>(entity).map(Parent::parent);
            }
        }
        count
    })
}
