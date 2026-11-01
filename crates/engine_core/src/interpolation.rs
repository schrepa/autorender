mod gpu;

pub use gpu::{
    GpuCurrent, GpuPresentation, NO_PARENT, NODE_ACTIVE, NODE_BLEND, NODE_PARENT_ACTIVE,
    NODE_VALID, NodeInfo, NodePose, PoseRows,
};

use crate::collections::HashSet;

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemState;
use glam::Mat4;

use crate::entity_indices::EntityIndices;

use crate::{
    FixedTime, FrameTime, GlobalTransform, GlobalTransforms, InterpolateTransform,
    MAX_HIERARCHY_DEPTH, MAX_TRANSFORM_NODES, Parent, Transform,
    hierarchy::{HierarchyScratch, Source, TransformIndex, ValueIds, same_pose_bits},
};

pub(crate) fn track_markers(world: &mut World) {
    world
        .register_component_hooks::<InterpolateTransform>()
        .on_insert(crate::hierarchy::tracking::topology_changed)
        .on_remove(crate::hierarchy::tracking::topology_changed);
}

type ResetQuery = SystemState<
    Query<
        'static,
        'static,
        (Entity, Ref<'static, InterpolateTransform>),
        Changed<InterpolateTransform>,
    >,
>;

pub(crate) struct ResetWatch {
    query: ResetQuery,
    moved: Vec<Entity>,
}

impl ResetWatch {
    pub(crate) fn new(world: &mut World) -> Self {
        Self {
            query: SystemState::new(world),
            moved: Vec::new(),
        }
    }

    pub(crate) fn snap(&mut self, world: &mut World) {
        self.moved.clear();
        if let Ok(changed) = self.query.get(world) {
            self.moved.extend(
                changed
                    .iter()
                    .filter(|(_, marker)| !marker.is_added())
                    .map(|(entity, _)| entity),
            );
        }
        if !self.moved.is_empty() {
            reset_roots(world, &self.moved);
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PresentedTransform(GlobalTransform);

impl PresentedTransform {
    pub fn matrix(&self) -> Option<Mat4> {
        self.0.matrix()
    }
    pub fn revision(&self) -> u64 {
        self.0.revision()
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum PresentationMode {
    #[default]
    Current,
    Interpolated {
        alpha: f32,
    },
}

impl PresentationMode {
    pub fn validate(self) -> Result<(), engine_settings::Error> {
        if let Self::Interpolated { alpha } = self
            && (!alpha.is_finite() || !(0.0..=1.0).contains(&alpha))
        {
            return Err(engine_settings::Error::new(
                engine_settings::ErrorCode::InvalidPresentationAlpha,
                "$.alpha",
            )
            .range(0.0, 1.0)
            .got(alpha));
        }
        Ok(())
    }
}

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
    pub fn history_capacity(&self) -> usize {
        self.history_capacity
    }
    pub fn history_bytes(&self) -> usize {
        self.history_bytes
    }
    pub fn presented_nodes(&self) -> usize {
        self.presented_nodes
    }
    pub fn presented_store_bytes(&self) -> usize {
        self.presented_store_bytes
    }
    pub fn presentation_scratch_bytes(&self) -> usize {
        self.presentation_scratch_bytes
    }
}

#[derive(Clone, Copy)]
struct History {
    entity: Entity,
    identity: Identity,
    interval_identity: Identity,
    invalidated: bool,
}

#[derive(Clone, Copy)]
struct Interval {
    epoch: u64,
    valid: bool,
    previous_valid: bool,
    cancelled: bool,
    stable: bool,
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

#[derive(Resource)]
pub struct PresentedTransforms {
    history_query: HistoryQuery,
    history: Vec<History>,
    intervals: Vec<Interval>,
    previous: Vec<NodePose>,
    history_nodes: Vec<u32>,
    next_history: Vec<History>,
    next_intervals: Vec<Interval>,
    next_previous: Vec<NodePose>,
    next_history_nodes: Vec<u32>,
    indices: EntityIndices,
    epoch: u64,
    active: bool,
    invalidated: Vec<usize>,
    valid: Vec<bool>,
    presented: Vec<Mat4>,
    presented_revisions: Vec<u64>,
    presented_entities: Vec<Entity>,
    presented_map: EntityIndices,
    prepared_topology: u64,
    value_ids: ValueIds,
    revision: u64,
    gpu: gpu::GpuInputs,
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
            valid: Vec::new(),
            presented: Vec::new(),
            presented_revisions: Vec::new(),
            presented_entities: Vec::new(),
            presented_map: EntityIndices::default(),
            prepared_topology: 0,
            value_ids: ValueIds::default(),
            revision: 0,
            gpu: gpu::GpuInputs::default(),
        }
    }
}

impl PresentedTransforms {
    pub fn gpu_presentation<'a>(
        &'a self,
        hierarchy: &'a GlobalTransforms,
    ) -> GpuPresentation<'a> {
        self.gpu.view(hierarchy)
    }

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

    #[inline]
    pub(crate) fn prepared_matrix(&self, index: usize) -> Option<Mat4> {
        self.valid
            .get(index)
            .filter(|&&valid| valid)
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
            .take(self.valid.len())
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
        if self.valid[index] {
            Some(self.presented[index])
        } else {
            None
        }
    }

    #[inline]
    fn commit(&mut self, index: usize, value: Option<Mat4>) -> bool {
        let valid = &mut self.valid[index];
        let matrix = &mut self.presented[index];
        let revision = &mut self.presented_revisions[index];
        let changed = *revision == 0
            || *valid != value.is_some()
            || value
                .is_some_and(|value| !crate::hierarchy::same_finite_matrix_bits(*matrix, value));
        if changed {
            if let Some(value) = value {
                *matrix = value;
            }
            *valid = value.is_some();
            *revision = self.value_ids.next();
        }
        changed
    }

    fn resize_presented(&mut self, nodes: usize) {
        if self.valid.len() < nodes {
            self.valid.reserve_exact(nodes - self.valid.len());
            self.presented.reserve_exact(nodes - self.presented.len());
            self.presented_revisions
                .reserve_exact(nodes - self.presented_revisions.len());
        }
        self.valid.resize(nodes, false);
        self.presented.resize(nodes, Mat4::IDENTITY);
        self.presented_revisions.resize(nodes, 0);
    }

    fn prepare_membership(&mut self, hierarchy: &HierarchyScratch) {
        if self.prepared_topology == hierarchy.topology_revision()
            && self.prepared_topology != u64::MAX
        {
            return;
        }
        self.presented_entities.clear();
        self.presented_map.clear();
        self.presented_entities
            .reserve_exact(hierarchy.sources().len());
        self.presented_map.reserve(hierarchy.sources().len());
        for (index, source) in hierarchy.sources().iter().enumerate() {
            self.presented_entities.push(source.entity);
            self.presented_map.insert(source.entity, index);
        }
        self.resize_presented(hierarchy.links().len());
        self.prepared_topology = hierarchy.topology_revision();
    }

    #[inline]
    pub(crate) fn observer(&mut self) -> Observer<'_> {
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

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    pub(crate) fn reconcile(&mut self, world: &World, hierarchy: &HierarchyScratch) {
        self.revision += 1;
        let count = self
            .history_query
            .iter(world)
            .len()
            .min(hierarchy.node_limit());
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
            .take(hierarchy.node_limit())
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
        self.invalidated.clear();
        for (index, (record, interval)) in self.history.iter_mut().zip(&self.intervals).enumerate()
        {
            record.invalidated = !interval.valid;
            if !interval.valid {
                push_invalid(&mut self.invalidated, index);
            }
        }
        self.history_nodes.fill(u32::MAX);
    }

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
        state.presented_store_bytes = self.valid.capacity() * size_of::<bool>()
            + self.presented.capacity() * size_of::<Mat4>()
            + self.presented_revisions.capacity() * size_of::<u64>()
            + self.presented_entities.capacity() * size_of::<Entity>();
        state.presentation_scratch_bytes = self.invalidated.capacity() * size_of::<usize>();
    }

    fn prepare(&mut self, world: &mut World) -> usize {
        self.observe_invalid();
        world.resource_scope(|_world, hierarchy: Mut<HierarchyScratch>| {
            self.prepare_membership(&hierarchy);
            self.apply_current(&hierarchy)
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
        let mut written = 0;
        for index in 0..self.valid.len() {
            written += usize::from(self.commit(index, hierarchy.global(index)));
        }
        engine_profile::counter("interpolation.current_reused", f64::from(written == 0));
        engine_profile::counter("interpolation.written_presented", written as f64);
        hierarchy.valid_count()
    }
}

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
    #[inline]
    fn record_for(&mut self, hint: u32, node: usize, entity: Entity) -> Option<usize> {
        if self.history_nodes.get(hint as usize) == Some(&(node as u32)) {
            Some(hint as usize)
        } else {
            resolve_record(self.history_nodes, self.indices, node, entity)
        }
    }

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
            debug_assert!(history.identity == identity, "record_identity_unchanged");
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
        begin_interval(history, interval, previous, self.epoch, local);
        interval.scales_compatible = compatible_scales(previous, local);
        if !valid && !history.invalidated {
            push_invalid(self.invalidated, index);
            history.invalidated = true;
        }
        interval.valid = valid;
    }
}

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

#[expect(clippy::too_many_arguments, reason = "history and record, apart")]
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
                .sources()
                .get(index)
                .is_some_and(|source| source.entity == history.entity)
        })
        .or_else(|| hierarchy.node_index(history.entity))
        .map(|index| hierarchy.locals()[index]);
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
    interval.scales_compatible = compatible_scales(previous, &current);
}

#[expect(clippy::too_many_arguments, reason = "history and record, apart")]
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

#[inline]
fn compatible_scales(previous: &NodePose, current: &NodePose) -> bool {
    [3, 8, 9].into_iter().fold(true, |compatible, component| {
        let (a, b) = (previous[component].to_bits(), current[component].to_bits());
        compatible & (a << 1 != 0) & (b << 1 != 0) & ((a ^ b) >> 31 == 0)
    })
}

pub(crate) fn begin_tick(world: &mut World) {
    let _profile = engine_profile::span("interpolation", "capture_tick_begin");
    world.resource_scope(|world, mut state: Mut<InterpolationState>| {
        state.epoch = state.epoch.wrapping_add(1);
        if state.epoch == 0 {
            state.epoch = 1;
            for interval in &mut state.intervals {
                interval.epoch = 0;
            }
        }
        state.active = true;
        state.revision += 1;
        let hierarchy = world
            .get_resource::<HierarchyScratch>()
            .map(|hierarchy| hierarchy.revision());
        state.gpu.begin_tick(hierarchy);
        state.counters(&mut world.resource_mut::<PresentationState>());
    });
}
pub(crate) fn end_tick(world: &mut World) {
    let _profile = engine_profile::span("interpolation", "capture_tick_end");
    world.resource_scope(|world, mut state: Mut<InterpolationState>| {
        state.observe_invalid();
        state.active = false;
        state.revision += 1;
        state.counters(&mut world.resource_mut::<PresentationState>());
    });
}

pub(crate) fn prepare(world: &mut World) {
    let _profile = engine_profile::span("interpolation", "prepare_presented_transforms");
    world.resource_scope(|world, mut interpolation: Mut<InterpolationState>| {
        let count = interpolation.prepare(world);
        let frame = world.get_resource::<FrameTime>().map(|time| time.frame);
        let tick = world.get_resource::<FixedTime>().map(|time| time.tick);
        let mut state = world.resource_mut::<PresentationState>();
        state.mode = PresentationMode::Current;
        state.frame = frame;
        state.tick = tick;
        state.presented_nodes = count;
        state.gpu_pending = false;
        interpolation.counters(&mut state);
    });
}

pub(crate) fn prepare_gpu(world: &mut World, alpha: f32) {
    let _profile = engine_profile::span("interpolation", "prepare_gpu_presentation");
    world.resource_scope(|world, mut interpolation: Mut<InterpolationState>| {
        interpolation.observe_invalid();
        world.resource_scope(|_world, mut hierarchy: Mut<HierarchyScratch>| {
            interpolation.gather_gpu(&mut hierarchy, alpha);
        });
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

pub(crate) fn present_gpu_fraction(world: &mut World, alpha: f32, hold: bool) {
    let _profile = engine_profile::span("interpolation", "present_gpu_fraction");
    world.resource_scope(|world, mut interpolation: Mut<InterpolationState>| {
        interpolation.gpu.set_alpha(alpha, hold);
        let frame = world.get_resource::<FrameTime>().map(|time| time.frame);
        let mut state = world.resource_mut::<PresentationState>();
        state.mode = PresentationMode::Interpolated { alpha };
        state.frame = frame;
        state.gpu_pending = true;
    });
}

pub fn reset_interpolation(world: &mut World, root: Entity) -> usize {
    reset_roots(world, &[root])
}

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
