use crate::collections::HashMap;
use crate::interpolation::{InterpolationState, Observer};
use crate::ticks::{ChangeWindow, STAMP_BLOCK, ScanClock, consume_through};
use crate::{InterpolateTransform, NodePose, Transform, entity_indices::EntityIndices};
use bevy_ecs::change_detection::ContiguousRef;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemState;
use engine_settings::{Error, ErrorCode, Expected};
use glam::{Mat4, Vec4};
use std::convert::Infallible;

mod global;
#[cfg(debug_assertions)]
mod invariants;
pub(crate) mod tracking;
mod validation;
mod views;
pub use global::GlobalTransform;
pub(crate) use global::ValueIds;
use validation::Certificate;

pub const MAX_HIERARCHY_DEPTH: usize = 64;
pub const MAX_TRANSFORM_NODES: usize = u32::MAX as usize - 1;

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformNodeLimit(pub usize);

impl Default for TransformNodeLimit {
    fn default() -> Self {
        Self(MAX_TRANSFORM_NODES)
    }
}

fn node_limit(world: &World) -> usize {
    world
        .get_resource::<TransformNodeLimit>()
        .map_or(MAX_TRANSFORM_NODES, |limit| {
            limit.0.min(MAX_TRANSFORM_NODES)
        })
}
pub const MAX_NORMAL_INVERSE_NORM: f32 = 10_000.0;

#[derive(Component, Debug, Clone, Copy)]
#[component(immutable,
    on_insert = tracking::topology_changed,
    on_discard = tracking::topology_changed,
    on_remove = tracking::topology_changed)]
pub struct Parent(Entity);

impl Parent {
    pub fn parent(&self) -> Entity {
        self.0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HierarchyReport {
    pub transforms: usize,
    pub invalid_transforms: usize,
    pub overflow_transforms: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Broken {
    MissingEntity(Entity),
    MissingTransform(Entity),
    Cycle(Entity),
    DepthExceeded,
}

impl Broken {
    fn error(self) -> Error {
        match self {
            Self::MissingEntity(entity) => Error::new(ErrorCode::UnknownEntity, "$.entity")
                .expected(Expected::Live)
                .got(entity.to_bits()),
            Self::MissingTransform(entity) => {
                Error::new(ErrorCode::MissingComponent, "$.entity.transform")
                    .expected(Expected::Present)
                    .got(entity.to_bits())
            }
            Self::Cycle(entity) => Error::new(ErrorCode::HierarchyCycle, "$.parent")
                .expected(Expected::Acyclic)
                .got(entity.to_bits()),
            Self::DepthExceeded => Error::new(ErrorCode::HierarchyDepth, "$.parent")
                .at_most(MAX_HIERARCHY_DEPTH as u64),
        }
    }
}

fn require_transform(world: &World, entity: Entity) -> Result<(), Error> {
    if !world.entities().contains(entity) {
        return Err(Broken::MissingEntity(entity).error());
    }
    if world.get::<Transform>(entity).is_none() {
        return Err(Broken::MissingTransform(entity).error());
    }
    Ok(())
}

pub fn set_parent(world: &mut World, child: Entity, parent: Option<Entity>) -> Result<(), Error> {
    set_parents(world, &[(child, parent)])
}

struct RelationshipNode {
    entity: Entity,
    parent: Option<Entity>,
    state: State,
    depth: Result<usize, Broken>,
    affected: bool,
}

pub fn set_parents(world: &mut World, changes: &[(Entity, Option<Entity>)]) -> Result<(), Error> {
    let limit = node_limit(world);
    let over = |count: usize| {
        Error::new(ErrorCode::HierarchyDepth, "$")
            .at_most(limit as u64)
            .got(count)
    };
    if changes.len() > limit {
        return Err(over(changes.len()));
    }
    let mut query = world.query::<(Entity, &Transform, Option<&Parent>)>();
    let count = query.iter(world).len();
    if count > limit {
        return Err(over(count));
    }
    if changes.is_empty() {
        return Ok(());
    }
    let mut replacements = HashMap::with_capacity_and_hasher(changes.len(), Default::default());
    for &(child, parent) in changes {
        require_transform(world, child)?;
        if let Some(parent) = parent {
            require_transform(world, parent)?;
        }
        if replacements.insert(child, parent).is_some() {
            return Err(Error::new(ErrorCode::HierarchyCycle, "$.child")
                .expected(Expected::Unique)
                .got(child.to_bits()));
        }
    }
    let mut nodes = Vec::with_capacity(count);
    let mut indices = HashMap::with_capacity_and_hasher(count, Default::default());
    for (entity, _, parent) in query.iter(world) {
        indices.insert(entity, nodes.len());
        nodes.push(RelationshipNode {
            entity,
            parent: replacements
                .get(&entity)
                .copied()
                .unwrap_or_else(|| parent.map(Parent::parent)),
            state: State::Pending,
            depth: Ok(0),
            affected: replacements.contains_key(&entity),
        });
    }
    complete_parents_first(
        &mut nodes,
        &mut Vec::new(),
        |node| node.parent.and_then(|parent| indices.get(&parent).copied()),
        |nodes, index| {
            let (depth, inherited) = match nodes[index].parent {
                None => (Ok(0), false),
                Some(parent) => match indices.get(&parent) {
                    None => (
                        Err(if world.entities().contains(parent) {
                            Broken::MissingTransform(parent)
                        } else {
                            Broken::MissingEntity(parent)
                        }),
                        false,
                    ),
                    Some(parent) => {
                        let parent = &nodes[*parent];
                        let depth = if parent.state == State::Complete {
                            parent.depth.and_then(|depth| {
                                if depth < MAX_HIERARCHY_DEPTH {
                                    Ok(depth + 1)
                                } else {
                                    Err(Broken::DepthExceeded)
                                }
                            })
                        } else {
                            Err(Broken::Cycle(parent.entity))
                        };
                        (depth, parent.affected)
                    }
                },
            };
            let node = &mut nodes[index];
            node.depth = depth;
            node.affected |= inherited;
            if node.affected {
                depth?;
            }
            Ok(())
        },
    )
    .map_err(Broken::error)?;
    for &(child, parent) in changes {
        let mut entity = world.entity_mut(child);
        if let Some(parent) = parent {
            entity.insert(Parent(parent));
        } else {
            entity.remove::<Parent>();
        }
    }
    crate::interpolation::reset_changed_parents(world, changes);
    Ok(())
}

#[inline]
pub fn valid_global_matrix(matrix: Mat4) -> bool {
    validation::valid_global_matrix(matrix)
}

#[inline(always)]
pub(crate) fn same_pose_bits(left: &NodePose, right: &NodePose) -> bool {
    let lanes = |pose: &NodePose| (Vec4::from_slice(&pose[..4]), Vec4::from_slice(&pose[4..8]));
    let ((a, b), (c, d)) = (lanes(left), lanes(right));
    let tail = left[8].to_bits() == right[8].to_bits() && left[9].to_bits() == right[9].to_bits();
    let signs = (a.is_negative_bitmask() ^ c.is_negative_bitmask())
        | (b.is_negative_bitmask() ^ d.is_negative_bitmask());
    if (a.cmpeq(c) & b.cmpeq(d)).all() && signs == 0 {
        return tail;
    }
    if !(a.is_nan() || b.is_nan() || c.is_nan() || d.is_nan()) {
        return false;
    }
    std::hint::cold_path();
    let bits = |value: Vec4| value.to_array().map(f32::to_bits);
    tail && bits(a) == bits(c) && bits(b) == bits(d)
}

#[inline]
pub(crate) fn same_local_bits(left: &Transform, right: &Transform) -> bool {
    let bits = |pose: &Transform| {
        [
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            pose.rotation.x,
            pose.rotation.y,
            pose.rotation.z,
            pose.rotation.w,
            pose.scale.x,
            pose.scale.y,
            pose.scale.z,
        ]
        .map(f32::to_bits)
    };
    let (left, right) = (bits(left), bits(right));
    left.iter()
        .zip(&right)
        .fold(0, |difference, (left, right)| difference | (left ^ right))
        == 0
}

type TransformQuery = QueryState<(
    Entity,
    Ref<'static, Transform>,
    Option<&'static Parent>,
    Option<Ref<'static, InterpolateTransform>>,
)>;
type ChangedTransforms = SystemState<Query<'static, 'static, (Entity, Ref<'static, Transform>)>>;

#[derive(Debug, Clone, Copy)]
pub struct TransformIndex(pub(crate) usize);
impl Default for TransformIndex {
    fn default() -> Self {
        Self(usize::MAX)
    }
}

impl TransformIndex {
    pub fn index(&self) -> usize {
        self.0
    }
}

const _: () = assert!(MAX_TRANSFORM_NODES < u32::MAX as usize, "indices_fit_u32");
const _: () = assert!(MAX_HIERARCHY_DEPTH < u8::MAX as usize, "depths_fit_u8");
const NO_NODE: u32 = u32::MAX;

pub(crate) struct Source {
    pub(crate) entity: Entity,
    pub(crate) parent: Option<Entity>,
    pub(crate) transform_added: u32,
    pub(crate) marker_added: Option<u32>,
}

#[derive(Clone, Copy)]
pub(crate) struct Link {
    parent: u32,
    first_child: u32,
    next_sibling: u32,
    depth: u8,
    state: State,
    topology_valid: bool,
    global: Global,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Global {
    Invalid,
    Stored,
    Deferred,
}

impl Link {
    const DETACHED: Self = Self {
        parent: NO_NODE,
        first_child: NO_NODE,
        next_sibling: NO_NODE,
        depth: 0,
        state: State::Pending,
        topology_valid: false,
        global: Global::Invalid,
    };

    #[inline]
    fn index(value: u32) -> Option<usize> {
        (value != NO_NODE).then_some(value as usize)
    }

    #[inline]
    pub(crate) fn parent(&self) -> Option<usize> {
        Self::index(self.parent)
    }

    #[inline]
    fn first_child(&self) -> Option<usize> {
        Self::index(self.first_child)
    }

    #[inline]
    fn next_sibling(&self) -> Option<usize> {
        Self::index(self.next_sibling)
    }

    #[inline]
    pub(crate) fn topology_valid(&self) -> bool {
        self.topology_valid
    }

    #[inline]
    pub(crate) fn depth(&self) -> u8 {
        self.depth
    }

    #[inline]
    pub(crate) fn has_global(&self) -> bool {
        self.global != Global::Invalid
    }

    #[inline]
    pub(crate) fn has_children(&self) -> bool {
        self.first_child != NO_NODE
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum State {
    Pending,
    Visiting,
    Complete,
}

trait ParentLinked {
    fn state(&mut self) -> &mut State;
}

impl ParentLinked for RelationshipNode {
    fn state(&mut self) -> &mut State {
        &mut self.state
    }
}

impl ParentLinked for Link {
    fn state(&mut self) -> &mut State {
        &mut self.state
    }
}

fn complete_parents_first<N: ParentLinked, E>(
    nodes: &mut [N],
    path: &mut Vec<usize>,
    parent: impl Fn(&N) -> Option<usize>,
    mut complete: impl FnMut(&mut [N], usize) -> Result<(), E>,
) -> Result<(), E> {
    for start in 0..nodes.len() {
        if *nodes[start].state() == State::Complete {
            continue;
        }
        path.clear();
        let mut cursor = Some(start);
        while let Some(index) = cursor {
            let node = &mut nodes[index];
            if *node.state() != State::Pending {
                break;
            }
            *node.state() = State::Visiting;
            path.push(index);
            cursor = parent(node);
        }
        while let Some(index) = path.pop() {
            complete(nodes, index)?;
            *nodes[index].state() = State::Complete;
        }
    }
    Ok(())
}

#[derive(Resource)]
pub struct GlobalTransforms {
    query: TransformQuery,
    changed_query: Option<ChangedTransforms>,
    entities: Vec<Entity>,
    sources: Vec<Source>,
    locals: Vec<NodePose>,
    local_revisions: Vec<u64>,
    history_hints: Vec<u32>,
    links: Vec<Link>,
    globals: Vec<Mat4>,
    certificates: Vec<Certificate>,
    revisions: Vec<u64>,
    indices: EntityIndices,
    order: Vec<usize>,
    validity_epoch: u64,
    path: Vec<usize>,
    dirty: Dirty,
    changes: ChangeLog,
    scans: ScanClock,
    incomplete: bool,
    limit: usize,
    report: HierarchyReport,
    topology_revision: u64,
    revision: u64,
}

pub(crate) type HierarchyScratch = GlobalTransforms;

impl FromWorld for GlobalTransforms {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: world.query(),
            changed_query: Some(SystemState::new(world)),
            entities: Vec::new(),
            sources: Vec::new(),
            locals: Vec::new(),
            local_revisions: Vec::new(),
            history_hints: Vec::new(),
            links: Vec::new(),
            globals: Vec::new(),
            certificates: Vec::new(),
            revisions: Vec::new(),
            indices: EntityIndices::default(),
            order: Vec::new(),
            validity_epoch: 0,
            path: Vec::new(),
            dirty: Dirty::default(),
            changes: ChangeLog::default(),
            scans: ScanClock::default(),
            incomplete: true,
            limit: node_limit(world),
            report: HierarchyReport::default(),
            topology_revision: 0,
            revision: 0,
        }
    }
}

#[derive(Default)]
struct PropagationWork {
    evaluated: usize,
    deferred: usize,
    written: usize,
    value_ids: ValueIds,
}

#[derive(Default)]
struct Dirty {
    flags: Vec<bool>,
    seeds: Vec<usize>,
    count: usize,
}

impl Dirty {
    fn clear(&mut self) {
        self.seeds.clear();
        self.count = 0;
    }

    #[inline]
    fn mark(&mut self, index: usize, threshold: usize) {
        self.flags[index] = true;
        if self.count < threshold {
            self.seeds.push(index);
        }
        self.count += 1;
    }
}

pub(crate) struct ChangeLog {
    nodes: Vec<u32>,
    marks: Vec<u32>,
    generation: u32,
    limit: usize,
    overflow: bool,
}

impl Default for ChangeLog {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            marks: Vec::new(),
            generation: 1,
            limit: 0,
            overflow: true,
        }
    }
}

impl ChangeLog {
    fn reset(&mut self, nodes: usize) {
        self.nodes.clear();
        self.marks.clear();
        self.marks.resize(nodes, 0);
        self.generation = 1;
        self.limit = nodes / 4;
        self.overflow = true;
    }

    #[inline(always)]
    fn record(&mut self, index: usize) {
        if self.overflow {
            return;
        }
        let mark = &mut self.marks[index];
        if *mark != self.generation {
            *mark = self.generation;
            if self.nodes.len() == self.limit {
                self.overflow = true;
            } else {
                self.nodes.push(index as u32);
            }
        }
    }

    pub(crate) fn take(&mut self) -> Option<Vec<u32>> {
        (!self.overflow).then(|| std::mem::take(&mut self.nodes))
    }

    pub(crate) fn restore(&mut self, nodes: Option<Vec<u32>>) {
        if let Some(nodes) = nodes {
            self.nodes = nodes;
        }
        self.nodes.clear();
        self.overflow = false;
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill(0);
            self.generation = 1;
        }
    }
}

struct Scan {
    topology_changed: bool,
    observed: usize,
}

struct Nodes<'a> {
    locals: &'a mut [NodePose],
    history_hints: &'a mut [u32],
    links: &'a mut [Link],
    globals: &'a mut [Mat4],
    certificates: &'a mut [Certificate],
    revisions: &'a mut [u64],
    flags: &'a mut [bool],
    invalid_transforms: &'a mut usize,
    validity_epoch: &'a mut u64,
    changes: &'a mut ChangeLog,
}

#[cold]
#[inline(never)]
fn find_index(indices: &EntityIndices, entity: Entity) -> Option<usize> {
    indices.get(&entity).copied()
}

#[cfg_attr(not(feature = "attribution"), inline)]
#[cfg_attr(feature = "attribution", inline(never))]
fn deferred_global(links: &[Link], globals: &[Mat4], locals: &[NodePose], index: usize) -> Mat4 {
    let parent = links[index].parent().map_or(Mat4::IDENTITY, |parent| {
        debug_assert!(links[parent].global == Global::Stored, "parent_stored");
        globals[parent]
    });
    parent * Transform::from_pose(&locals[index]).matrix()
}

impl Nodes<'_> {
    #[inline(always)]
    fn trimmed(self, len: usize) -> Self {
        let Self {
            locals,
            history_hints,
            links,
            globals,
            certificates,
            revisions,
            flags,
            invalid_transforms,
            validity_epoch,
            changes,
        } = self;
        Self {
            locals: &mut locals[..len],
            history_hints: &mut history_hints[..len],
            links: &mut links[..len],
            globals: &mut globals[..len],
            certificates: &mut certificates[..len],
            revisions: &mut revisions[..len],
            flags: &mut flags[..len],
            invalid_transforms,
            validity_epoch,
            changes,
        }
    }

    #[inline(always)]
    fn global(&self, index: usize) -> Option<Mat4> {
        match self.links[index].global {
            Global::Stored => Some(self.globals[index]),
            Global::Deferred => Some(deferred_global(
                self.links,
                self.globals,
                self.locals,
                index,
            )),
            Global::Invalid => None,
        }
    }

    #[cfg_attr(not(feature = "attribution"), inline(always))]
    #[cfg_attr(feature = "attribution", inline(never))]
    fn derived_global(&self, index: usize, local: &Transform) -> Option<Mat4> {
        let link = &self.links[index];
        if !link.topology_valid {
            return None;
        }
        let parent = match link.parent() {
            Some(parent) => self.global(parent)?,
            None => Mat4::IDENTITY,
        };
        let matrix = parent * local_matrix(local)?;
        valid_global_matrix(matrix).then_some(matrix)
    }

    #[cfg_attr(not(feature = "attribution"), inline(always))]
    #[cfg_attr(feature = "attribution", inline(never))]
    fn certified(&self, index: usize, local: &Transform) -> bool {
        let link = &self.links[index];
        if !link.topology_valid || link.has_children() {
            return false;
        }
        let certificate = match link.parent() {
            Some(parent) if self.links[parent].global == Global::Stored => {
                &self.certificates[parent]
            }
            Some(_) => return false,
            None => &Certificate::ROOT,
        };
        certificate.admits(local)
    }

    #[inline(always)]
    fn settle(
        &mut self,
        index: usize,
        local: &Transform,
        work: &mut PropagationWork,
        observer: &mut Option<Observer<'_>>,
    ) -> bool {
        if self.certified(index, local) {
            self.defer(index, work, observer);
            return true;
        }
        let next = self.derived_global(index, local);
        self.commit(index, next, work, observer)
    }

    #[inline(always)]
    fn update(
        &mut self,
        index: usize,
        work: &mut PropagationWork,
        observer: &mut Option<Observer<'_>>,
    ) -> bool {
        let local = Transform::from_pose(&self.locals[index]);
        self.settle(index, &local, work, observer)
    }

    #[cfg_attr(not(feature = "attribution"), inline(always))]
    #[cfg_attr(feature = "attribution", inline(never))]
    fn defer(
        &mut self,
        index: usize,
        work: &mut PropagationWork,
        observer: &mut Option<Observer<'_>>,
    ) {
        let link = &mut self.links[index];
        let revision = &mut self.revisions[index];
        let was_invalid = link.global == Global::Invalid;
        if was_invalid {
            *self.invalid_transforms -= 1;
            *self.validity_epoch += 1;
        }
        if link.global != Global::Deferred {
            link.global = Global::Deferred;
        }
        if (was_invalid || *revision == 0)
            && let Some(observer) = observer
        {
            observer.observe_validity(self.history_hints[index], index, &self.locals[index], true);
        }
        self.flags[index] = false;
        work.evaluated += 1;
        work.deferred += 1;
        *revision = work.value_ids.next();
        work.written += 1;
        self.changes.record(index);
    }

    #[cfg_attr(not(feature = "attribution"), inline(always))]
    #[cfg_attr(feature = "attribution", inline(never))]
    fn commit(
        &mut self,
        index: usize,
        next: Option<Mat4>,
        work: &mut PropagationWork,
        observer: &mut Option<Observer<'_>>,
    ) -> bool {
        let link = &mut self.links[index];
        let global = &mut self.globals[index];
        let revision = &mut self.revisions[index];
        let valid = next.is_some();
        let before = link.global;
        let validity_changed = (before != Global::Invalid) != valid;
        let changed = validity_changed
            || before == Global::Deferred
            || next.is_some_and(|matrix| !same_finite_matrix_bits(*global, matrix));
        if validity_changed {
            if valid {
                *self.invalid_transforms -= 1;
            } else {
                *self.invalid_transforms += 1;
            }
            *self.validity_epoch += 1;
        }
        let after = if valid {
            Global::Stored
        } else {
            Global::Invalid
        };
        if before != after {
            link.global = after;
        }
        if (validity_changed || *revision == 0)
            && let Some(observer) = observer
        {
            observer.observe_validity(self.history_hints[index], index, &self.locals[index], valid);
        }
        let publish = changed || *revision == 0;
        if let Some(matrix) = next {
            if changed {
                *global = matrix;
            }
            if publish && link.has_children() {
                self.certificates[index] = Certificate::of(matrix);
            }
        }
        self.flags[index] = false;
        work.evaluated += 1;
        if publish {
            *revision = work.value_ids.next();
            work.written += 1;
            self.changes.record(index);
        }
        changed
    }
}

impl GlobalTransforms {
    pub fn get(&self, entity: Entity) -> Option<GlobalTransform> {
        let index = *self.indices.get(&entity)?;
        Some(self.value_at(index))
    }

    #[inline]
    pub fn get_indexed(
        &self,
        entity: Entity,
        hint: &mut TransformIndex,
    ) -> Option<GlobalTransform> {
        self.get_indexed_with_revision(entity, hint)
            .map(|(value, _)| value)
    }

    #[inline]
    pub fn get_indexed_with_revision(
        &self,
        entity: Entity,
        hint: &mut TransformIndex,
    ) -> Option<(GlobalTransform, u64)> {
        let index = if self
            .sources
            .get(hint.0)
            .is_some_and(|source| source.entity == entity)
        {
            hint.0
        } else {
            *self.indices.get(&entity)?
        };
        hint.0 = index;
        Some((self.value_at(index), self.revisions[index]))
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
    pub(crate) fn valid_count(&self) -> usize {
        self.report.transforms - self.report.invalid_transforms
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Entity, GlobalTransform)> + '_ {
        self.sources
            .iter()
            .enumerate()
            .map(|(index, source)| (source.entity, self.value_at(index)))
    }

    fn value_at(&self, index: usize) -> GlobalTransform {
        GlobalTransform::read_value(self.global(index), self.revisions[index])
    }

    #[inline]
    pub(crate) fn revision_at(&self, index: usize) -> u64 {
        self.revisions[index]
    }

    #[inline]
    pub(crate) fn global(&self, index: usize) -> Option<Mat4> {
        match self.links[index].global {
            Global::Stored => Some(self.globals[index]),
            Global::Deferred => Some(deferred_global(
                &self.links,
                &self.globals,
                &self.locals,
                index,
            )),
            Global::Invalid => None,
        }
    }

    pub(crate) fn children(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(self.links[index].first_child(), |&child| {
            self.links[child].next_sibling()
        })
    }

    fn dense_threshold(&self) -> usize {
        self.links.len().div_ceil(2)
    }

    fn scan_changed(
        &mut self,
        world: &World,
        work: &mut PropagationWork,
        interpolation: &mut Option<&mut InterpolationState>,
    ) -> Scan {
        let _profile = engine_profile::span("hierarchy", "change_tick_scan");
        engine_profile::counter("hierarchy.full_scan", 0.0);
        engine_profile::counter("hierarchy.scanned_nodes", 0.0);
        engine_profile::counter(
            "hierarchy.metadata_nodes",
            self.query.iter(world).len() as f64,
        );
        self.dirty.clear();
        let mut changed_query = self.changed_query.take().expect("scan_state_present");
        let scan = self.scan_changed_rows(&mut changed_query, world, work, interpolation);
        self.changed_query = Some(changed_query);
        scan
    }

    #[expect(
        clippy::too_many_lines,
        reason = "longer than the line ceiling; it may only shrink"
    )]
    fn scan_changed_rows(
        &mut self,
        changed_query: &mut ChangedTransforms,
        world: &World,
        work: &mut PropagationWork,
        interpolation: &mut Option<&mut InterpolationState>,
    ) -> Scan {
        let threshold = self.dense_threshold();
        let next_revision = self.revision.wrapping_add(1);
        let mut checked = 0;
        let mut observed = 0;
        let mut early = 0;
        let mut next_index = 0;
        let Self {
            entities: source_entities,
            sources,
            indices,
            locals,
            local_revisions,
            history_hints,
            links,
            globals,
            certificates,
            revisions,
            dirty,
            report,
            changes,
            validity_epoch,
            ..
        } = self;
        let Dirty {
            flags,
            seeds,
            count: dirty_count,
        } = dirty;
        let len = source_entities.len();
        let (source_entities, sources, local_revisions) = (
            &source_entities[..len],
            &mut sources[..len],
            &mut local_revisions[..len],
        );
        let mut nodes = Nodes {
            locals,
            history_hints,
            links,
            globals,
            certificates,
            revisions,
            flags,
            invalid_transforms: &mut report.invalid_transforms,
            validity_epoch,
            changes,
        }
        .trimmed(len);
        let mut observer = interpolation
            .as_deref_mut()
            .map(InterpolationState::observer);
        let query = changed_query.get(world).expect("transform_query_read_only");
        for (entities, transforms) in query.contiguous_iter().expect("transform_dense_storage") {
            let (transforms, ticks) = ContiguousRef::split(transforms);
            let window = ChangeWindow::new(ticks.last_run(), ticks.this_run());
            let (changed, added) = (ticks.changed(), ticks.added());
            if !window.any(changed) {
                continue;
            }
            for block in (0..changed.len()).step_by(STAMP_BLOCK) {
                let end = (block + STAMP_BLOCK).min(changed.len());
                if !window.any_in_block(&changed[block..end]) {
                    continue;
                }
                for row in block..end {
                    if !window.contains(changed[row]) {
                        continue;
                    }
                    checked += 1;
                    let entity = entities[row];
                    let transform = &transforms[row];
                    let index = if source_entities.get(next_index) == Some(&entity) {
                        next_index
                    } else if let Some(index) = find_index(indices, entity) {
                        index
                    } else {
                        return Scan {
                            topology_changed: true,
                            observed: 0,
                        };
                    };
                    next_index = index + 1;
                    let local = &mut nodes.locals[index];
                    let pose = transform.to_pose();
                    let local_changed = !same_pose_bits(local, &pose);
                    let transform_added = added[row];
                    let identity_changed = window.contains(transform_added)
                        && sources[index].transform_added != transform_added.get();
                    if local_changed || identity_changed {
                        observed += 1;
                        if identity_changed {
                            sources[index].transform_added = transform_added.get();
                        }
                        if let Some(observer) = observer.as_mut() {
                            let hint = &mut nodes.history_hints[index];
                            let motion = !identity_changed
                                && observer.observe_motion(*hint, index, local, &pose);
                            if !motion
                                && (*hint != u32::MAX || sources[index].marker_added.is_some())
                            {
                                observer.observe_source(
                                    hint,
                                    index,
                                    &sources[index],
                                    (local, &transform.to_pose()),
                                    local_changed,
                                    identity_changed,
                                );
                            }
                        }
                        *local = pose;
                    }
                    if !local_changed {
                        continue;
                    }
                    local_revisions[index] = next_revision;
                    nodes.changes.record(index);
                    let link = &nodes.links[index];
                    if !link.has_children()
                        && link.parent().is_none_or(|parent| !nodes.flags[parent])
                    {
                        nodes.settle(index, transform, work, &mut observer);
                        early += 1;
                    } else {
                        nodes.flags[index] = true;
                        if *dirty_count < threshold {
                            seeds.push(index);
                        }
                        *dirty_count += 1;
                    }
                }
            }
        }
        engine_profile::counter("hierarchy.validated_nodes", checked as f64);
        engine_profile::counter("hierarchy.scan_evaluated_nodes", early as f64);
        Scan {
            topology_changed: false,
            observed,
        }
    }

    fn scan_full(
        &mut self,
        world: &World,
        interpolation: &mut Option<&mut InterpolationState>,
    ) -> Scan {
        let _profile = engine_profile::span("hierarchy", "change_tick_scan");
        engine_profile::counter("hierarchy.full_scan", 1.0);
        self.dirty.clear();
        self.changes.overflow = true;
        let threshold = self.dense_threshold();
        let mut count = 0;
        let mut metadata = 0;
        let mut checked = 0;
        let mut observed = 0;
        for (entity, transform, parent, marker) in self.query.iter(world) {
            metadata += 1;
            if count < self.limit {
                let Some(source) = self.sources.get_mut(count) else {
                    return Scan {
                        topology_changed: true,
                        observed: 0,
                    };
                };
                if source.entity != entity || source.parent != parent.map(Parent::parent) {
                    return Scan {
                        topology_changed: true,
                        observed: 0,
                    };
                }
                let local = &mut self.locals[count];
                let pose = transform.to_pose();
                let local_changed = !same_pose_bits(local, &pose);
                let marker_added = marker.as_ref().map(|marker| marker.added().get());
                let transform_added = transform.added().get();
                let identity_changed = source.transform_added != transform_added
                    || source.marker_added != marker_added;
                if local_changed || identity_changed {
                    observed += 1;
                    source.transform_added = transform_added;
                    source.marker_added = marker_added;
                    if let Some(state) = interpolation.as_deref_mut() {
                        state.observe_source(
                            &mut self.history_hints[count],
                            count,
                            source,
                            (local, &pose),
                            local_changed,
                            identity_changed,
                        );
                    }
                    *local = pose;
                    if local_changed {
                        self.local_revisions[count] = self.revision.wrapping_add(1);
                    }
                }
                checked += 1;
                if local_changed {
                    self.dirty.mark(count, threshold);
                }
            }
            count += 1;
        }
        engine_profile::counter("hierarchy.metadata_nodes", metadata as f64);
        engine_profile::counter("hierarchy.validated_nodes", checked as f64);
        engine_profile::counter("hierarchy.scanned_nodes", count as f64);
        Scan {
            topology_changed: count != self.report.transforms
                || self.sources.len() != count.min(self.limit),
            observed,
        }
    }

    fn rebuild_topology(
        &mut self,
        world: &mut World,
        work: &mut PropagationWork,
        interpolation: &mut Option<&mut InterpolationState>,
    ) {
        let _profile = engine_profile::span("hierarchy", "topology_rebuild");
        if let Some(state) = interpolation.as_deref_mut() {
            state.reconcile(world, self);
        }
        self.entities.clear();
        self.sources.clear();
        self.locals.clear();
        self.local_revisions.clear();
        self.history_hints.clear();
        self.links.clear();
        self.globals.clear();
        self.certificates.clear();
        self.revisions.clear();
        self.indices.clear();
        self.order.clear();
        self.path.clear();
        self.dirty.clear();
        let admitted = self.query.iter(world).len().min(self.limit);
        self.entities.reserve_exact(admitted);
        self.sources.reserve_exact(admitted);
        self.locals.reserve_exact(admitted);
        self.local_revisions.reserve_exact(admitted);
        self.history_hints.reserve_exact(admitted);
        self.links.reserve_exact(admitted);
        self.globals.reserve_exact(admitted);
        self.certificates.reserve_exact(admitted);
        self.revisions.reserve_exact(admitted);
        self.order.reserve_exact(admitted);
        self.path.reserve_exact(admitted);
        self.dirty.flags.clear();
        self.dirty.flags.reserve_exact(admitted);
        self.dirty.seeds.reserve_exact(admitted.div_ceil(2));
        let mut report = HierarchyReport::default();
        for (entity, transform, parent, marker) in self.query.iter(world) {
            report.transforms += 1;
            if self.links.len() == self.limit {
                report.overflow_transforms += 1;
                continue;
            }
            let source = Source {
                entity,
                parent: parent.map(Parent::parent),
                transform_added: transform.added().get(),
                marker_added: marker.as_ref().map(|marker| marker.added().get()),
            };
            let mut history_hint = u32::MAX;
            if let Some(state) = interpolation.as_deref_mut() {
                state.observe_source(
                    &mut history_hint,
                    self.sources.len(),
                    &source,
                    (&transform.to_pose(), &transform.to_pose()),
                    false,
                    true,
                );
            }
            self.entities.push(entity);
            self.sources.push(source);
            self.locals.push(transform.to_pose());
            self.local_revisions.push(self.revision.wrapping_add(1));
            self.history_hints.push(history_hint);
            self.links.push(Link::DETACHED);
            self.globals.push(Mat4::IDENTITY);
            self.certificates.push(Certificate::NONE);
            self.revisions.push(0);
        }
        self.indices.clear();
        self.indices.reserve(admitted);
        for (index, source) in self.sources.iter().enumerate() {
            self.indices.insert(source.entity, index);
        }
        for (link, source) in self.links.iter_mut().zip(&self.sources) {
            link.parent = source
                .parent
                .and_then(|parent| self.indices.get(&parent))
                .map_or(NO_NODE, |&index| index as u32);
        }
        let (sources, order) = (&self.sources, &mut self.order);
        let Ok(()) = complete_parents_first(
            &mut self.links,
            &mut self.path,
            Link::parent,
            |links, index| -> Result<(), Infallible> {
                let depth = match sources[index].parent {
                    None => Some(0),
                    Some(_) => links[index].parent().and_then(|parent| {
                        let parent = &links[parent];
                        (parent.state == State::Complete
                            && parent.topology_valid
                            && usize::from(parent.depth) < MAX_HIERARCHY_DEPTH)
                            .then_some(parent.depth + 1)
                    }),
                };
                let link = &mut links[index];
                link.topology_valid = depth.is_some();
                link.depth = depth.unwrap_or(0);
                order.push(index);
                Ok(())
            },
        );
        for index in 0..self.links.len() {
            if self.links[index].topology_valid
                && let Some(parent) = self.links[index].parent()
            {
                self.links[index].next_sibling = self.links[parent].first_child;
                self.links[parent].first_child = index as u32;
            }
        }
        self.dirty.flags.resize(self.links.len(), true);
        self.dirty.count = self.links.len();
        self.changes.reset(self.links.len());
        report.invalid_transforms = report.transforms;
        self.report = report;
        self.validity_epoch += 1;
        self.topology_revision = work.value_ids.next();
    }

    fn evaluate_dirty(
        &mut self,
        work: &mut PropagationWork,
        interpolation: &mut Option<&mut InterpolationState>,
    ) {
        let _profile = engine_profile::span("hierarchy", "evaluate_dirty");
        let dense = self.dirty.count >= self.dense_threshold();
        let Self {
            locals,
            history_hints,
            links,
            globals,
            certificates,
            revisions,
            order,
            path,
            dirty,
            report,
            changes,
            validity_epoch,
            ..
        } = self;
        let len = links.len();
        let mut nodes = Nodes {
            locals,
            history_hints,
            links,
            globals,
            certificates,
            revisions,
            flags: &mut dirty.flags,
            invalid_transforms: &mut report.invalid_transforms,
            validity_epoch,
            changes,
        }
        .trimmed(len);
        let mut observer = interpolation
            .as_deref_mut()
            .map(InterpolationState::observer);
        if dense {
            for &index in order.iter() {
                if nodes.flags[index] && nodes.update(index, work, &mut observer) {
                    let mut child = nodes.links[index].first_child();
                    while let Some(next) = child {
                        nodes.flags[next] = true;
                        child = nodes.links[next].next_sibling();
                    }
                }
            }
            return;
        }
        for &index in dirty.seeds.iter() {
            while nodes.flags[index] {
                let mut root = index;
                if nodes.links[index].topology_valid {
                    let mut parent = nodes.links[index].parent();
                    while let Some(ancestor) = parent {
                        if nodes.flags[ancestor] {
                            root = ancestor;
                        }
                        parent = nodes.links[ancestor].parent();
                    }
                }
                path.clear();
                path.push(root);
                while let Some(current) = path.pop() {
                    if nodes.update(current, work, &mut observer) {
                        let mut child = nodes.links[current].first_child();
                        while let Some(index) = child {
                            path.push(index);
                            child = nodes.links[index].next_sibling();
                        }
                    }
                }
            }
        }
    }

    fn propagate(
        &mut self,
        world: &mut World,
        mut interpolation: Option<&mut InterpolationState>,
    ) -> HierarchyReport {
        let limit = node_limit(world);
        if limit != self.limit {
            self.limit = limit;
            self.incomplete = true;
        }
        let interrupted = std::mem::replace(&mut self.incomplete, true);
        let scan_tick = world.read_change_tick();
        let aged = self.scans.aged(scan_tick);
        let structural =
            std::mem::take(&mut world.resource_mut::<tracking::Invalidation>().full_scan);
        let force = interrupted || aged || structural || self.report.transforms >= self.limit;
        let mut work = PropagationWork::default();
        let mut scan = if force {
            self.scan_full(world, &mut interpolation)
        } else {
            self.scan_changed(world, &mut work, &mut interpolation)
        };
        scan.topology_changed |= interrupted || structural;
        if scan.topology_changed {
            self.rebuild_topology(world, &mut work, &mut interpolation);
        }
        if self.dirty.count != 0 {
            self.evaluate_dirty(&mut work, &mut interpolation);
        }
        if scan.topology_changed || scan.observed != 0 || work.evaluated != 0 {
            self.revision = self.revision.wrapping_add(1);
        }
        engine_profile::counter(
            "hierarchy.topology_rebuilt",
            f64::from(scan.topology_changed),
        );
        engine_profile::counter("hierarchy.evaluated_nodes", work.evaluated as f64);
        engine_profile::counter("hierarchy.deferred_globals", work.deferred as f64);
        engine_profile::counter("hierarchy.written_globals", work.written as f64);
        let completed_tick = world.read_change_tick();
        self.scans.scanned(completed_tick);
        consume_through(
            self.changed_query.as_mut().expect("scan_state_restored"),
            completed_tick,
        );
        world.increment_change_tick();
        self.incomplete = false;
        #[cfg(debug_assertions)]
        self.check();
        self.report
    }
}

#[inline]
fn local_matrix(local: &Transform) -> Option<Mat4> {
    ((local.rotation.length_squared() - 1.0).abs() <= 1.0e-3).then(|| local.matrix())
}

#[inline(always)]
pub(crate) fn same_finite_matrix_bits(left: Mat4, right: Mat4) -> bool {
    debug_assert!(left.is_finite() && right.is_finite(), "matrices_finite");
    let equal = left.x_axis.cmpeq(right.x_axis)
        & left.y_axis.cmpeq(right.y_axis)
        & left.z_axis.cmpeq(right.z_axis)
        & left.w_axis.cmpeq(right.w_axis);
    let signs = (left.x_axis.is_negative_bitmask() ^ right.x_axis.is_negative_bitmask())
        | (left.y_axis.is_negative_bitmask() ^ right.y_axis.is_negative_bitmask())
        | (left.z_axis.is_negative_bitmask() ^ right.z_axis.is_negative_bitmask())
        | (left.w_axis.is_negative_bitmask() ^ right.w_axis.is_negative_bitmask());
    equal.all() && signs == 0
}

pub fn propagate_transforms(world: &mut World) -> HierarchyReport {
    let _profile = engine_profile::span("hierarchy", "propagate");
    tracking::init(world);
    world.flush();
    world.check_change_ticks();
    world.init_resource::<HierarchyScratch>();
    world.resource_scope(|world, mut scratch: Mut<HierarchyScratch>| {
        if world.contains_resource::<InterpolationState>() {
            world.resource_scope(|world, mut interpolation: Mut<InterpolationState>| {
                scratch.propagate(world, Some(&mut interpolation))
            })
        } else {
            scratch.propagate(world, None)
        }
    })
}
