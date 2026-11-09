use crate::collections::HashMap;
use crate::presentation::pose_rows;
use crate::ticks::{ChangeWindow, STAMP_BLOCK, ScanClock, consume_through};
use crate::{
    Cursor, Feed, FeedSettings, Mobility, NodePose, NodeWord, PoseRows, Since, Transform,
    entity_indices::EntityIndices,
};
use bevy_ecs::change_detection::ContiguousRef;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemState;
use engine_settings::{Error, ErrorCode, Expected};
use glam::{Mat4, Vec4};
use std::collections::BTreeSet;
use std::convert::Infallible;

mod global;
#[cfg(debug_assertions)]
mod invariants;
pub(crate) mod tracking;
mod validation;
mod views;
mod words;
pub use global::{GlobalTransform, Revision};
pub(crate) use global::ValueIds;

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
            depth: Ok(0),
            affected: replacements.contains_key(&entity),
        });
    }
    complete_parents_first(
        &mut nodes,
        &mut Vec::new(),
        &mut Vec::new(),
        |node| node.parent.and_then(|parent| indices.get(&parent).copied()),
        |nodes, states, index| {
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
                    Some(&parent) => {
                        let complete = states[parent] == State::Complete;
                        let parent = &nodes[parent];
                        let depth = if complete {
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
    if let Some(mut poses) = world.get_resource_mut::<crate::DrawnPoses>() {
        for &(child, _) in changes {
            poses.fresh(child);
        }
    }
    Ok(())
}

pub(crate) fn despawn_owned(world: &mut World, owned: &[Entity]) {
    let gone: BTreeSet<Entity> = owned.iter().copied().collect();
    let mut parented = world.query::<(Entity, &Parent)>();
    let guests: Vec<Entity> = parented
        .iter(world)
        .filter(|(child, parent)| gone.contains(&parent.parent()) && !gone.contains(child))
        .map(|(child, _)| child)
        .collect();
    for &guest in &guests {
        let pose = world
            .get_resource::<GlobalTransforms>()
            .and_then(|globals| globals.pose(guest));
        if let Some(pose) = pose {
            let (scale, rotation, translation) = pose.to_scale_rotation_translation();
            let _ = crate::edit_transform(world, guest, |transform| {
                *transform = Transform {
                    translation,
                    rotation,
                    scale,
                };
            });
        }
        if let Ok(mut held) = world.get_entity_mut(guest) {
            held.remove::<Parent>();
        }
    }
    crate::presentation::make_fresh(world, &guests);
    for &entity in owned {
        world.despawn(entity);
    }
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

type TransformQuery = QueryState<(Entity, Ref<'static, Transform>, Option<&'static Parent>)>;
type ChangedTransforms = SystemState<Query<'static, 'static, (Entity, Ref<'static, Transform>)>>;

const _: () = assert!(MAX_TRANSFORM_NODES < u32::MAX as usize, "indices_fit_u32");
const _: () = assert!(MAX_HIERARCHY_DEPTH < u8::MAX as usize, "depths_fit_u8");
const NO_NODE: u32 = u32::MAX;

pub(crate) struct Source {
    pub(crate) entity: Entity,
    pub(crate) parent: Option<Entity>,
    pub(crate) transform_added: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct Link {
    parent: u32,
    first_child: u32,
    next_sibling: u32,
    depth: u8,
    topology_valid: bool,
    valid: bool,
}

impl Link {
    const DETACHED: Self = Self {
        parent: NO_NODE,
        first_child: NO_NODE,
        next_sibling: NO_NODE,
        depth: 0,
        topology_valid: false,
        valid: false,
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
        self.valid
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

fn complete_parents_first<N, E>(
    nodes: &mut [N],
    states: &mut Vec<State>,
    path: &mut Vec<usize>,
    parent: impl Fn(&N) -> Option<usize>,
    mut complete: impl FnMut(&mut [N], &[State], usize) -> Result<(), E>,
) -> Result<(), E> {
    states.clear();
    states.resize(nodes.len(), State::Pending);
    for start in 0..nodes.len() {
        if states[start] == State::Complete {
            continue;
        }
        path.clear();
        let mut cursor = Some(start);
        while let Some(index) = cursor {
            if states[index] != State::Pending {
                break;
            }
            states[index] = State::Visiting;
            path.push(index);
            cursor = parent(&nodes[index]);
        }
        while let Some(index) = path.pop() {
            complete(nodes, states, index)?;
            states[index] = State::Complete;
        }
    }
    Ok(())
}

struct StaticMoves {
    moved: BTreeSet<Entity>,
    feed: Feed<Entity>,
}

impl StaticMoves {
    fn record(&mut self, entity: Entity) {
        self.moved.insert(entity);
        self.feed.push(entity);
    }
}

#[derive(Resource)]
pub struct GlobalTransforms {
    query: TransformQuery,
    changed_query: Option<ChangedTransforms>,
    entities: Vec<Entity>,
    sources: Vec<Source>,
    locals: Vec<NodePose>,
    local_revisions: Vec<u64>,
    links: Vec<Link>,
    globals: Vec<PoseRows>,
    revisions: Vec<Option<Revision>>,
    indices: EntityIndices,
    order: Vec<usize>,
    path: Vec<usize>,
    states: Vec<State>,
    dirty: Dirty,
    changes: Feed<u32>,
    touched: Vec<Entity>,
    declared: Vec<words::Declared>,
    words: Vec<NodeWord>,
    words_revision: u64,
    woken: Vec<Entity>,
    asleep: Vec<Entity>,
    static_moves: StaticMoves,
    scans: ScanClock,
    incomplete: bool,
    limit: usize,
    report: HierarchyReport,
    topology_revision: Revision,
    revision: u64,
}

pub(crate) type HierarchyScratch = GlobalTransforms;

impl FromWorld for GlobalTransforms {
    fn from_world(world: &mut World) -> Self {
        let feed = world
            .get_resource::<FeedSettings>()
            .map_or(FeedSettings::default().transforms, |feeds| feeds.transforms)
            as usize;
        Self {
            query: world.query(),
            changed_query: Some(SystemState::new(world)),
            entities: Vec::new(),
            sources: Vec::new(),
            locals: Vec::new(),
            local_revisions: Vec::new(),
            links: Vec::new(),
            globals: Vec::new(),
            revisions: Vec::new(),
            indices: EntityIndices::default(),
            order: Vec::new(),
            path: Vec::new(),
            states: Vec::new(),
            dirty: Dirty::default(),
            changes: Feed::new(feed),
            touched: Vec::new(),
            declared: Vec::new(),
            words: Vec::new(),
            words_revision: 0,
            woken: Vec::new(),
            asleep: Vec::new(),
            static_moves: StaticMoves {
                moved: BTreeSet::new(),
                feed: Feed::new(feed),
            },
            scans: ScanClock::default(),
            incomplete: true,
            limit: node_limit(world),
            report: HierarchyReport::default(),
            topology_revision: ValueIds::default().next(),
            revision: 0,
        }
    }
}

#[derive(Default)]
struct PropagationWork {
    evaluated: usize,
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

#[derive(Clone, Copy)]
enum ScanPlan {
    Changed,
    Full,
    Rebuild,
}

struct Scan {
    topology_changed: bool,
    observed: usize,
}

struct Nodes<'a> {
    locals: &'a mut [NodePose],
    links: &'a mut [Link],
    globals: &'a mut [PoseRows],
    revisions: &'a mut [Option<Revision>],
    flags: &'a mut [bool],
    words: &'a [NodeWord],
    invalid_transforms: &'a mut usize,
    changes: &'a mut Feed<u32>,
}

#[cold]
#[inline(never)]
fn find_index(indices: &EntityIndices, entity: Entity) -> Option<usize> {
    indices.get(&entity).copied()
}

#[inline(always)]
fn rows_matrix(rows: &PoseRows) -> Mat4 {
    Mat4::from_cols(
        Vec4::from_array(rows[0]),
        Vec4::from_array(rows[1]),
        Vec4::from_array(rows[2]),
        Vec4::W,
    )
    .transpose()
}

#[inline(always)]
fn same_rows_bits(left: &PoseRows, right: &PoseRows) -> bool {
    left.as_flattened()
        .iter()
        .zip(right.as_flattened())
        .fold(0, |difference, (left, right)| {
            difference | (left.to_bits() ^ right.to_bits())
        })
        == 0
}

impl Nodes<'_> {
    #[inline(always)]
    fn trimmed(self, len: usize) -> Self {
        let Self {
            locals,
            links,
            globals,
            revisions,
            flags,
            words,
            invalid_transforms,
            changes,
        } = self;
        Self {
            locals: &mut locals[..len],
            links: &mut links[..len],
            globals: &mut globals[..len],
            revisions: &mut revisions[..len],
            flags: &mut flags[..len],
            words: &words[..len],
            invalid_transforms,
            changes,
        }
    }

    #[inline(always)]
    fn asleep(&self, index: usize) -> bool {
        !self.words[index].active && self.revisions[index].is_some()
    }

    #[inline(always)]
    fn global(&self, index: usize) -> Option<Mat4> {
        self.links[index]
            .valid
            .then(|| rows_matrix(&self.globals[index]))
    }

    #[inline(always)]
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

    #[inline(always)]
    fn update(&mut self, index: usize, work: &mut PropagationWork) -> bool {
        if self.asleep(index) {
            self.flags[index] = false;
            return false;
        }
        let local = Transform::from_pose(&self.locals[index]);
        let next = self.derived_global(index, &local);
        self.commit(index, next, work)
    }

    #[inline(always)]
    fn commit(&mut self, index: usize, next: Option<Mat4>, work: &mut PropagationWork) -> bool {
        let link = &mut self.links[index];
        let global = &mut self.globals[index];
        let revision = &mut self.revisions[index];
        let rows = next.map(pose_rows);
        let valid = rows.is_some();
        let validity_changed = link.valid != valid;
        let changed = validity_changed
            || rows
                .as_ref()
                .is_some_and(|rows| !same_rows_bits(global, rows));
        if validity_changed {
            if valid {
                *self.invalid_transforms -= 1;
            } else {
                *self.invalid_transforms += 1;
            }
            link.valid = valid;
        }
        let publish = changed || revision.is_none();
        if let Some(rows) = rows
            && changed
        {
            *global = rows;
        }
        self.flags[index] = false;
        work.evaluated += 1;
        if publish {
            *revision = Some(work.value_ids.next());
            work.written += 1;
            self.changes.push(index as u32);
        }
        changed
    }
}

impl GlobalTransforms {
    pub fn get(&self, entity: Entity) -> Option<GlobalTransform> {
        let index = *self.indices.get(&entity)?;
        Some(self.value_at(index))
    }

    pub fn pose(&self, entity: Entity) -> Option<glam::Affine3A> {
        let index = *self.indices.get(&entity)?;
        self.global(index).map(glam::Affine3A::from_mat4)
    }

    pub fn changes_since(&self, cursor: &mut Cursor) -> Since<'_, u32> {
        self.changes.since(cursor)
    }

    pub fn entity_at(&self, index: u32) -> Option<Entity> {
        self.entities.get(index as usize).copied()
    }

    pub(crate) fn resize_changes(&mut self, size: usize) {
        self.changes.resize(size);
        self.static_moves.feed.resize(size);
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
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
    fn global(&self, index: usize) -> Option<Mat4> {
        self.links[index]
            .valid
            .then(|| rows_matrix(&self.globals[index]))
    }

    pub(crate) fn children(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(self.links[index].first_child(), |&child| {
            self.links[child].next_sibling()
        })
    }

    fn dense_threshold(&self) -> usize {
        self.links.len().div_ceil(2)
    }

    fn scan_changed(&mut self, world: &World, work: &mut PropagationWork) -> Scan {
        let _profile = engine_profile::span("hierarchy", "change_tick_scan");
        engine_profile::counter("hierarchy.full_scan", 0.0);
        engine_profile::counter("hierarchy.scanned_nodes", 0.0);
        engine_profile::counter(
            "hierarchy.metadata_nodes",
            self.query.iter(world).len() as f64,
        );
        self.dirty.clear();
        let mut changed_query = self.changed_query.take().expect("scan_state_present");
        let scan = self.scan_changed_rows(&mut changed_query, world, work);
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
            links,
            globals,
            revisions,
            dirty,
            report,
            changes,
            words,
            static_moves,
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
            links,
            globals,
            revisions,
            flags,
            words,
            invalid_transforms: &mut report.invalid_transforms,
            changes,
        }
        .trimmed(len);
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
                        *local = pose;
                    }
                    if !local_changed {
                        continue;
                    }
                    local_revisions[index] = next_revision;
                    nodes.changes.push(index as u32);
                    if nodes.words[index].mobility == Mobility::Static {
                        static_moves.record(entity);
                    }
                    if nodes.asleep(index) {
                        continue;
                    }
                    let link = &nodes.links[index];
                    if !link.has_children()
                        && link.parent().is_none_or(|parent| !nodes.flags[parent])
                    {
                        let next = nodes.derived_global(index, transform);
                        nodes.commit(index, next, work);
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

    fn scan_full(&mut self, world: &World) -> Scan {
        let _profile = engine_profile::span("hierarchy", "change_tick_scan");
        engine_profile::counter("hierarchy.full_scan", 1.0);
        self.dirty.clear();
        self.changes.replace();
        let threshold = self.dense_threshold();
        let mut count = 0;
        let mut metadata = 0;
        let mut checked = 0;
        let mut observed = 0;
        for (entity, transform, parent) in self.query.iter(world) {
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
                let transform_added = transform.added().get();
                let identity_changed = source.transform_added != transform_added;
                if local_changed || identity_changed {
                    observed += 1;
                    source.transform_added = transform_added;
                    *local = pose;
                    if local_changed {
                        self.local_revisions[count] = self.revision.wrapping_add(1);
                    }
                }
                checked += 1;
                if local_changed {
                    if self.words[count].mobility == Mobility::Static {
                        self.static_moves.record(entity);
                    }
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

    fn rebuild_topology(&mut self, world: &mut World, work: &mut PropagationWork) {
        let _profile = engine_profile::span("hierarchy", "topology_rebuild");
        for (entity, transform, _) in self.query.iter(world) {
            let Some(&index) = self.indices.get(&entity) else {
                continue;
            };
            if self.words[index].mobility == Mobility::Static
                && !same_pose_bits(&self.locals[index], &transform.to_pose())
            {
                self.static_moves.record(entity);
            }
        }
        self.asleep.clear();
        self.asleep.extend(
            self.entities
                .iter()
                .zip(&self.words)
                .filter(|(_, word)| !word.active)
                .map(|(&entity, _)| entity),
        );
        let admitted = self.query.iter(world).len().min(self.limit);
        self.empty_tables(admitted);
        let mut report = HierarchyReport::default();
        for (entity, transform, parent) in self.query.iter(world) {
            report.transforms += 1;
            if self.links.len() == self.limit {
                report.overflow_transforms += 1;
                continue;
            }
            let source = Source {
                entity,
                parent: parent.map(Parent::parent),
                transform_added: transform.added().get(),
            };
            self.entities.push(entity);
            self.sources.push(source);
            self.locals.push(transform.to_pose());
            self.local_revisions.push(self.revision.wrapping_add(1));
            self.links.push(Link::DETACHED);
            self.globals.push(pose_rows(Mat4::IDENTITY));
            self.revisions.push(None);
            self.declared.push(words::Declared::default());
            self.words.push(NodeWord::ROOT);
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
            &mut self.states,
            &mut self.path,
            Link::parent,
            |links, states, index| -> Result<(), Infallible> {
                let depth = match sources[index].parent {
                    None => Some(0),
                    Some(_) => links[index].parent().and_then(|parent| {
                        let complete = states[parent] == State::Complete;
                        let parent = &links[parent];
                        (complete
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
        self.changes.replace();
        let indices = &self.indices;
        self.static_moves
            .moved
            .retain(|entity| indices.get(entity).is_some());
        report.invalid_transforms = report.transforms;
        self.report = report;
        self.topology_revision = work.value_ids.next();
    }

    fn empty_tables(&mut self, admitted: usize) {
        self.declared.clear();
        self.words.clear();
        self.entities.clear();
        self.sources.clear();
        self.locals.clear();
        self.local_revisions.clear();
        self.links.clear();
        self.globals.clear();
        self.revisions.clear();
        self.indices.clear();
        self.order.clear();
        self.path.clear();
        self.dirty.clear();
        self.entities.reserve_exact(admitted);
        self.sources.reserve_exact(admitted);
        self.locals.reserve_exact(admitted);
        self.local_revisions.reserve_exact(admitted);
        self.links.reserve_exact(admitted);
        self.globals.reserve_exact(admitted);
        self.revisions.reserve_exact(admitted);
        self.declared.reserve_exact(admitted);
        self.words.reserve_exact(admitted);
        self.order.reserve_exact(admitted);
        self.path.reserve_exact(admitted);
        self.dirty.flags.clear();
        self.dirty.flags.reserve_exact(admitted);
        self.dirty.seeds.reserve_exact(admitted.div_ceil(2));
    }

    fn evaluate_dirty(&mut self, work: &mut PropagationWork) {
        let _profile = engine_profile::span("hierarchy", "evaluate_dirty");
        let dense = self.dirty.count >= self.dense_threshold();
        let Self {
            locals,
            links,
            globals,
            revisions,
            order,
            path,
            dirty,
            report,
            changes,
            words,
            ..
        } = self;
        let len = links.len();
        let mut nodes = Nodes {
            locals,
            links,
            globals,
            revisions,
            flags: &mut dirty.flags,
            words,
            invalid_transforms: &mut report.invalid_transforms,
            changes,
        }
        .trimmed(len);
        if dense {
            for &index in order.iter() {
                if nodes.flags[index] && nodes.update(index, work) {
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
                    if nodes.update(current, work) {
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

    fn plan(&mut self, world: &mut World) -> ScanPlan {
        let limit = node_limit(world);
        if limit != self.limit {
            self.limit = limit;
            self.incomplete = true;
        }
        let interrupted = std::mem::replace(&mut self.incomplete, true);
        let aged = self.scans.aged(world.read_change_tick());
        let structural =
            std::mem::take(&mut world.resource_mut::<tracking::Invalidation>().full_scan);
        if interrupted || structural {
            ScanPlan::Rebuild
        } else if aged || self.report.transforms >= self.limit {
            ScanPlan::Full
        } else {
            ScanPlan::Changed
        }
    }

    fn propagate(&mut self, world: &mut World) -> HierarchyReport {
        let plan = self.plan(world);
        let mut work = PropagationWork::default();
        let scan = match plan {
            ScanPlan::Changed => self.scan_changed(world, &mut work),
            ScanPlan::Full => self.scan_full(world),
            ScanPlan::Rebuild => Scan {
                topology_changed: true,
                observed: 0,
            },
        };
        let rebuilt = scan.topology_changed;
        if rebuilt {
            self.rebuild_topology(world, &mut work);
        }
        let worded = self.declare(world, rebuilt) && self.settle_words(rebuilt);
        if self.dirty.count != 0 {
            self.evaluate_dirty(&mut work);
        }
        if rebuilt || worded || scan.observed != 0 || work.evaluated != 0 {
            self.revision = self.revision.wrapping_add(1);
        }
        engine_profile::counter("hierarchy.topology_rebuilt", f64::from(rebuilt));
        engine_profile::counter("hierarchy.evaluated_nodes", work.evaluated as f64);
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

pub fn propagate_transforms(world: &mut World) -> HierarchyReport {
    let _profile = engine_profile::span("hierarchy", "propagate");
    tracking::init(world);
    world.flush();
    world.check_change_ticks();
    world.init_resource::<HierarchyScratch>();
    let report =
        world.resource_scope(|world, mut scratch: Mut<HierarchyScratch>| scratch.propagate(world));
    if world.resource::<HierarchyScratch>().has_woken() {
        let woken = world.resource_mut::<HierarchyScratch>().take_woken();
        crate::presentation::make_fresh(world, &woken);
    }
    report
}
