use crate::collections::{HashMap, HashSet};
use crate::presentation::pose_rows;
use crate::ticks::{ChangeWindow, STAMP_BLOCK, ScanClock, consume_through};
use crate::{Changes, NodePose, NodeWord, PoseRows, Transform, entity_indices::EntityIndices};
use bevy_ecs::change_detection::ContiguousRef;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemState;
use engine_settings::{Error, ErrorCode, Expected};
use glam::{Mat4, Vec4};
use std::convert::Infallible;

mod global;
mod structure;
pub(crate) mod tracking;
mod validation;
mod views;
mod words;
pub(crate) use global::ValueIds;
pub use global::GlobalTransform;
pub(crate) use global::Revision;

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
pub(crate) const MAX_NORMAL_INVERSE_NORM: f32 = 10_000.0;

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

pub fn subtree(world: &mut World, root: Entity) -> Vec<Entity> {
    hierarchy_now(world);
    let globals = world.resource::<GlobalTransforms>();
    let mut found = vec![root];
    let mut next = 0;
    while let Some(&parent) = found.get(next) {
        found.extend(globals.children_of(parent));
        next += 1;
    }
    found
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
    let count = hierarchy_now(world);
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
    world
        .resource::<GlobalTransforms>()
        .check_parents(world, &replacements)
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

fn missing(world: &World, entity: Entity) -> Broken {
    if world.entities().contains(entity) {
        Broken::MissingTransform(entity)
    } else {
        Broken::MissingEntity(entity)
    }
}

fn below(depth: Result<Option<usize>, Broken>) -> Result<usize, Broken> {
    match depth? {
        None => Ok(0),
        Some(depth) if depth < MAX_HIERARCHY_DEPTH => Ok(depth + 1),
        Some(_) => Err(Broken::DepthExceeded),
    }
}

type Depths = HashMap<Entity, Result<usize, Broken>>;

impl GlobalTransforms {
    fn children_of(&self, parent: Entity) -> impl Iterator<Item = Entity> + '_ {
        let linked = self
            .node_index(parent)
            .into_iter()
            .flat_map(|index| self.children(index));
        let waiting = self
            .orphans
            .iter()
            .copied()
            .filter(move |&orphan| self.sources[orphan].parent == Some(parent));
        linked.chain(waiting).map(|index| self.entities[index])
    }

    fn chain_depth(
        &self,
        world: &World,
        entity: Entity,
        parent_of: &impl Fn(Entity) -> Option<Entity>,
        depths: &mut Depths,
    ) -> Result<usize, Broken> {
        let mut path = Vec::new();
        let mut cursor = entity;
        let mut above = loop {
            if let Some(&known) = depths.get(&cursor) {
                if path.is_empty() {
                    return known;
                }
                break known.map(Some);
            }
            if path.contains(&cursor) {
                break Err(Broken::Cycle(cursor));
            }
            path.push(cursor);
            match parent_of(cursor) {
                None => break Ok(None),
                Some(parent) if self.node_index(parent).is_none() => {
                    break Err(missing(world, parent));
                }
                Some(parent) => cursor = parent,
            }
        };
        let mut depth = Err(Broken::DepthExceeded);
        for &node in path.iter().rev() {
            depth = below(above);
            depths.insert(node, depth);
            above = depth.map(Some);
        }
        depth
    }

    fn check_parents(
        &self,
        world: &World,
        replacements: &HashMap<Entity, Option<Entity>>,
    ) -> Result<(), Broken> {
        let parent_of = |entity: Entity| match replacements.get(&entity) {
            Some(&parent) => parent,
            None => self
                .node_index(entity)
                .and_then(|index| self.sources[index].parent),
        };
        let mut depths = Depths::default();
        let mut stack = Vec::with_capacity(replacements.len());
        for &child in replacements.keys() {
            stack.push((child, self.chain_depth(world, child, &parent_of, &mut depths)?));
        }
        let mut seen: HashSet<Entity> = replacements.keys().copied().collect();
        while let Some((entity, depth)) = stack.pop() {
            for child in self.children_of(entity) {
                if replacements.contains_key(&child) || !seen.insert(child) {
                    continue;
                }
                stack.push((child, below(Ok(Some(depth)))?));
            }
        }
        Ok(())
    }
}

fn hierarchy_now(world: &mut World) -> usize {
    tracking::init(world);
    world.init_resource::<HierarchyScratch>();
    world.resource_scope(|world, mut globals: Mut<HierarchyScratch>| {
        globals.settle_structure(world, &mut PropagationWork::default());
        globals.report.transforms
    })
}

pub(crate) fn despawn_owned(world: &mut World, owned: &[Entity]) {
    hierarchy_now(world);
    let gone: HashSet<Entity> = owned.iter().copied().collect();
    let mut seen = HashSet::default();
    let globals = world.resource::<GlobalTransforms>();
    let guests: Vec<Entity> = owned
        .iter()
        .flat_map(|&entity| globals.children_of(entity))
        .filter(|child| !gone.contains(child) && seen.insert(*child))
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

impl Source {
    const EMPTY: Self = Self {
        entity: Entity::PLACEHOLDER,
        parent: None,
        transform_added: 0,
    };
}

#[derive(Clone, Copy)]
pub(crate) struct Link {
    parent: u32,
    first_child: u32,
    next_sibling: u32,
    prev_sibling: u32,
    depth: u8,
    topology_valid: bool,
    valid: bool,
}

impl Link {
    const DETACHED: Self = Self {
        parent: NO_NODE,
        first_child: NO_NODE,
        next_sibling: NO_NODE,
        prev_sibling: NO_NODE,
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

    #[inline]
    fn segment(&self) -> usize {
        if self.topology_valid {
            usize::from(self.depth)
        } else {
            0
        }
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
    slots: Slots,
    orphans: Vec<usize>,
    walk: Walk,
    dirty: Dirty,
    moved: Changes,
    placed: Changes,
    touched: Vec<Entity>,
    declared: Vec<words::Declared>,
    words: Vec<NodeWord>,
    worded: Vec<usize>,
    words_stale: bool,
    restructured: bool,
    woken: Vec<Entity>,
    asleep: Vec<Entity>,
    scans: ScanClock,
    incomplete: bool,
    limit: usize,
    report: HierarchyReport,
    topology_revision: Revision,
    revision: u64,
}

struct Slots {
    order: Vec<usize>,
    starts: [usize; SEGMENTS + 1],
    positions: Vec<usize>,
    free: Vec<usize>,
}

struct Walk {
    visits: Vec<structure::Visit>,
    pending: Vec<usize>,
    path: Vec<usize>,
    states: Vec<State>,
}

pub(crate) type HierarchyScratch = GlobalTransforms;

const SEGMENTS: usize = MAX_HIERARCHY_DEPTH + 1;
const UNPLACED: usize = usize::MAX;

impl FromWorld for GlobalTransforms {
    fn from_world(world: &mut World) -> Self {
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
            slots: Slots {
                order: Vec::new(),
                starts: [0; SEGMENTS + 1],
                positions: Vec::new(),
                free: Vec::new(),
            },
            orphans: Vec::new(),
            walk: Walk {
                visits: Vec::new(),
                pending: Vec::new(),
                path: Vec::new(),
                states: Vec::new(),
            },
            dirty: Dirty::default(),
            moved: Changes::new(2),
            placed: Changes::new(2),
            touched: Vec::new(),
            declared: Vec::new(),
            words: Vec::new(),
            worded: Vec::new(),
            words_stale: false,
            restructured: false,
            woken: Vec::new(),
            asleep: Vec::new(),
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
        if self.flags[index] {
            return;
        }
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
    moved: &'a mut Changes,
    placed: &'a mut Changes,
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
            moved,
            placed,
        } = self;
        Self {
            locals: &mut locals[..len],
            links: &mut links[..len],
            globals: &mut globals[..len],
            revisions: &mut revisions[..len],
            flags: &mut flags[..len],
            words: &words[..len],
            invalid_transforms,
            moved,
            placed,
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
            self.placed.mark(index as u32);
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
            self.moved.mark(index as u32);
        }
        changed
    }
}

impl GlobalTransforms {
    pub fn get(&self, entity: Entity) -> Option<GlobalTransform> {
        let index = *self.indices.get(&entity)?;
        Some(self.value_at(index))
    }

    pub(crate) fn pose(&self, entity: Entity) -> Option<glam::Affine3A> {
        let index = *self.indices.get(&entity)?;
        self.global(index).map(glam::Affine3A::from_mat4)
    }

    pub(crate) fn entity_at(&self, index: u32) -> Option<Entity> {
        let index = index as usize;
        self.live(index).then(|| self.entities[index])
    }

    #[inline]
    fn live(&self, index: usize) -> bool {
        self.slots.positions.get(index).is_some_and(|&at| at != UNPLACED)
    }

    pub(crate) fn len(&self) -> usize {
        self.links.len() - self.slots.free.len()
    }

    fn value_at(&self, index: usize) -> GlobalTransform {
        GlobalTransform::read_value(self.global(index))
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
        engine_profile::counter("hierarchy.metadata_nodes", self.report.transforms as f64);
        let mut changed_query = self.changed_query.take().expect("scan_state_present");
        let scan = self.scan_changed_rows(&mut changed_query, world, work);
        self.changed_query = Some(changed_query);
        scan
    }

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
            moved,
            placed,
            words,
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
            moved,
            placed,
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
                    nodes.moved.mark(index as u32);
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
        self.moved.mark_all();
        let threshold = self.dense_threshold();
        let mut count = 0;
        let mut checked = 0;
        let mut observed = 0;
        for (entity, transform, parent) in self.query.iter(world) {
            count += 1;
            let Some(&index) = self.indices.get(&entity) else {
                if self.report.overflow_transforms != 0 {
                    continue;
                }
                return Scan {
                    topology_changed: true,
                    observed: 0,
                };
            };
            let source = &mut self.sources[index];
            if source.parent != parent.map(Parent::parent) {
                return Scan {
                    topology_changed: true,
                    observed: 0,
                };
            }
            let local = &mut self.locals[index];
            let pose = transform.to_pose();
            let local_changed = !same_pose_bits(local, &pose);
            let transform_added = transform.added().get();
            let identity_changed = source.transform_added != transform_added;
            if local_changed || identity_changed {
                observed += 1;
                source.transform_added = transform_added;
                *local = pose;
                if local_changed {
                    self.local_revisions[index] = self.revision.wrapping_add(1);
                }
            }
            checked += 1;
            if local_changed {
                self.dirty.mark(index, threshold);
            }
        }
        engine_profile::counter("hierarchy.metadata_nodes", count as f64);
        engine_profile::counter("hierarchy.validated_nodes", checked as f64);
        engine_profile::counter("hierarchy.scanned_nodes", count as f64);
        Scan {
            topology_changed: count != self.report.transforms,
            observed,
        }
    }

    fn rebuild_topology(&mut self, world: &World, work: &mut PropagationWork) {
        let _profile = engine_profile::span("hierarchy", "topology_rebuild");
        self.asleep.clear();
        self.asleep.extend(
            self.slots.order
                .iter()
                .filter(|&&index| !self.words[index].active)
                .map(|&index| self.entities[index]),
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
            self.walk.visits.push(structure::Visit::Idle);
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
        let sources = &self.sources;
        let Ok(()) = complete_parents_first(
            &mut self.links,
            &mut self.walk.states,
            &mut self.walk.path,
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
                Ok(())
            },
        );
        for index in 0..self.links.len() {
            let link = self.links[index];
            match link.parent() {
                Some(parent) if link.topology_valid => self.link_child(parent, index),
                Some(_) | None if self.sources[index].parent.is_some() => {
                    self.orphans.push(index);
                }
                Some(_) | None => {}
            }
        }
        self.lay_order();
        self.dirty.flags.resize(self.links.len(), true);
        self.dirty.count = self.links.len();
        self.moved.mark_all();
        self.placed.mark_all();
        report.invalid_transforms = report.transforms;
        self.report = report;
        self.topology_revision = work.value_ids.next();
        self.words_stale = true;
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
        self.slots.order.clear();
        self.slots.positions.clear();
        self.slots.free.clear();
        self.orphans.clear();
        self.walk.visits.clear();
        self.walk.pending.clear();
        self.worded.clear();
        self.walk.path.clear();
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
        self.walk.visits.reserve_exact(admitted);
        self.walk.path.reserve_exact(admitted);
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
            slots: Slots { order, .. },
            walk: Walk { path, .. },
            dirty,
            report,
            moved,
            placed,
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
            moved,
            placed,
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

    fn settle_structure(&mut self, world: &mut World, work: &mut PropagationWork) -> bool {
        let limit = node_limit(world);
        if limit != self.limit {
            self.limit = limit;
            self.incomplete = true;
        }
        let interrupted = std::mem::replace(&mut self.incomplete, true);
        let mut invalidation = world.resource_mut::<tracking::Invalidation>();
        let whole = std::mem::take(&mut invalidation.full_scan);
        let mut placed = std::mem::take(&mut invalidation.placed);
        let rebuilt = interrupted
            || whole
            || (self.report.overflow_transforms != 0 && !placed.is_empty())
            || !self.restructure(world, &mut placed);
        if rebuilt {
            self.rebuild_topology(world, work);
        }
        placed.clear();
        let mut invalidation = world.resource_mut::<tracking::Invalidation>();
        if invalidation.placed.is_empty() {
            invalidation.placed = placed;
        }
        self.declare(world);
        let worded = self.settle_words();
        self.restructured |= rebuilt || worded;
        self.incomplete = false;
        rebuilt
    }

    fn scan_plan(&self, world: &World) -> ScanPlan {
        if self.scans.aged(world.read_change_tick()) || self.report.transforms >= self.limit {
            ScanPlan::Full
        } else {
            ScanPlan::Changed
        }
    }

    fn propagate(&mut self, world: &mut World) -> HierarchyReport {
        let mut work = PropagationWork::default();
        let mut rebuilt = self.settle_structure(world, &mut work);
        self.incomplete = true;
        let scan = match self.scan_plan(world) {
            ScanPlan::Changed => self.scan_changed(world, &mut work),
            ScanPlan::Full => self.scan_full(world),
        };
        if scan.topology_changed {
            self.rebuild_topology(world, &mut work);
            self.declare(world);
            self.settle_words();
            rebuilt = true;
        }
        if self.dirty.count != 0 {
            self.evaluate_dirty(&mut work);
        }
        self.dirty.clear();
        if std::mem::take(&mut self.restructured)
            || rebuilt
            || scan.observed != 0
            || work.evaluated != 0
        {
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

pub(crate) fn settle_changes(world: &mut World) {
    if let Some(mut globals) = world.get_resource_mut::<GlobalTransforms>() {
        let globals = globals.bypass_change_detection();
        globals.placed.settle();
        globals.moved.settle();
    }
}
