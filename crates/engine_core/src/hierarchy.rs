use crate::interpolation::{InterpolateTransform, InterpolationState, Observer};
use crate::{NodePose, Transform};
use bevy_ecs::change_detection::{ContiguousRef, MAX_CHANGE_AGE, Tick};
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemState;
use glam::{Mat4, Vec4};
use std::{collections::HashMap, fmt};

mod global;
pub(crate) mod tracking;
mod validation;
pub use global::GlobalTransform;
pub(crate) use global::ValueIds;
use validation::Certificate;

/// Maximum parent edges from a root to a transformed entity.
pub const MAX_HIERARCHY_DEPTH: usize = 64;
/// Most transformed entities one propagation pass can index (node indices
/// are 32-bit). [`TransformNodeLimit`] sets a lower budget.
pub const MAX_TRANSFORM_NODES: usize = u32::MAX as usize - 1;

/// Transformed entities propagation admits; beyond it the rest have no
/// world pose and count as overflow in the [`HierarchyReport`]. The default,
/// [`MAX_TRANSFORM_NODES`], sets no budget: memory bounds it.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransformNodeLimit(pub usize);

impl Default for TransformNodeLimit {
    fn default() -> Self {
        Self(MAX_TRANSFORM_NODES)
    }
}

/// The world's transform-node limit, at most [`MAX_TRANSFORM_NODES`].
fn node_limit(world: &World) -> usize {
    world
        .get_resource::<TransformNodeLimit>()
        .map_or(MAX_TRANSFORM_NODES, |limit| {
            limit.0.min(MAX_TRANSFORM_NODES)
        })
}
/// Conservative bound for GPU normal transformation conditioning.
///
/// After normalizing the cofactor matrix's largest absolute coefficient to one,
/// its inverse infinity norm must not exceed this value. This limits cancellation
/// and keeps transformed unit normals away from floating-point underflow.
pub const MAX_NORMAL_INVERSE_NORM: f32 = 10_000.0;

/// A non-owning relationship established through [`set_parent`].
///
/// Despawning a parent never despawns children. An orphan has no valid global
/// transform until explicitly detached or attached to another live parent.
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

/// Counts from the latest explicit propagation pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HierarchyReport {
    pub transforms: usize,
    pub invalid_transforms: usize,
    /// Nodes beyond the fixed cache bound; also included in invalid transforms.
    pub overflow_transforms: usize,
}

/// A rejected relationship edit. Rejection leaves the world unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HierarchyError {
    MissingEntity(Entity),
    MissingTransform(Entity),
    Cycle(Entity),
    DuplicateChild(Entity),
    DepthExceeded,
    CapacityExceeded,
}

impl fmt::Display for HierarchyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEntity(entity) => write!(formatter, "entity {entity:?} does not exist"),
            Self::MissingTransform(entity) => {
                write!(formatter, "entity {entity:?} has no local Transform")
            }
            Self::Cycle(entity) => write!(formatter, "parent relationship cycles at {entity:?}"),
            Self::DuplicateChild(entity) => {
                write!(formatter, "relationship batch repeats child {entity:?}")
            }
            Self::DepthExceeded => write!(
                formatter,
                "parent relationship exceeds {MAX_HIERARCHY_DEPTH} edges from a root"
            ),
            Self::CapacityExceeded => {
                formatter.write_str("world exceeds its transform-node limit (TransformNodeLimit)")
            }
        }
    }
}

impl std::error::Error for HierarchyError {}

fn require_transform(world: &World, entity: Entity) -> Result<(), HierarchyError> {
    if !world.entities().contains(entity) {
        return Err(HierarchyError::MissingEntity(entity));
    }
    if world.get::<Transform>(entity).is_none() {
        return Err(HierarchyError::MissingTransform(entity));
    }
    Ok(())
}

/// Changes a relationship while preserving the child's **local** transform.
///
/// Validates both the prospective ancestor chain and the moved subtree before
/// mutation. This explicit control operation allocates bounded topology storage;
/// ordinary propagation uses a separate reusable linear-time cache. Detachment
/// is explicit and does not preserve the previous world matrix. Derived globals
/// become fresh at the next [`propagate_transforms`] call.
pub fn set_parent(
    world: &mut World,
    child: Entity,
    parent: Option<Entity>,
) -> Result<(), HierarchyError> {
    set_parents(world, &[(child, parent)])
}

struct RelationshipNode {
    entity: Entity,
    parent: Option<Entity>,
    state: State,
    depth: Result<usize, HierarchyError>,
    affected: bool,
}

/// Atomically validates and installs several local-preserving relationships.
///
/// Use one batch for authored scene installation. Costs O(N + changes) work and
/// bounded temporary storage, where N is the world's transformed entity count.
/// Changed nodes and their descendants must have valid ancestry and depth;
/// unrelated invalid branches do not block repairs or scene installation.
/// Repeated child IDs are rejected, including identical repeated changes.
pub fn set_parents(
    world: &mut World,
    changes: &[(Entity, Option<Entity>)],
) -> Result<(), HierarchyError> {
    let limit = node_limit(world);
    if changes.len() > limit {
        return Err(HierarchyError::CapacityExceeded);
    }
    let mut query = world.query::<(Entity, &Transform, Option<&Parent>)>();
    let count = query.iter(world).len();
    if count > limit {
        return Err(HierarchyError::CapacityExceeded);
    }
    if changes.is_empty() {
        return Ok(());
    }
    let mut replacements = HashMap::with_capacity(changes.len());
    for &(child, parent) in changes {
        require_transform(world, child)?;
        if let Some(parent) = parent {
            require_transform(world, parent)?;
        }
        if replacements.insert(child, parent).is_some() {
            return Err(HierarchyError::DuplicateChild(child));
        }
    }
    let mut nodes = Vec::with_capacity(count);
    let mut indices = HashMap::with_capacity(count);
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
    let mut path = Vec::new();
    for start in 0..nodes.len() {
        if nodes[start].state == State::Complete {
            continue;
        }
        path.clear();
        let mut cursor = Some(start);
        while let Some(index) = cursor {
            if nodes[index].state != State::Pending {
                break;
            }
            nodes[index].state = State::Visiting;
            path.push(index);
            cursor = nodes[index]
                .parent
                .and_then(|parent| indices.get(&parent).copied());
        }
        while let Some(index) = path.pop() {
            let node = &nodes[index];
            let (depth, inherited) = match node.parent {
                None => (Ok(0), false),
                Some(parent) => match indices.get(&parent) {
                    None => (
                        Err(if world.entities().contains(parent) {
                            HierarchyError::MissingTransform(parent)
                        } else {
                            HierarchyError::MissingEntity(parent)
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
                                    Err(HierarchyError::DepthExceeded)
                                }
                            })
                        } else {
                            Err(HierarchyError::Cycle(parent.entity))
                        };
                        (depth, parent.affected)
                    }
                },
            };
            let node = &mut nodes[index];
            node.depth = depth;
            node.affected |= inherited;
            node.state = State::Complete;
            if node.affected {
                depth?;
            }
        }
    }
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

/// Shared admission predicate for composed affine transform matrices.
///
/// Rejects nonfinite, non-affine or singular matrices and unsafe normal data.
/// The renderer rescales the basis and its cofactors before normalizing normals.
/// The normalized normal matrix must satisfy [`MAX_NORMAL_INVERSE_NORM`]; this
/// conservative conditioning bound rejects severe anisotropy and near-singular
/// shear even when an inverse technically contains finite values.
#[inline]
pub fn valid_global_matrix(matrix: Mat4) -> bool {
    validation::valid_global_matrix(matrix)
}

/// Exact bit equality of two poses in upload layout, including NaN payloads
/// and signed zeros, in vector compares (a plain lane fold is scalarized in
/// some contexts, measured slower in the change scan). Non-NaN values are
/// bit-identical exactly when they compare equal with equal sign bits; only a
/// NaN lane needs the bit comparison. No call takes either pose, so callers'
/// poses can stay in registers.
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
    // Unequal compares prove different bits unless a lane holds NaN.
    if !(a.is_nan() || b.is_nan() || c.is_nan() || d.is_nan()) {
        return false;
    }
    std::hint::cold_path();
    let bits = |value: Vec4| value.to_array().map(f32::to_bits);
    tail && bits(a) == bits(c) && bits(b) == bits(d)
}

/// Exact bit equality of two local poses, including NaN payloads and signed
/// zeros. Branch-free, so hot scans need no memcmp or early exits.
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
// Unfiltered: scans read each table's stamp slice through `contiguous_iter`.
type ChangedTransforms = SystemState<Query<'static, 'static, (Entity, Ref<'static, Transform>)>>;

/// Stamps classified per block before any payload or source record is read.
const STAMP_BLOCK: usize = 16;

/// Bevy's `Tick::is_newer_than` for one fixed scan window. With wrapping u32
/// subtraction and `since = min(this_run - last_run, MAX_CHANGE_AGE)`, a stamp
/// `t` changed exactly when `min(this_run - t, MAX_CHANGE_AGE) < since`. Because
/// `since <= MAX_CHANGE_AGE`, that is `this_run - t < since`.
#[derive(Clone, Copy)]
pub(crate) struct ChangeWindow {
    this_run: u32,
    since: u32,
}

impl ChangeWindow {
    pub(crate) fn new(last_run: Tick, this_run: Tick) -> Self {
        Self {
            this_run: this_run.get(),
            since: this_run
                .get()
                .wrapping_sub(last_run.get())
                .min(MAX_CHANGE_AGE),
        }
    }

    #[inline]
    pub(crate) fn contains(self, tick: Tick) -> bool {
        self.this_run.wrapping_sub(tick.get()) < self.since
    }

    /// Whether any stamp of a block of at most [`STAMP_BLOCK`] changed.
    /// Branch-free over the whole block so it can compile to vector compares.
    #[cfg_attr(not(feature = "attribution"), inline)]
    #[cfg_attr(feature = "attribution", inline(never))]
    pub(crate) fn any_in_block(self, ticks: &[Tick]) -> bool {
        ticks
            .iter()
            .fold(false, |any, &tick| any | self.contains(tick))
    }

    /// Whether any stamp changed. Fixed-size blocks compile to unrolled vector
    /// compares, and the test stops at the first changed block.
    #[inline]
    pub(crate) fn any(self, ticks: &[Tick]) -> bool {
        let (blocks, rest) = ticks.as_chunks::<STAMP_BLOCK>();
        blocks.iter().any(|block| self.any_in_block(block)) || self.any_in_block(rest)
    }
}

/// Validated cache hint into a published transform store.
///
/// A hint never identifies an entity on its own. Every read checks the complete
/// generational Entity before using it, including after topology changes.
#[derive(Debug, Clone, Copy)]
pub struct TransformIndex(pub(crate) usize);
impl Default for TransformIndex {
    fn default() -> Self {
        Self(usize::MAX)
    }
}

// Node indices and depths are stored compactly in `Link`.
const _: () = assert!(MAX_TRANSFORM_NODES < u32::MAX as usize);
const _: () = assert!(MAX_HIERARCHY_DEPTH < u8::MAX as usize);
const NO_NODE: u32 = u32::MAX;

/// Source identity and change keys. Every per-node array (`sources`, `locals`,
/// `links`, `globals`, `revisions`) shares one index space. Passes stream only
/// the arrays they read: presentation and evaluation never load identity keys
/// with every matrix, and topology/validity checks never load matrices.
pub(crate) struct Source {
    pub(crate) entity: Entity,
    pub(crate) parent: Option<Entity>,
    pub(crate) transform_added: u32,
    pub(crate) marker_added: Option<u32>,
}

/// Compact traversal and validity record (16 bytes).
#[derive(Clone, Copy)]
pub(crate) struct Link {
    parent: u32,
    first_child: u32,
    next_sibling: u32,
    depth: u8,
    state: State,
    // Topology validity does not depend on the current pose. An invalid pose
    // can recover without rebuilding ancestry, while cycles/depth cannot.
    topology_valid: bool,
    global: Global,
}

/// Publication state of a node's authoritative global.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Global {
    /// No valid global; invalid payloads are never exposed.
    Invalid,
    /// Valid and stored in `globals[index]`.
    Stored,
    /// A leaf proven valid by its parent's [`Certificate`] and not composed:
    /// reads compose the parent's stored global with the cached local, the
    /// same operations and bits as eager evaluation (decision 016).
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

    /// Parent edges from the root; meaningful only with valid topology.
    #[inline]
    pub(crate) fn depth(&self) -> u8 {
        self.depth
    }

    #[inline]
    pub(crate) fn has_global(&self) -> bool {
        self.global != Global::Invalid
    }

    /// Children with valid topology are linked; others never read this node.
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

/// Canonical authoritative world poses from the latest propagation boundary.
///
/// Local Transform remains ordinary mutable ECS data. Read these derived values
/// through this resource; gameplay cannot mutate or replace the published poses.
/// An absent entry is unprepared, removed, or beyond the transform budget.
#[derive(Resource)]
pub struct GlobalTransforms {
    query: TransformQuery,
    // Taken out while scanning so changed leaves can evaluate in the same pass.
    changed_query: Option<ChangedTransforms>,
    // Each source's Entity, the only source key the change scan reads, in a
    // dense copy so matching rows streams 8 bytes per node.
    entities: Vec<Entity>,
    pub(crate) sources: Vec<Source>,
    // The sole cached copy of each admitted local pose, in upload layout: GPU
    // presentation uploads current poses from it directly.
    pub(crate) locals: Vec<NodePose>,
    // Boundary revision (`Self::revision`) at which each local last changed.
    pub(crate) local_revisions: Vec<u64>,
    // Interpolation record of each node (u32::MAX: none), paired with the
    // record's node index and rewritten at every topology rebuild.
    pub(crate) history_hints: Vec<u32>,
    pub(crate) links: Vec<Link>,
    globals: Vec<Mat4>,
    // Admission bounds of each stored global with children, checked by its
    // leaves instead of composing theirs (decision 016).
    certificates: Vec<Certificate>,
    revisions: Vec<u64>,
    pub(crate) indices: HashMap<Entity, usize>,
    pub(crate) order: Vec<usize>,
    path: Vec<usize>,
    dirty: Dirty,
    // Nodes whose local or output revision changed since GPU presentation
    // last gathered (decision 014): its gather visits only these.
    pub(crate) changes: ChangeLog,
    last_scan: Option<Tick>,
    incomplete: bool,
    // Transformed entities admitted ([`TransformNodeLimit`]).
    pub(crate) limit: usize,
    report: HierarchyReport,
    pub(crate) topology_revision: u64,
    // Advances whenever a boundary observed a changed source, rebuilt topology
    // or evaluated nodes: consumers of local/world state can skip otherwise.
    pub(crate) revision: u64,
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
            indices: HashMap::new(),
            order: Vec::new(),
            path: Vec::new(),
            dirty: Dirty::default(),
            changes: ChangeLog::default(),
            last_scan: None,
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
    // Certified leaves whose globals were not composed.
    deferred: usize,
    written: usize,
    value_ids: ValueIds,
}

/// Nodes awaiting evaluation at the current boundary.
#[derive(Default)]
struct Dirty {
    // Compact flags parallel to `nodes`: marking a changed source does not
    // touch its large matrix/topology record before evaluation reads it.
    flags: Vec<bool>,
    // Sparse traversal seeds. Collection stops at the dense threshold because
    // dense traversal streams every node in parent-first order instead.
    seeds: Vec<usize>,
    count: usize,
}

impl Dirty {
    fn clear(&mut self) {
        self.seeds.clear();
        self.count = 0;
    }

    /// Each index is marked at most once per scan: sources are unique rows.
    #[inline]
    fn mark(&mut self, index: usize, threshold: usize) {
        self.flags[index] = true;
        if self.count < threshold {
            self.seeds.push(index);
        }
        self.count += 1;
    }
}

/// Nodes whose cached local or output revision changed since the log was
/// last drained, each listed once. GPU presentation's gather drains it and
/// visits only these nodes instead of every node (decision 014). Past a
/// quarter of the nodes the log stops listing and reports `overflow`: the
/// next gather walks every node. Rebuilds and full scans also overflow it.
pub(crate) struct ChangeLog {
    nodes: Vec<u32>,
    // Per node: the drain generation in which it was last listed.
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
            // Nothing drained this node space yet.
            overflow: true,
        }
    }
}

impl ChangeLog {
    /// Starts a new node space: every node counts as changed.
    fn reset(&mut self, nodes: usize) {
        self.nodes.clear();
        self.marks.clear();
        self.marks.resize(nodes, 0);
        self.generation = 1;
        self.limit = nodes / 4;
        self.overflow = true;
    }

    /// Lists `index` once per drain. One flag test after overflow.
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

    /// Changes since the last drain: every node listed, or `None` after an
    /// overflow. Hand the list back with [`Self::restore`].
    pub(crate) fn take(&mut self) -> Option<Vec<u32>> {
        (!self.overflow).then(|| std::mem::take(&mut self.nodes))
    }

    /// Ends a drain: later changes are listed again. Keeps the capacity of a
    /// list returned by [`Self::take`].
    pub(crate) fn restore(&mut self, nodes: Option<Vec<u32>>) {
        if let Some(nodes) = nodes {
            self.nodes = nodes;
        }
        self.nodes.clear();
        self.overflow = false;
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Marks from 2^32 drains ago must not pass as current.
            self.marks.fill(0);
            self.generation = 1;
        }
    }
}

struct Scan {
    topology_changed: bool,
    // Rows whose local pose or identity keys changed.
    observed: usize,
}

/// Split borrow of the per-node arrays that evaluation reads and writes.
/// Passes hold one for their whole loop, so each node's work keeps array
/// pointers and lengths in locals instead of reloading them from the store
/// after every element write.
struct Nodes<'a> {
    locals: &'a mut [NodePose],
    history_hints: &'a mut [u32],
    links: &'a mut [Link],
    globals: &'a mut [Mat4],
    certificates: &'a mut [Certificate],
    revisions: &'a mut [u64],
    flags: &'a mut [bool],
    invalid_transforms: &'a mut usize,
    changes: &'a mut ChangeLog,
}

/// Index of a source whose row left the retained table order. Out of line
/// and by value, so the common sequential match keeps its Entity in registers.
#[cold]
#[inline(never)]
fn find_index(indices: &HashMap<Entity, usize>, entity: Entity) -> Option<usize> {
    indices.get(&entity).copied()
}

/// A deferred leaf's global: its parent's stored global (identity for roots)
/// times its cached local, exactly as eager evaluation composes it.
#[cfg_attr(not(feature = "attribution"), inline)]
#[cfg_attr(feature = "attribution", inline(never))]
fn deferred_global(links: &[Link], globals: &[Mat4], locals: &[NodePose], index: usize) -> Mat4 {
    let parent = links[index].parent().map_or(Mat4::IDENTITY, |parent| {
        debug_assert!(links[parent].global == Global::Stored);
        globals[parent]
    });
    parent * Transform::from_pose(&locals[index]).matrix()
}

impl Nodes<'_> {
    /// Views every array at `len` elements, the common per-node length: after
    /// one checked access at an index, compiled passes skip the other checks.
    /// Inlined so the view stays in registers.
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
            changes,
        }
    }

    /// Authoritative matrix at an admitted index; `None` when invalid.
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

    /// The global of `index` for `local`, which equals its cached local's bits.
    // Measured natively: keeping this per-node path inlined into both
    // traversals matters more than batching local matrix construction.
    // Attribution builds (docs/profiling.md) keep it and other hot helpers
    // out of line so sampling profilers can separate their costs.
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

    /// Whether `index`, a leaf with `local` (its cached local's bits), is
    /// proven valid by its parent's certificate without composing its global.
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

    /// Evaluates a changed node for `local` (its cached local's bits): a
    /// certified leaf defers its global, anything else composes and validates
    /// it. Returns whether descendants need to inherit a changed global.
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
            // Deferred nodes are leaves: nothing inherits their global.
            return true;
        }
        let next = self.derived_global(index, local);
        self.commit(index, next, work, observer)
    }

    /// Evaluates a dirty node from its cached local; returns whether
    /// descendants need to inherit a changed global.
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

    /// Publishes a certified leaf without composing its global. Its inputs
    /// changed, so it always takes a new output revision (a conservative one
    /// in the rare case that the composed bits are unchanged).
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

    /// Publishes an evaluated global; returns whether it changed. Unchanged
    /// validity and matrix bits leave their records unwritten. A node with
    /// children also refreshes the certificate its leaves are checked against.
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
        // A deferred global was never stored: the stored bits are stale.
        let changed = validity_changed
            || before == Global::Deferred
            || next.is_some_and(|matrix| !same_finite_matrix_bits(*global, matrix));
        if validity_changed {
            if valid {
                *self.invalid_transforms -= 1;
            } else {
                *self.invalid_transforms += 1;
            }
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
    /// Returns a read value. Invalid admitted poses have matrix() == None.
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

    /// Revision changes when matrix bits or validity change, and may be
    /// conservatively reissued when the topology cache is rebuilt. u64::MAX is an
    /// exhausted revision sentinel and must never be reused by a consumer cache.
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

    /// Output revision at an admitted index.
    #[inline]
    pub(crate) fn revision_at(&self, index: usize) -> u64 {
        self.revisions[index]
    }

    /// Authoritative matrix at an admitted index; `None` when invalid.
    #[inline]
    pub(crate) fn global(&self, index: usize) -> Option<Mat4> {
        match self.links[index].global {
            Global::Stored => Some(self.globals[index]),
            // Composed on read from the parent's stored global and the cached
            // local: the bits eager evaluation would have stored.
            Global::Deferred => Some(deferred_global(
                &self.links,
                &self.globals,
                &self.locals,
                index,
            )),
            Global::Invalid => None,
        }
    }

    /// Children with valid ancestry, reusing the bounded topology cache.
    pub(crate) fn children(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(self.links[index].first_child(), |&child| {
            self.links[child].next_sibling()
        })
    }

    /// Dense traversal visits the whole parent-first order at this many seeds.
    fn dense_threshold(&self) -> usize {
        self.links.len().div_ceil(2)
    }

    /// Classifies each table's change stamps in blocks before touching any
    /// payload or source record; unchanged blocks cost one pass over their
    /// stamps. The system state owns its window independently of
    /// `World::clear_trackers`.
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
        let mut changed_query = self
            .changed_query
            .take()
            .expect("change scan state is present between scans");
        let scan = self.scan_changed_rows(&mut changed_query, world, work, interpolation);
        self.changed_query = Some(changed_query);
        scan
    }

    fn scan_changed_rows(
        &mut self,
        changed_query: &mut ChangedTransforms,
        world: &World,
        work: &mut PropagationWork,
        interpolation: &mut Option<&mut InterpolationState>,
    ) -> Scan {
        let threshold = self.dense_threshold();
        // Every observed change advances this boundary's revision.
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
            changes,
        }
        .trimmed(len);
        let mut observer = interpolation
            .as_deref_mut()
            .map(InterpolationState::observer);
        let query = changed_query.get(world).expect("read-only transform query");
        for (entities, transforms) in query
            .contiguous_iter()
            .expect("Transform uses dense table storage")
        {
            let (transforms, ticks) = ContiguousRef::split(transforms);
            let window = ChangeWindow::new(ticks.last_run(), ticks.this_run());
            let (changed, added) = (ticks.changed(), ticks.added());
            // Refreshes between ticks usually find no change: one unrolled pass
            // over the table's stamps then replaces the per-block loop.
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
                    // Consecutive changed rows use the retained table order; sparse
                    // changes or unrelated archetype moves fall back to full Entity IDs.
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
                    // The stamp passed this window; only an equal-value mutable
                    // write remains to be filtered out.
                    let pose = transform.to_pose();
                    let local_changed = !same_pose_bits(local, &pose);
                    // Parent and marker changes are structural and never reach
                    // this scan; only a re-added Transform changes identity
                    // (`World::modify_component` resets its added tick without
                    // an insertion hook). That tick then lies in this window, so
                    // other rows never load their source record.
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
                            // Non-interpolated sources keep their empty hint.
                            if !motion
                                && (*hint != u32::MAX || sources[index].marker_added.is_some())
                            {
                                // A fresh copy keeps `pose` out of memory on the
                                // common path.
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
                    // A changed leaf under a parent not waiting for this
                    // boundary evaluates now, while its pose is hot, instead of in
                    // the parent-first pass. A parent marked later in this scan
                    // re-evaluates its subtree there.
                    let link = &nodes.links[index];
                    if !link.has_children()
                        && link.parent().is_none_or(|parent| !nodes.flags[parent])
                    {
                        // Evaluated from the ECS value just cached, bit for
                        // bit, instead of unpacking the upload-layout copy.
                        nodes.settle(index, transform, work, &mut observer);
                        early += 1;
                    } else {
                        // Each index is marked at most once per scan:
                        // sources are unique rows.
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
        engine_profile::counter("hierarchy.global_matrix_checks", 0.0);
        Scan {
            topology_changed: false,
            observed,
        }
    }

    /// Revalidate all source identities and payloads after structural changes,
    /// interrupted propagation, or an aged change-stamp window.
    fn scan_full(
        &mut self,
        world: &World,
        interpolation: &mut Option<&mut InterpolationState>,
    ) -> Scan {
        let _profile = engine_profile::span("hierarchy", "change_tick_scan");
        engine_profile::counter("hierarchy.full_scan", 1.0);
        self.dirty.clear();
        // Rare: consumers of the change log walk every node once.
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
        engine_profile::counter("hierarchy.global_matrix_checks", 0.0);
        engine_profile::counter("hierarchy.scanned_nodes", count as f64);
        Scan {
            topology_changed: count != self.report.transforms
                || self.sources.len() != count.min(self.limit),
            observed,
        }
    }

    /// Structural edits rebuild indices, ancestry, order and child links once.
    /// Local-pose edits never clear these retained collections.
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
        // Sparse seeds never exceed the dense threshold.
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
                // Reconciliation cleared every record's node index; this
                // resolves the record by Entity and restores the pair.
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
        for start in 0..self.links.len() {
            if self.links[start].state == State::Complete {
                continue;
            }
            self.path.clear();
            let mut cursor = Some(start);
            while let Some(index) = cursor {
                if self.links[index].state != State::Pending {
                    break;
                }
                self.links[index].state = State::Visiting;
                self.path.push(index);
                cursor = self.links[index].parent();
            }
            while let Some(index) = self.path.pop() {
                let depth = match self.sources[index].parent {
                    None => Some(0),
                    Some(_) => self.links[index].parent().and_then(|parent| {
                        let parent = &self.links[parent];
                        (parent.state == State::Complete
                            && parent.topology_valid
                            && usize::from(parent.depth) < MAX_HIERARCHY_DEPTH)
                            .then_some(parent.depth + 1)
                    }),
                };
                let link = &mut self.links[index];
                link.topology_valid = depth.is_some();
                link.depth = depth.unwrap_or(0);
                link.state = State::Complete;
                self.order.push(index);
            }
        }
        // Only valid ancestry enters child traversal. Cycles and their dependent
        // branches remain invalid until a structural edit repairs them.
        for index in 0..self.links.len() {
            if self.links[index].topology_valid
                && let Some(parent) = self.links[index].parent()
            {
                self.links[index].next_sibling = self.links[parent].first_child;
                self.links[parent].first_child = index as u32;
            }
        }
        // Every node is dirty, which always selects dense parent-first traversal.
        self.dirty.flags.resize(self.links.len(), true);
        self.dirty.count = self.links.len();
        self.changes.reset(self.links.len());
        // A structural rebuild conservatively reissues output revisions. No
        // duplicate full-pose buffer is retained solely to preserve those stamps.
        report.invalid_transforms = report.transforms;
        self.report = report;
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
            changes,
        }
        .trimmed(len);
        let mut observer = interpolation
            .as_deref_mut()
            .map(InterpolationState::observer);
        // Dense movement streams the cached parent-first order instead of
        // walking the ancestor chain separately for every changed entity.
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
            // A dirty ancestor can evaluate to the same matrix (or remain
            // invalid). Its independently edited descendant must still run.
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
        // A changed limit rescans every source.
        let limit = node_limit(world);
        if limit != self.limit {
            self.limit = limit;
            self.incomplete = true;
        }
        let interrupted = std::mem::replace(&mut self.incomplete, true);
        let scan_tick = world.read_change_tick();
        let aged = self
            .last_scan
            .is_none_or(|last| scan_tick.get().wrapping_sub(last.get()) >= MAX_CHANGE_AGE);
        let structural =
            std::mem::take(&mut world.resource_mut::<tracking::Invalidation>().full_scan);
        let force = interrupted || aged || structural || self.report.transforms >= self.limit;
        let mut work = PropagationWork::default();
        let mut scan = if force {
            self.scan_full(world, &mut interpolation)
        } else {
            self.scan_changed(world, &mut work, &mut interpolation)
        };
        // Source reconciliation includes component lifetimes and interpolation
        // membership, not just parent links. A forced rebuild is rare.
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
        engine_profile::counter("memory.global_write_buffer_bytes", 0.0);
        // Only the local input scan touches ECS timestamps. A completed boundary
        // owns its window independently of external clear_trackers calls.
        let completed_tick = world.read_change_tick();
        self.last_scan = Some(completed_tick);
        self.changed_query
            .as_mut()
            .expect("change scan state is restored after scanning")
            .meta_mut()
            .set_last_run(completed_tick);
        // The next ordinary mutable borrow must be newer than the window just
        // consumed, even without a schedule run or clear_trackers in between.
        world.increment_change_tick();
        self.incomplete = false;
        self.report
    }
}

/// Local poses must be finite with a near-unit rotation before composition.
///
/// Only the rotation needs an explicit test: `<=` also rejects NaN/infinite
/// norms. A non-finite scale or translation component makes every lane of
/// its matrix column non-finite (0 * inf is NaN; other products are infinite),
/// and multiplying by any parent keeps that column non-finite, so admission of
/// the composed matrix rejects it exactly as an early local check would.
#[inline]
fn local_matrix(local: &Transform) -> Option<Mat4> {
    ((local.rotation.length_squared() - 1.0).abs() <= 1.0e-3).then(|| local.matrix())
}

#[cfg(test)]
pub(crate) fn same_matrix_bits(left: Mat4, right: Mat4) -> bool {
    left.to_cols_array().map(f32::to_bits) == right.to_cols_array().map(f32::to_bits)
}

/// Bit equality for finite matrices (admitted poses): non-NaN floats are
/// bit-identical exactly when they compare equal and share a sign bit, which
/// also distinguishes the two zeros. Four vector compares and sign masks.
#[inline(always)]
pub(crate) fn same_finite_matrix_bits(left: Mat4, right: Mat4) -> bool {
    debug_assert!(left.is_finite() && right.is_finite());
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

/// Refreshes derived globals without advancing clocks or running gameplay.
///
/// Ordinary mutable writes use Bevy change stamps and an engine-owned scan
/// window. Metadata is visited in table order; unchanged payloads are not read.
/// Structural changes and aged scan windows use full source validation. Explicit
/// change-detection bypasses/manual stamp edits are unsupported for transform data.
///
/// Orphans, cycles, excessive depth, invalid matrices, and overflow become
/// `GlobalTransform::matrix() == None`; healthy unrelated branches still update.
/// Removing `Transform` removes its published pose at the next refresh. Storage is bounded by
/// [`TransformNodeLimit`]; traversal is iterative, with ancestor walks bounded
/// by [`MAX_HIERARCHY_DEPTH`].
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

#[cfg(test)]
mod tests;
