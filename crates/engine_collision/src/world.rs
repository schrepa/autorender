use std::collections::HashMap;

use bevy_ecs::prelude::*;
use bevy_ecs::system::{SystemParam, SystemState};
use engine_core::{
    FixedTime, FrameTime, GlobalTransforms, Transform, TransformIndex, propagate_transforms,
};
use glam::{Mat4, Vec3};

use crate::{
    bvh::Bvh,
    geometry::{AffineShape, Bounds, NarrowScratch, intersects},
    types::*,
};

type ColliderData = (Entity, &'static Collider, Option<&'static Transform>);

/// Scheduled synchronization reads collider/local components and the canonical
/// global store, and writes only collision cache state. Engine boundaries have
/// already propagated the authoritative transforms.
#[derive(SystemParam)]
pub(crate) struct CollisionSync<'w, 's> {
    colliders: Query<'w, 's, ColliderData>,
    globals: Res<'w, GlobalTransforms>,
    config: Res<'w, CollisionConfig>,
    frame: Option<Res<'w, FrameTime>>,
    tick: Option<Res<'w, FixedTime>>,
    collisions: ResMut<'w, CollisionWorld>,
}

impl CollisionSync<'_, '_> {
    fn refresh(&mut self) -> CollisionReport {
        self.collisions.refresh(
            *self.config,
            self.frame.as_ref().map_or(0, |time| time.frame),
            self.tick.as_ref().map_or(0, |time| time.tick),
            &self.colliders,
            &self.globals,
        )
    }
}

/// Cached query/access metadata for the explicit World API, initialized lazily.
/// It owns no collider snapshot and may outlive replacement of CollisionWorld;
/// each use fetches the current resources and updates matching archetypes.
#[derive(Resource)]
struct DirectSyncState(SystemState<CollisionSync<'static, 'static>>);

impl FromWorld for DirectSyncState {
    fn from_world(world: &mut World) -> Self {
        Self(SystemState::new(world))
    }
}

#[derive(Clone, Copy)]
struct Input {
    entity: Entity,
    collider: Collider,
    // Authoritative output revision of this pose (None without a local
    // Transform or admitted entry). Equal revisions certify equal matrix bits
    // and validity; the exhausted u64::MAX stamp never counts as unchanged.
    revision: Option<u64>,
    // Validated store index hint, never an identity.
    hint: TransformIndex,
    status: ColliderStatus,
}

/// The collider's authoritative pose and output revision, read through a
/// validated hint instead of a hash lookup by entity.
fn read_pose(
    globals: &GlobalTransforms,
    entity: Entity,
    has_local: bool,
    hint: &mut TransformIndex,
) -> (Option<Mat4>, Option<u64>) {
    if !has_local {
        return (None, None);
    }
    globals
        .get_indexed_with_revision(entity, hint)
        .map_or((None, None), |(pose, revision)| {
            (pose.matrix(), Some(revision))
        })
}
#[derive(Clone, Copy)]
struct Entry {
    entity: Entity,
    collider: Collider,
    affine: AffineShape,
}

/// Owned authoritative collision snapshot. Queries never scan ECS or update events.
#[derive(Resource)]
pub struct CollisionWorld {
    config: CollisionConfig,
    inputs: Vec<Input>,
    statuses: HashMap<Entity, ColliderStatus>,
    entries: Vec<Entry>,
    bounds: Vec<Bounds>,
    bvh: Bvh,
    narrow: NarrowScratch,
    next_pairs: Vec<SensorPair>,
    report: CollisionReport,
    last_error: Option<CollisionError>,
}

impl FromWorld for CollisionWorld {
    fn from_world(_world: &mut World) -> Self {
        Self {
            config: CollisionConfig::default(),
            inputs: Vec::new(),
            statuses: HashMap::new(),
            entries: Vec::new(),
            bounds: Vec::new(),
            bvh: Bvh::default(),
            narrow: NarrowScratch::default(),
            next_pairs: Vec::new(),
            report: CollisionReport::default(),
            last_error: None,
        }
    }
}

impl CollisionWorld {
    pub fn report(&self) -> CollisionReport {
        self.report
    }
    pub fn config(&self) -> CollisionConfig {
        self.config
    }
    pub fn last_error(&self) -> Option<CollisionError> {
        self.last_error
    }
    /// None includes entities outside the admitted cache capacity.
    pub fn collider_status(&self, entity: Entity) -> Option<ColliderStatus> {
        self.statuses.get(&entity).copied()
    }

    fn ready(&self) -> Result<(), CollisionError> {
        if self.report.complete {
            Ok(())
        } else {
            Err(self.last_error.unwrap_or(CollisionError::IncompleteWorld))
        }
    }

    fn refresh(
        &mut self,
        config: CollisionConfig,
        frame: u64,
        tick: u64,
        colliders: &Query<'_, '_, ColliderData>,
        globals: &GlobalTransforms,
    ) -> CollisionReport {
        let _profile = engine_profile::span("collision", "sync");
        let count = colliders.iter().len();
        self.report.frame = frame;
        self.report.tick = tick;
        let limit = config.max_colliders.min(MAX_COLLISION_NODES);
        let admitted = count.min(limit);
        // Collider shape, identity and pose revision determine every cached
        // value. Revisions change whenever matrix bits or validity change.
        let unchanged = self.report.revision != 0
            && config == self.config
            && count == self.report.colliders
            && self.inputs.len() == admitted
            && colliders.iter().take(limit).zip(&mut self.inputs).all(
                |((entity, collider, local), old)| {
                    old.entity == entity
                        && old.collider == *collider
                        && old.revision != Some(u64::MAX)
                        && read_pose(globals, entity, local.is_some(), &mut old.hint).1
                            == old.revision
                },
            );
        if unchanged {
            return self.report;
        }
        let _rebuild = engine_profile::span("collision", "cache_rebuild");
        self.config = config;
        self.report.revision = self.report.revision.saturating_add(1);
        self.report.colliders = count;
        self.report.invalid_colliders = 0;
        self.report.overflow_colliders = count.saturating_sub(limit);
        self.entries.clear();
        self.bounds.clear();
        // Inputs are rewritten in place so unchanged membership and statuses
        // can keep the status map instead of reinserting every collider.
        let mut statuses_changed = self.inputs.len() != admitted;
        self.inputs.truncate(admitted);
        self.inputs
            .reserve_exact(admitted.saturating_sub(self.inputs.len()));
        self.entries.reserve_exact(admitted);
        self.bounds.reserve_exact(admitted);
        let configuration_valid = config.validate().is_ok();
        for (index, (entity, collider, local)) in colliders.iter().take(limit).enumerate() {
            let mut hint = self
                .inputs
                .get(index)
                .filter(|old| old.entity == entity)
                .map_or_else(TransformIndex::default, |old| old.hint);
            let (matrix, revision) = read_pose(globals, entity, local.is_some(), &mut hint);
            let affine = matrix
                .and_then(|matrix| AffineShape::new(collider.shape(), matrix).ok())
                .filter(|_| configuration_valid);
            let status = if let Some(affine) = affine {
                self.entries.push(Entry {
                    entity,
                    collider: *collider,
                    affine,
                });
                self.bounds.push(affine.bounds);
                ColliderStatus::Active
            } else {
                self.report.invalid_colliders += 1;
                ColliderStatus::Invalid
            };
            let input = Input {
                entity,
                collider: *collider,
                revision,
                hint,
                status,
            };
            if let Some(old) = self.inputs.get_mut(index) {
                statuses_changed |= old.entity != entity || old.status != status;
                *old = input;
            } else {
                statuses_changed = true;
                self.inputs.push(input);
            }
        }
        if statuses_changed {
            self.statuses.clear();
            self.statuses.reserve(admitted);
            self.statuses
                .extend(self.inputs.iter().map(|input| (input.entity, input.status)));
        }
        // Unchanged membership keeps every entry's index: refit the tree.
        self.bvh.update(&self.bounds, !statuses_changed);
        self.report.active_colliders = self.entries.len();
        self.report.bvh_nodes = self.bvh.len();
        self.report.cache_bytes = self.inputs.capacity() * size_of::<Input>()
            + self.entries.capacity() * size_of::<Entry>()
            + self.bounds.capacity() * size_of::<Bounds>()
            + self.bvh.bytes();
        self.report.complete = configuration_valid
            && self.report.invalid_colliders == 0
            && self.report.overflow_colliders == 0;
        self.last_error = if !configuration_valid {
            Some(CollisionError::InvalidConfig)
        } else if !self.report.complete {
            Some(CollisionError::IncompleteWorld)
        } else {
            None
        };
        self.report
    }

    pub fn raycast(
        &self,
        ray: CollisionRay,
        filter: QueryFilter,
    ) -> Result<Option<RayHit>, CollisionError> {
        let _profile = engine_profile::span("collision.query", "raycast");
        let mut precise = engine_profile::Aggregate::new("collision.narrow", "ray_tests");
        let mut broad = engine_profile::Aggregate::new("collision.broad", "bvh_traversal");
        let ray = ray.normalized()?;
        self.ready()?;
        let mut nearest: Option<RayHit> = None;
        for entry in candidates(
            (&self.bvh, &self.entries),
            self.config.max_query_candidates,
            |bounds| bounds.ray(ray),
            filter,
            &mut broad,
        ) {
            let entry = entry?;
            if let Some((distance, normal, started_inside)) =
                precise.measure(|| entry.affine.ray(ray))
            {
                let hit = RayHit {
                    entity: entry.entity,
                    distance,
                    point: ray.origin + ray.direction * distance,
                    normal,
                    started_inside,
                };
                if nearest.is_none_or(|old| (distance, entry.entity) < (old.distance, old.entity)) {
                    nearest = Some(hit);
                }
            }
        }
        Ok(nearest)
    }

    /// Sorted by distance then generational entity identity. Errors clear output.
    pub fn raycast_all(
        &self,
        ray: CollisionRay,
        filter: QueryFilter,
        output: &mut Vec<RayHit>,
        limit: usize,
    ) -> Result<(), CollisionError> {
        let _profile = engine_profile::span("collision.query", "raycast_all");
        let mut precise = engine_profile::Aggregate::new("collision.narrow", "ray_tests");
        let mut broad = engine_profile::Aggregate::new("collision.broad", "bvh_traversal");
        output.clear();
        let result = (|| {
            let ray = ray.normalized()?;
            self.ready()?;
            self.validate_limit(limit)?;
            for entry in candidates(
                (&self.bvh, &self.entries),
                self.config.max_query_candidates,
                |bounds| bounds.ray(ray),
                filter,
                &mut broad,
            ) {
                let entry = entry?;
                if let Some((distance, normal, started_inside)) =
                    precise.measure(|| entry.affine.ray(ray))
                {
                    if output.len() == limit {
                        return Err(CollisionError::ResultBudget);
                    }
                    output.push(RayHit {
                        entity: entry.entity,
                        distance,
                        point: ray.origin + ray.direction * distance,
                        normal,
                        started_inside,
                    });
                }
            }
            output.sort_unstable_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then(a.entity.cmp(&b.entity))
            });
            Ok(())
        })();
        if result.is_err() {
            output.clear();
        }
        result
    }

    /// Geometric narrow-phase overlaps within finite tolerance; sorted entity IDs.
    /// Result/candidate exhaustion returns an error and clears output.
    pub fn overlap(
        &self,
        shape: ColliderShape,
        matrix: Mat4,
        filter: QueryFilter,
        output: &mut Vec<Entity>,
        limit: usize,
    ) -> Result<(), CollisionError> {
        let _profile = engine_profile::span("collision.query", "overlap");
        let mut precise = engine_profile::Aggregate::new("collision.narrow", "overlap_tests");
        let mut broad = engine_profile::Aggregate::new("collision.broad", "bvh_traversal");
        output.clear();
        let result = (|| {
            let shape = AffineShape::new(shape, matrix)?;
            self.ready()?;
            self.validate_limit(limit)?;
            for entry in candidates(
                (&self.bvh, &self.entries),
                self.config.max_query_candidates,
                |bounds| bounds.overlaps(shape.bounds),
                filter,
                &mut broad,
            ) {
                let entry = entry?;
                if precise.measure(|| intersects(&shape, &entry.affine))? {
                    if output.len() == limit {
                        return Err(CollisionError::ResultBudget);
                    }
                    output.push(entry.entity);
                }
            }
            output.sort_unstable();
            Ok(())
        })();
        if result.is_err() {
            output.clear();
        }
        result
    }

    pub fn cast_shape(
        &mut self,
        shape: ColliderShape,
        matrix: Mat4,
        displacement: Vec3,
        filter: QueryFilter,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeHit>, CollisionError> {
        let _profile = engine_profile::span("collision.query", "shape_cast");
        let mut precise = engine_profile::Aggregate::new("collision.narrow", "shape_cast_tests");
        let mut broad = engine_profile::Aggregate::new("collision.broad", "bvh_traversal");
        validate_shape_cast(shape, matrix, displacement, options)?;
        self.ready()?;
        let shape = AffineShape::new(shape, matrix)?;
        let swept = shape
            .bounds
            .swept(displacement)
            .expanded(options.target_distance);
        let mut nearest: Option<ShapeHit> = None;
        for entry in candidates(
            (&self.bvh, &self.entries),
            self.config.max_query_candidates,
            |bounds| bounds.overlaps(swept),
            filter,
            &mut broad,
        ) {
            let entry = entry?;
            if let Some((fraction, point, normal, started_overlapping)) = precise.measure(|| {
                self.narrow
                    .cast(&shape, &entry.affine, displacement, options)
            })? {
                let hit = ShapeHit {
                    entity: entry.entity,
                    fraction,
                    point,
                    normal,
                    started_overlapping,
                };
                if nearest.is_none_or(|old| (fraction, entry.entity) < (old.fraction, old.entity)) {
                    nearest = Some(hit);
                }
            }
        }
        Ok(nearest)
    }

    fn validate_limit(&self, limit: usize) -> Result<(), CollisionError> {
        if limit > self.config.max_query_hits {
            Err(CollisionError::ResultBudget)
        } else {
            Ok(())
        }
    }

    fn detect(&mut self, events: &mut SensorEvents, tick: u64) {
        let _profile = engine_profile::span("collision", "sensor_detect");
        let mut precise =
            engine_profile::Aggregate::new("collision.narrow", "sensor_overlap_tests");
        let mut broad = engine_profile::Aggregate::new("collision.broad", "bvh_traversal");
        events.sequence = events.sequence.saturating_add(1);
        events.tick = Some(tick);
        events.events.clear();
        events.candidate_pairs = 0;
        self.next_pairs.clear();
        let result = (|| {
            self.ready()?;
            for (index, a) in self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.collider.is_sensor())
            {
                for other in self
                    .bvh
                    .query_profiled(|bounds| bounds.overlaps(a.affine.bounds), &mut broad)
                {
                    if index == other {
                        continue;
                    }
                    let b = &self.entries[other];
                    if b.collider.is_sensor() && other < index {
                        continue;
                    }
                    candidate(&mut events.candidate_pairs, self.config.max_candidate_pairs)?;
                    if !a.collider.layers().interacts(b.collider.layers()) {
                        continue;
                    }
                    if precise.measure(|| intersects(&a.affine, &b.affine))? {
                        if self.next_pairs.len() == self.config.max_sensor_pairs {
                            return Err(CollisionError::PairBudget);
                        }
                        self.next_pairs.push(SensorPair::new(a.entity, b.entity));
                    }
                }
            }
            Ok(())
        })();
        events.error = result.err();
        events.complete = events.error.is_none();
        if !events.complete {
            self.next_pairs.clear();
            return;
        }
        self.next_pairs.sort_unstable();
        let (mut old, mut new) = (0, 0);
        while old < events.pairs.len() || new < self.next_pairs.len() {
            let a = events.pairs.get(old).copied();
            let b = self.next_pairs.get(new).copied();
            match (a, b) {
                (Some(a), Some(b)) if a == b => {
                    old += 1;
                    new += 1;
                }
                (Some(a), Some(b)) if a < b => {
                    events.events.push(SensorEvent {
                        pair: a,
                        kind: SensorEventKind::Exit,
                    });
                    old += 1;
                }
                (Some(a), None) => {
                    events.events.push(SensorEvent {
                        pair: a,
                        kind: SensorEventKind::Exit,
                    });
                    old += 1;
                }
                (_, Some(b)) => {
                    events.events.push(SensorEvent {
                        pair: b,
                        kind: SensorEventKind::Enter,
                    });
                    new += 1;
                }
                (None, None) => break,
            }
        }
        std::mem::swap(&mut events.pairs, &mut self.next_pairs);
        self.next_pairs.clear();
        events.committed_tick = Some(tick);
    }
}

/// The entries a query visits: those of the tree whose bounds pass `test`,
/// in tree order, that `filter` accepts. Visiting more than `limit`
/// candidates, filtered ones included, is an error.
fn candidates<'a>(
    (bvh, entries): (&'a Bvh, &'a [Entry]),
    limit: usize,
    test: impl Fn(Bounds) -> bool + 'a,
    filter: QueryFilter,
    broad: &'a mut engine_profile::Aggregate,
) -> impl Iterator<Item = Result<Entry, CollisionError>> + 'a {
    let mut count = 0;
    bvh.query_profiled(test, broad).filter_map(move |index| {
        if let Err(error) = candidate(&mut count, limit) {
            return Some(Err(error));
        }
        let entry = entries[index];
        filter
            .accepts(entry.entity, entry.collider)
            .then_some(Ok(entry))
    })
}

fn candidate(count: &mut usize, limit: usize) -> Result<(), CollisionError> {
    if *count == limit {
        return Err(CollisionError::CandidateBudget);
    }
    *count += 1;
    Ok(())
}

/// Last attempted fixed-tick sensor batch, retained until the next fixed tick.
/// Failed attempts preserve pairs from the last complete tick and expose no events.
#[derive(Resource, Default)]
pub struct SensorEvents {
    events: Vec<SensorEvent>,
    pairs: Vec<SensorPair>,
    sequence: u64,
    tick: Option<u64>,
    committed_tick: Option<u64>,
    complete: bool,
    candidate_pairs: usize,
    error: Option<CollisionError>,
}
impl SensorEvents {
    pub fn events(&self) -> &[SensorEvent] {
        &self.events
    }
    pub fn pairs(&self) -> &[SensorPair] {
        &self.pairs
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn tick(&self) -> Option<u64> {
        self.tick
    }
    pub fn committed_tick(&self) -> Option<u64> {
        self.committed_tick
    }
    pub fn complete(&self) -> bool {
        self.complete
    }
    pub fn candidate_pairs(&self) -> usize {
        self.candidate_pairs
    }
    pub fn error(&self) -> Option<CollisionError> {
        self.error
    }
}

/// Refreshes authoritative matrices and the query snapshot only. Sensor event
/// history, presented matrices and clocks remain unchanged. No plugin is an error.
pub fn refresh_collision_world(world: &mut World) -> Result<CollisionReport, CollisionError> {
    if !world.contains_resource::<CollisionWorld>() {
        return Err(CollisionError::MissingPlugin);
    }
    propagate_transforms(world);
    world.init_resource::<DirectSyncState>();
    let report = world.resource_scope(|world, mut state: Mut<DirectSyncState>| {
        let mut sync = state
            .0
            .get_mut(world)
            .expect("CollisionPlugin resources must remain installed");
        sync.refresh()
    });
    Ok(report)
}

pub(crate) fn sync(mut sync: CollisionSync) {
    sync.refresh();
}

pub(crate) fn detect(
    mut collisions: ResMut<CollisionWorld>,
    mut events: ResMut<SensorEvents>,
    time: Res<FixedTime>,
) {
    collisions.detect(&mut events, time.tick);
}

#[cfg(test)]
mod tests;
