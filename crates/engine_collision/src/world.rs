use std::any::Any;

use bevy_ecs::prelude::*;
use engine_core::profile::{Aggregate, span};
use engine_core::{
    Error, FixedTime, FrameTime, GlobalTransforms, Section, Transform, TransformIndex,
};
use engine_math::collections::HashMap;
use engine_places::physics::{
    Collide, Collider, ColliderShape, Collision, CollisionRay, QueryFilter, RayHit, SensorEvent,
    SensorEventKind, SensorPair, ShapeCastOptions, ShapeHit,
};
use engine_places::solids::SolidSource;
use glam::{Mat4, Vec3};

use crate::{
    bvh::Bvh,
    geometry::{AffineShape, Bounds, Narrow, intersects},
    placed::PlacedShapes,
    terrain::TerrainFields,
    types::*,
};

type ColliderData = (Entity, &'static Collider, Has<Transform>);

pub(crate) type Sources = (Entity, Ref<'static, SolidSource>);

#[derive(Clone, Copy)]
struct Input {
    entity: Entity,
    collider: Collider,
    revision: Option<u64>,
    hint: TransformIndex,
    status: ColliderStatus,
    entry: u32,
}

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

pub struct CollisionWorld {
    config: CollisionConfig,
    inputs: Vec<Input>,
    statuses: HashMap<Entity, ColliderStatus>,
    entries: Vec<Entry>,
    previous_entries: Vec<Entry>,
    bounds: Vec<Bounds>,
    bvh: Bvh,
    next_pairs: Vec<SensorPair>,
    report: CollisionReport,
    last_error: Option<Fault>,
    sources: Vec<(Entity, u32)>,
    pub(crate) sensors: SensorEvents,
    pub(crate) terrain: TerrainFields,
    pub(crate) placed: PlacedShapes,
}

impl Default for CollisionWorld {
    fn default() -> Self {
        Self {
            config: CollisionConfig::default(),
            inputs: Vec::new(),
            statuses: HashMap::default(),
            entries: Vec::new(),
            previous_entries: Vec::new(),
            bounds: Vec::new(),
            bvh: Bvh::default(),
            next_pairs: Vec::new(),
            report: CollisionReport::default(),
            last_error: None,
            sources: Vec::new(),
            sensors: SensorEvents::default(),
            terrain: TerrainFields::default(),
            placed: PlacedShapes::default(),
        }
    }
}

pub(crate) fn installed(collision: &mut Collision) -> &mut CollisionWorld {
    if !(&*collision.0 as &dyn Any).is::<CollisionWorld>() {
        collision.0 = Box::new(CollisionWorld::default());
    }
    (&mut *collision.0 as &mut dyn Any)
        .downcast_mut::<CollisionWorld>()
        .expect("collision_world_installed")
}

pub(crate) fn world_of(collision: &Collision) -> Option<&CollisionWorld> {
    (&*collision.0 as &dyn Any).downcast_ref::<CollisionWorld>()
}

impl CollisionWorld {
    pub(crate) fn stats(&self) -> CollisionStats {
        CollisionStats::of(&self.report, &self.config)
    }
    pub(crate) fn collider_status(&self, entity: Entity) -> Option<ColliderStatus> {
        self.statuses.get(&entity).copied()
    }

    fn ready(&self) -> Result<(), Fault> {
        if self.report.complete {
            Ok(())
        } else {
            Err(self.last_error.unwrap_or(Fault::IncompleteWorld))
        }
    }

    fn sync_ground(&mut self, sources: &Query<'_, '_, Sources>) {
        let same = self.sources.len() == sources.iter().len()
            && sources
                .iter()
                .zip(&self.sources)
                .all(|((entity, source), &(seen, changed))| {
                    entity == seen && source.last_changed().get() == changed
                });
        if same {
            return;
        }
        self.sources.clear();
        self.sources.extend(
            sources
                .iter()
                .map(|(entity, source)| (entity, source.last_changed().get())),
        );
        self.terrain.sync(sources);
        if self.placed.sync(sources) {
            self.report.placed_colliders = self.placed.len();
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
        let _profile = span("collision", "sync");
        let count = colliders.iter().len();
        self.report.frame = frame;
        self.report.tick = tick;
        let limit = config.colliders();
        let admitted = count.min(limit);
        let unchanged = self.report.revision != 0
            && config == self.config
            && count == self.report.colliders
            && self.inputs.len() == admitted
            && colliders.iter().take(limit).zip(&mut self.inputs).all(
                |((entity, collider, local), old)| {
                    old.entity == entity
                        && old.collider == *collider
                        && old.revision != Some(u64::MAX)
                        && read_pose(globals, entity, local, &mut old.hint).1
                            == old.revision
                },
            );
        if unchanged {
            return self.report;
        }
        let _rebuild = span("collision", "cache_rebuild");
        let reuse = self.report.revision != 0 && config == self.config;
        self.config = config;
        self.report.revision = self.report.revision.saturating_add(1);
        self.report.colliders = count;
        self.report.invalid_colliders = 0;
        self.report.overflow_colliders = count.saturating_sub(limit);
        std::mem::swap(&mut self.entries, &mut self.previous_entries);
        self.entries.clear();
        self.bounds.clear();
        let mut statuses_changed = self.inputs.len() != admitted;
        self.inputs.truncate(admitted);
        self.inputs
            .reserve_exact(admitted.saturating_sub(self.inputs.len()));
        self.entries.reserve_exact(admitted);
        self.bounds.reserve_exact(admitted);
        let configuration_valid = config.check().is_ok();
        for (index, (entity, collider, local)) in colliders.iter().take(limit).enumerate() {
            let previous = self
                .inputs
                .get(index)
                .copied()
                .filter(|old| old.entity == entity);
            let mut hint = previous.map_or_else(TransformIndex::default, |old| old.hint);
            let (matrix, revision) = read_pose(globals, entity, local, &mut hint);
            let kept = previous
                .filter(|old| {
                    reuse
                        && old.status == ColliderStatus::Active
                        && old.collider == *collider
                        && old.revision == revision
                        && revision.is_some_and(|revision| revision != u64::MAX)
                })
                .and_then(|old| self.previous_entries.get(old.entry as usize))
                .map(|entry| entry.affine);
            let affine = kept.or_else(|| {
                matrix
                    .and_then(|matrix| AffineShape::new(collider.shape(), matrix).ok())
                    .filter(|_| configuration_valid)
            });
            let entry = self.entries.len() as u32;
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
                entry: if status == ColliderStatus::Active {
                    entry
                } else {
                    u32::MAX
                },
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
        self.bvh.update(&self.bounds, !statuses_changed);
        self.report.active_colliders = self.entries.len();
        self.report.bvh_nodes = self.bvh.len();
        self.report.cache_bytes = self.inputs.capacity() * size_of::<Input>()
            + (self.entries.capacity() + self.previous_entries.capacity()) * size_of::<Entry>()
            + self.bounds.capacity() * size_of::<Bounds>()
            + self.bvh.bytes();
        self.report.complete = configuration_valid
            && self.report.invalid_colliders == 0
            && self.report.overflow_colliders == 0;
        self.last_error = if !configuration_valid {
            Some(Fault::InvalidConfig)
        } else if !self.report.complete {
            Some(Fault::IncompleteWorld)
        } else {
            None
        };
        self.report
    }

    fn nearest_ray(&self, ray: CollisionRay, filter: QueryFilter) -> Result<Option<RayHit>, Fault> {
        let _profile = span("collision.query", "raycast");
        let mut precise = Aggregate::new("collision.narrow", "ray_tests");
        let mut broad = Aggregate::new("collision.broad", "bvh_traversal");
        let ray = normalized(ray)?;
        self.ready()?;
        let mut nearest: Option<RayHit> = None;
        for entry in candidates(
            (&self.bvh, &self.entries),
            self.config.query_candidates(),
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
                    placed: None,
                };
                if nearest.is_none_or(|old| (distance, entry.entity) < (old.distance, old.entity)) {
                    nearest = Some(hit);
                }
            }
        }
        for hit in [self.placed.ray(ray, filter), self.terrain.ray(ray, filter)]
            .into_iter()
            .flatten()
        {
            if nearest.is_none_or(|old| (hit.distance, hit.entity) < (old.distance, old.entity)) {
                nearest = Some(hit);
            }
        }
        Ok(nearest)
    }

    fn nearest_cast(
        &self,
        shape: ColliderShape,
        matrix: Mat4,
        displacement: Vec3,
        filter: QueryFilter,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeHit>, Fault> {
        let _profile = span("collision.query", "shape_cast");
        let mut precise = Aggregate::new("collision.narrow", "shape_cast_tests");
        let mut broad = Aggregate::new("collision.broad", "bvh_traversal");
        check_shape_cast(shape, matrix, displacement, options)?;
        self.ready()?;
        let shape = AffineShape::new(shape, matrix)?;
        let swept = shape
            .bounds
            .swept(displacement)
            .expanded(options.target_distance);
        let mut nearest: Option<ShapeHit> = None;
        for entry in candidates(
            (&self.bvh, &self.entries),
            self.config.query_candidates(),
            |bounds| bounds.overlaps(swept),
            filter,
            &mut broad,
        ) {
            let entry = entry?;
            if let Some((fraction, point, normal, started_overlapping)) =
                precise.measure(|| Narrow::cast(&shape, &entry.affine, displacement, options))?
            {
                let hit = ShapeHit {
                    entity: entry.entity,
                    fraction,
                    point,
                    normal,
                    started_overlapping,
                    placed: None,
                };
                if nearest.is_none_or(|old| (fraction, entry.entity) < (old.fraction, old.entity)) {
                    nearest = Some(hit);
                }
            }
        }
        let placed = self.placed.cast(&shape, displacement, filter, options)?;
        let terrain = self.terrain.cast(&shape, displacement, filter, options)?;
        for hit in [placed, terrain].into_iter().flatten() {
            if nearest.is_none_or(|old| (hit.fraction, hit.entity) < (old.fraction, old.entity)) {
                nearest = Some(hit);
            }
        }
        Ok(nearest)
    }

    fn validate_limit(&self, limit: usize) -> Result<(), Fault> {
        if limit > self.config.query_hits() {
            Err(Fault::ResultBudget)
        } else {
            Ok(())
        }
    }

    fn detect(&mut self, tick: u64) {
        let mut events = std::mem::take(&mut self.sensors);
        self.detect_into(&mut events, tick);
        self.sensors = events;
    }

    fn detect_into(&mut self, events: &mut SensorEvents, tick: u64) {
        let _profile = span("collision", "sensor_detect");
        let mut precise = Aggregate::new("collision.narrow", "sensor_overlap_tests");
        let mut broad = Aggregate::new("collision.broad", "bvh_traversal");
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
                    candidate(&mut events.candidate_pairs, self.config.candidate_pairs())?;
                    if !a.collider.layers().interacts(b.collider.layers()) {
                        continue;
                    }
                    if precise.measure(|| intersects(&a.affine, &b.affine))? {
                        if self.next_pairs.len() == self.config.sensor_pairs() {
                            return Err(Fault::PairBudget);
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

impl Collide for CollisionWorld {
    fn raycast(&self, ray: CollisionRay, filter: QueryFilter) -> Result<Option<RayHit>, Error> {
        self.nearest_ray(ray, filter).map_err(Fault::error)
    }

    fn raycast_all(
        &self,
        ray: CollisionRay,
        filter: QueryFilter,
        output: &mut Vec<RayHit>,
        limit: usize,
    ) -> Result<(), Error> {
        let _profile = span("collision.query", "raycast_all");
        let mut precise = Aggregate::new("collision.narrow", "ray_tests");
        let mut broad = Aggregate::new("collision.broad", "bvh_traversal");
        output.clear();
        let result = (|| {
            let ray = normalized(ray)?;
            self.ready()?;
            self.validate_limit(limit)?;
            for entry in candidates(
                (&self.bvh, &self.entries),
                self.config.query_candidates(),
                |bounds| bounds.ray(ray),
                filter,
                &mut broad,
            ) {
                let entry = entry?;
                if let Some((distance, normal, started_inside)) =
                    precise.measure(|| entry.affine.ray(ray))
                {
                    if output.len() == limit {
                        return Err(Fault::ResultBudget);
                    }
                    output.push(RayHit {
                        entity: entry.entity,
                        distance,
                        point: ray.origin + ray.direction * distance,
                        normal,
                        started_inside,
                        placed: None,
                    });
                }
            }
            self.placed.ray_all(ray, filter, output);
            self.terrain.ray_all(ray, filter, output);
            if output.len() > limit {
                return Err(Fault::ResultBudget);
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
        result.map_err(Fault::error)
    }

    fn overlap(
        &self,
        shape: ColliderShape,
        matrix: Mat4,
        filter: QueryFilter,
        output: &mut Vec<Entity>,
        limit: usize,
    ) -> Result<(), Error> {
        let _profile = span("collision.query", "overlap");
        let mut precise = Aggregate::new("collision.narrow", "overlap_tests");
        let mut broad = Aggregate::new("collision.broad", "bvh_traversal");
        output.clear();
        let result = (|| {
            let shape = AffineShape::new(shape, matrix)?;
            self.ready()?;
            self.validate_limit(limit)?;
            for entry in candidates(
                (&self.bvh, &self.entries),
                self.config.query_candidates(),
                |bounds| bounds.overlaps(shape.bounds),
                filter,
                &mut broad,
            ) {
                let entry = entry?;
                if precise.measure(|| intersects(&shape, &entry.affine))? {
                    if output.len() == limit {
                        return Err(Fault::ResultBudget);
                    }
                    output.push(entry.entity);
                }
            }
            self.placed.overlap(&shape, filter, output)?;
            self.terrain.overlap(&shape, filter, output)?;
            if output.len() > limit {
                return Err(Fault::ResultBudget);
            }
            output.sort_unstable();
            Ok(())
        })();
        if result.is_err() {
            output.clear();
        }
        result.map_err(Fault::error)
    }

    fn cast_shape(
        &self,
        shape: ColliderShape,
        matrix: Mat4,
        displacement: Vec3,
        filter: QueryFilter,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeHit>, Error> {
        self.nearest_cast(shape, matrix, displacement, filter, options)
            .map_err(Fault::error)
    }

    fn events(&self) -> &[SensorEvent] {
        &self.sensors.events
    }

    fn pairs(&self) -> &[SensorPair] {
        &self.sensors.pairs
    }
}

fn candidates<'a>(
    (bvh, entries): (&'a Bvh, &'a [Entry]),
    limit: usize,
    test: impl Fn(Bounds) -> bool + 'a,
    filter: QueryFilter,
    broad: &'a mut Aggregate,
) -> impl Iterator<Item = Result<Entry, Fault>> + 'a {
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

fn candidate(count: &mut usize, limit: usize) -> Result<(), Fault> {
    if *count == limit {
        return Err(Fault::CandidateBudget);
    }
    *count += 1;
    Ok(())
}

#[derive(Default)]
pub(crate) struct SensorEvents {
    events: Vec<SensorEvent>,
    pairs: Vec<SensorPair>,
    sequence: u64,
    tick: Option<u64>,
    committed_tick: Option<u64>,
    complete: bool,
    candidate_pairs: usize,
    error: Option<Fault>,
}

impl SensorEvents {
    pub(crate) fn events(&self) -> &[SensorEvent] {
        &self.events
    }
    pub(crate) fn pairs(&self) -> &[SensorPair] {
        &self.pairs
    }
    pub(crate) fn sequence(&self) -> u64 {
        self.sequence
    }
    pub(crate) fn tick(&self) -> Option<u64> {
        self.tick
    }
    pub(crate) fn committed_tick(&self) -> Option<u64> {
        self.committed_tick
    }
    pub(crate) fn complete(&self) -> bool {
        self.complete
    }
    pub(crate) fn candidate_pairs(&self) -> usize {
        self.candidate_pairs
    }
    pub(crate) fn error(&self) -> Option<Error> {
        self.error.map(Fault::error)
    }
}

pub(crate) fn sync_ground(sources: Query<Sources>, mut collision: ResMut<Collision>) {
    installed(&mut collision).sync_ground(&sources);
}

pub(crate) fn sync(
    colliders: Query<ColliderData>,
    globals: Res<GlobalTransforms>,
    config: Res<CollisionConfig>,
    (frame, tick): (Res<FrameTime>, Res<FixedTime>),
    mut collision: ResMut<Collision>,
    mut stats: ResMut<CollisionStats>,
) {
    let world = installed(&mut collision);
    world.refresh(*config, frame.frame, tick.tick, &colliders, &globals);
    stats.set_if_neq(world.stats());
}

pub(crate) fn refresh(
    (sources, colliders): (Query<Sources>, Query<ColliderData>),
    globals: Res<GlobalTransforms>,
    config: Res<CollisionConfig>,
    (frame, tick): (Res<FrameTime>, Res<FixedTime>),
    mut collision: ResMut<Collision>,
    mut stats: ResMut<CollisionStats>,
) {
    let world = installed(&mut collision);
    world.sync_ground(&sources);
    world.refresh(*config, frame.frame, tick.tick, &colliders, &globals);
    stats.set_if_neq(world.stats());
}

pub(crate) fn detect(mut collision: ResMut<Collision>, time: Res<FixedTime>) {
    installed(&mut collision).detect(time.tick);
}

pub(crate) fn forget(mut stats: ResMut<CollisionStats>) {
    stats.set_if_neq(CollisionStats::default());
}
