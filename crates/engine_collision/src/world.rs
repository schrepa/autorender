use std::any::Any;
use std::sync::{Arc, Mutex, PoisonError};

use bevy_ecs::prelude::*;
use engine_core::profile::{Aggregate, span};
use engine_core::{
    Error, FixedTime, FrameTime, GlobalTransforms, Section, Transform, TransformsSince,
};
use engine_math::collections::HashMap;
use engine_places::physics::{
    Collide, Collider, ColliderShape, Collision, CollisionRay, QueryFilter, RayHit, SLIDE_STEPS,
    SensorEvent, SensorEventKind, SensorPair, ShapeCastOptions, ShapeHit, Slid, SlideHit,
};
use engine_places::solids::{Bounds, Bvh, SolidField, SolidSource};
use glam::{Mat4, Vec2, Vec3};

use crate::{
    bvh::profiled,
    geometry::{AffineShape, Narrow, intersects},
    placed, terrain,
    types::*,
};

const SKIN: f32 = 0.002;
const STOPPED: f32 = 1.0e-10;

type ColliderData = (Entity, &'static Collider, Has<Transform>);
type Edited<'w, 's> = Query<'w, 's, (), Changed<Collider>>;
type Sources = &'static SolidSource;

fn read(
    globals: &GlobalTransforms,
    (entity, collider, local): (Entity, &Collider, bool),
) -> Option<AffineShape> {
    if !local {
        return None;
    }
    let matrix = globals.get(entity)?.matrix()?;
    AffineShape::new(collider.shape(), matrix).ok()
}

#[derive(Clone, Copy)]
struct Entry {
    entity: Entity,
    collider: Collider,
    affine: AffineShape,
}

#[derive(Default)]
pub struct CollisionWorld {
    config: CollisionConfig,
    held: HashMap<Entity, Option<u32>>,
    entries: Vec<Entry>,
    bounds: Vec<Bounds>,
    bvh: Bvh,
    moved: u64,
    disabled: u64,
    next_pairs: Vec<SensorPair>,
    report: CollisionReport,
    last_error: Option<Fault>,
    sources: Vec<Arc<SolidField>>,
    hits: Mutex<Vec<SlideHit>>,
    pub(crate) sensors: SensorEvents,
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
        self.held.get(&entity).map(|slot| match slot {
            Some(_) => ColliderStatus::Active,
            None => ColliderStatus::Invalid,
        })
    }

    pub(crate) fn ground_owners(&self) -> impl Iterator<Item = Entity> + '_ {
        terrain::owners(&self.sources)
    }

    fn ready(&self) -> Result<(), Fault> {
        if self.report.complete {
            Ok(())
        } else {
            Err(self.last_error.unwrap_or(Fault::IncompleteWorld))
        }
    }

    fn sync_ground(&mut self, sources: &Query<'_, '_, Sources>) {
        self.sources.clear();
        self.sources
            .extend(sources.iter().map(|source| Arc::clone(&source.field)));
        self.report.placed_colliders = self
            .sources
            .iter()
            .map(|source| source.solids.solids().len())
            .sum();
    }

    fn refresh(
        &mut self,
        config: CollisionConfig,
        (frame, tick): (u64, u64),
        (colliders, edited): (&Query<'_, '_, ColliderData>, &Edited<'_, '_>),
        globals: &GlobalTransforms,
    ) {
        let _profile = span("collision", "sync");
        self.report.frame = frame;
        self.report.tick = tick;
        let since = globals.changes_since(&mut self.moved);
        let disabled = std::mem::replace(&mut self.disabled, globals.switches().disabled);
        let settled = (self.report.complete || self.last_error.is_some())
            && config == self.config
            && disabled == self.disabled
            && colliders.iter().len() == self.report.colliders
            && edited.is_empty()
            && match since {
                TransformsSince::Same => true,
                TransformsSince::Listed(moved) => self.sync_moved(moved, colliders, globals),
                TransformsSince::All => false,
            };
        if !settled {
            self.sync_all(config, colliders, globals);
        }
    }

    fn sync_all(
        &mut self,
        config: CollisionConfig,
        colliders: &Query<'_, '_, ColliderData>,
        globals: &GlobalTransforms,
    ) {
        let _profile = span("collision", "cache_rebuild");
        let count = colliders.iter().len();
        let limit = config.colliders();
        let admitted = count.min(limit);
        self.config = config;
        self.report.colliders = count;
        self.report.invalid_colliders = 0;
        self.report.overflow_colliders = count.saturating_sub(limit);
        let held = std::mem::take(&mut self.held);
        self.held.reserve(admitted);
        self.entries.clear();
        self.bounds.clear();
        self.entries.reserve_exact(admitted);
        self.bounds.reserve_exact(admitted);
        let configuration_valid = config.check().is_ok();
        let mut same_leaves = true;
        for data in colliders.iter().take(limit) {
            let (entity, collider, _) = data;
            let slot = read(globals, data)
                .filter(|_| configuration_valid)
                .map(|affine| {
                    self.bounds.push(affine.bounds);
                    self.entries.push(Entry {
                        entity,
                        collider: *collider,
                        affine,
                    });
                    (self.entries.len() - 1) as u32
                });
            if slot.is_none() {
                self.report.invalid_colliders += 1;
            }
            same_leaves &= held.get(&entity) == Some(&slot);
            self.held.insert(entity, slot);
        }
        same_leaves &= held.len() == self.held.len();
        {
            let _profile = span("collision", "bvh_update");
            self.bvh.update(&self.bounds, same_leaves);
        }
        self.report.active_colliders = self.entries.len();
        self.report.bvh_nodes = self.bvh.node_count();
        self.report.cache_bytes = self.held.capacity() * size_of::<(Entity, Option<u32>)>()
            + self.entries.capacity() * size_of::<Entry>()
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
    }

    fn sync_moved(
        &mut self,
        moved: &[Entity],
        colliders: &Query<'_, '_, ColliderData>,
        globals: &GlobalTransforms,
    ) -> bool {
        for &entity in moved {
            let Some(&slot) = self.held.get(&entity) else {
                continue;
            };
            let Ok(data) = colliders.get(entity) else {
                return false;
            };
            match (slot, read(globals, data)) {
                (Some(slot), Some(affine)) => {
                    let slot = slot as usize;
                    self.entries[slot].affine = affine;
                    self.bounds[slot] = affine.bounds;
                    self.bvh.move_leaf(slot, affine.bounds);
                }
                (None, None) => {}
                (Some(_), None) | (None, Some(_)) => return false,
            }
        }
        if self.bvh.degraded() {
            let _profile = span("collision", "bvh_rebuild");
            self.bvh.rebuild(&self.bounds);
        }
        true
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
        for hit in [
            placed::ray(&self.sources, ray, filter),
            terrain::ray(&self.sources, ray, filter),
        ]
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
        let solid = placed::cast(&self.sources, &shape, displacement, filter, options)?;
        let ground = terrain::cast(&self.sources, &shape, displacement, filter, options)?;
        for hit in [solid, ground].into_iter().flatten() {
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
        self.report.slide_hits = self
            .hits
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
            .len();
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
                for other in profiled(
                    self.bvh.query(|bounds| bounds.overlaps(a.affine.bounds)),
                    &mut broad,
                ) {
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
    fn colliders(&self) -> Option<usize> {
        Some(self.config.colliders())
    }

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
            placed::ray_all(&self.sources, ray, filter, output);
            terrain::ray_all(&self.sources, ray, filter, output);
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
            placed::overlap(&self.sources, &shape, filter, output)?;
            terrain::overlap(&self.sources, &shape, filter, output)?;
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

    fn slide(
        &self,
        shape: ColliderShape,
        matrix: Mat4,
        motion: Vec3,
        filter: QueryFilter,
    ) -> Result<Slid, Error> {
        let options = ShapeCastOptions {
            stop_at_penetration: false,
            target_distance: 0.0,
        };
        let mut pose = matrix;
        let mut at = matrix.w_axis.truncate();
        let mut left = motion;
        let mut hits = [None; SLIDE_STEPS];
        for found in &mut hits {
            if left.length_squared() < STOPPED {
                break;
            }
            pose.w_axis = at.extend(1.0);
            let Some(hit) = self.cast_shape(shape, pose, left, filter, options)? else {
                at += left;
                break;
            };
            at += left * hit.fraction + hit.normal * SKIN;
            left *= 1.0 - hit.fraction;
            left -= hit.normal * left.dot(hit.normal);
            *found = Some(hit);
            self.hits
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(SlideHit {
                    mover: filter.exclude,
                    hit,
                });
        }
        Ok(Slid { position: at, hits })
    }

    fn slide_hits(&self) -> Vec<SlideHit> {
        self.hits
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn events(&self) -> &[SensorEvent] {
        &self.sensors.events
    }

    fn pairs(&self) -> &[SensorPair] {
        &self.sensors.pairs
    }

    fn holds(&self, at: Vec2) -> bool {
        self.sources.iter().any(|source| source.holds(at))
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
    profiled(bvh.query(test), broad).filter_map(move |index| {
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

pub(crate) fn clear_hits(mut collision: ResMut<Collision>) {
    installed(&mut collision)
        .hits
        .get_mut()
        .unwrap_or_else(PoisonError::into_inner)
        .clear();
}

pub(crate) fn sync_ground(sources: Query<Sources>, mut collision: ResMut<Collision>) {
    installed(&mut collision).sync_ground(&sources);
}

pub(crate) fn sync(
    (colliders, edited): (Query<ColliderData>, Edited),
    globals: Res<GlobalTransforms>,
    config: Res<CollisionConfig>,
    (frame, tick): (Res<FrameTime>, Res<FixedTime>),
    mut collision: ResMut<Collision>,
    mut stats: ResMut<CollisionStats>,
) {
    let world = installed(&mut collision);
    world.refresh(
        *config,
        (frame.frame, tick.tick),
        (&colliders, &edited),
        &globals,
    );
    stats.set_if_neq(world.stats());
}

pub(crate) fn refresh(
    (sources, colliders, edited): (Query<Sources>, Query<ColliderData>, Edited),
    globals: Res<GlobalTransforms>,
    config: Res<CollisionConfig>,
    (frame, tick): (Res<FrameTime>, Res<FixedTime>),
    mut collision: ResMut<Collision>,
    mut stats: ResMut<CollisionStats>,
) {
    let world = installed(&mut collision);
    world.sync_ground(&sources);
    world.refresh(
        *config,
        (frame.frame, tick.tick),
        (&colliders, &edited),
        &globals,
    );
    stats.set_if_neq(world.stats());
}

pub(crate) fn detect(mut collision: ResMut<Collision>, time: Res<FixedTime>) {
    installed(&mut collision).detect(time.tick);
}

pub(crate) fn forget(mut stats: ResMut<CollisionStats>) {
    stats.set_if_neq(CollisionStats::default());
}
