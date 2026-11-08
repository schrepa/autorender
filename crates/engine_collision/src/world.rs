use std::any::Any;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use engine_core::profile::{Aggregate, span};
use engine_core::{
    Cursor, Error, Events, Feed, FixedTime, FrameTime, GlobalTransforms, Section, Since, Transform,
};
use engine_math::collections::HashMap;
use engine_places::queries::{CastFilter, CastHit, CastShape, SceneQueries, SceneQuery};
use engine_places_simulation::physics::{
    Collide, Collider, ColliderShape, ColliderSolid, Collision, CollisionLayers, CollisionRay,
    QueryFilter, RayHit, SLIDE_STEPS, SensorEvent, SensorEventKind, SensorFilter, SensorPair,
    ShapeCastOptions, ShapeHit, Slid,
};
use engine_places_simulation::solids::{Bounds, Bvh, SolidField, SolidSource};
use glam::{Mat4, Quat, Vec2, Vec3};

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

struct Solids {
    entries: Arc<Vec<Entry>>,
    bvh: Arc<Bvh>,
    sources: Vec<Arc<SolidField>>,
    limit: usize,
    ready: Result<(), Fault>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Synced {
    #[default]
    Never,
    Complete,
    Incomplete(Fault),
}

impl Synced {
    fn of(valid: bool, report: &CollisionReport) -> Self {
        if !valid {
            Self::Incomplete(Fault::InvalidConfig)
        } else if report.invalid_colliders > 0 || report.overflow_colliders > 0 {
            Self::Incomplete(Fault::IncompleteWorld)
        } else {
            Self::Complete
        }
    }

    fn ready(self) -> Result<(), Fault> {
        match self {
            Self::Complete => Ok(()),
            Self::Incomplete(fault) => Err(fault),
            Self::Never => Err(Fault::IncompleteWorld),
        }
    }
}

#[derive(Default)]
pub struct CollisionWorld {
    config: CollisionConfig,
    held: HashMap<Entity, Option<u32>>,
    entries: Arc<Vec<Entry>>,
    bounds: Vec<Bounds>,
    bvh: Arc<Bvh>,
    moved: Cursor,
    words: u64,
    next_pairs: Vec<SensorPair>,
    report: CollisionReport,
    synced: Synced,
    sources: Vec<Arc<SolidField>>,
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
        CollisionStats::of(&self.report, &self.config, self.synced == Synced::Complete)
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

    fn sweep(&self) -> Solids {
        Solids {
            entries: Arc::clone(&self.entries),
            bvh: Arc::clone(&self.bvh),
            sources: self.sources.clone(),
            limit: self.config.query_candidates(),
            ready: self.ready(),
        }
    }

    fn ready(&self) -> Result<(), Fault> {
        self.synced.ready()
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
        let words = std::mem::replace(&mut self.words, globals.words_revision());
        let settled = self.synced != Synced::Never
            && config == self.config
            && words == self.words
            && colliders.iter().len() == self.report.colliders
            && edited.is_empty()
            && match since {
                Since::Same => true,
                Since::Listed(moved) => self.sync_moved(moved, colliders, globals),
                Since::All => false,
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
        let entries = Arc::make_mut(&mut self.entries);
        entries.clear();
        self.bounds.clear();
        entries.reserve_exact(admitted);
        self.bounds.reserve_exact(admitted);
        let configuration_valid = config.check().is_ok();
        let mut same_leaves = true;
        let active = colliders
            .iter()
            .filter(|&(entity, ..)| globals.active(entity));
        for data in active.take(limit) {
            let (entity, collider, _) = data;
            let slot = read(globals, data)
                .filter(|_| configuration_valid)
                .map(|affine| {
                    self.bounds.push(affine.bounds);
                    entries.push(Entry {
                        entity,
                        collider: *collider,
                        affine,
                    });
                    (entries.len() - 1) as u32
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
            Arc::make_mut(&mut self.bvh)
                .update(&self.bounds, same_leaves);
        }
        self.report.active_colliders = self.entries.len();
        self.report.bvh_nodes = self.bvh.node_count();
        self.report.cache_bytes = self.held.capacity() * size_of::<(Entity, Option<u32>)>()
            + self.entries.capacity() * size_of::<Entry>()
            + self.bounds.capacity() * size_of::<Bounds>()
            + self.bvh.bytes();
        self.synced = Synced::of(configuration_valid, &self.report);
    }

    fn sync_moved(
        &mut self,
        moved: &[u32],
        colliders: &Query<'_, '_, ColliderData>,
        globals: &GlobalTransforms,
    ) -> bool {
        for &index in moved {
            let Some(entity) = globals.entity_at(index) else {
                continue;
            };
            let Some(&slot) = self.held.get(&entity) else {
                continue;
            };
            let Ok(data) = colliders.get(entity) else {
                return false;
            };
            match (slot, read(globals, data)) {
                (Some(slot), Some(affine)) => {
                    let slot = slot as usize;
                    Arc::make_mut(&mut self.entries)[slot].affine = affine;
                    self.bounds[slot] = affine.bounds;
                    Arc::make_mut(&mut self.bvh).move_leaf(slot, affine.bounds);
                }
                (None, None) => {}
                (Some(_), None) | (None, Some(_)) => return false,
            }
        }
        if self.bvh.degraded() {
            let _profile = span("collision", "bvh_rebuild");
            Arc::make_mut(&mut self.bvh).rebuild(&self.bounds);
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

    fn validate_limit(&self, limit: usize) -> Result<(), Fault> {
        if limit > self.config.query_hits() {
            Err(Fault::ResultBudget)
        } else {
            Ok(())
        }
    }

    fn detect(&mut self, tick: u64) {
        let mut events = std::mem::take(&mut self.sensors);
        events.events.resize(self.config.sensor_feed as usize);
        self.detect_into(&mut events, tick);
        self.sensors = events;
    }

    fn detect_into(&mut self, events: &mut SensorEvents, tick: u64) {
        let _profile = span("collision", "sensor_detect");
        let mut precise = Aggregate::new("collision.narrow", "sensor_overlap_tests");
        let mut broad = Aggregate::new("collision.broad", "bvh_traversal");
        events.sequence = events.sequence.saturating_add(1);
        events.tick = Some(tick);
        events.last = 0;
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
        if events.error.is_some() {
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
                    events.push(SensorEvent {
                        pair: a,
                        kind: SensorEventKind::Exit,
                    });
                    old += 1;
                }
                (Some(a), None) => {
                    events.push(SensorEvent {
                        pair: a,
                        kind: SensorEventKind::Exit,
                    });
                    old += 1;
                }
                (_, Some(b)) => {
                    events.push(SensorEvent {
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
    fn grounds(&self) -> &[Arc<SolidField>] {
        &self.sources
    }

    fn solids(&self, min: Vec3, max: Vec3, found: &mut Vec<ColliderSolid>) {
        let bounds = Bounds { min, max };
        found.extend(
            self.bvh
                .query(move |node| node.overlaps(bounds))
                .map(|index| self.entries[index])
                .filter(|entry| {
                    !entry.collider.is_sensor() && entry.affine.bounds.overlaps(bounds)
                })
                .map(|entry| ColliderSolid {
                    entity: entry.entity,
                    layers: entry.collider.layers(),
                    shape: entry.affine.shape,
                    center: entry.affine.center,
                    axes: entry.affine.basis,
                }),
        );
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
        cast(
            (&self.bvh, &self.entries, &self.sources),
            (self.config.query_candidates(), self.ready()),
            (shape, matrix, displacement),
            filter,
            options,
        )
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
        }
        Ok(Slid { position: at, hits })
    }

    fn events(&self) -> &Feed<SensorEvent> {
        &self.sensors.events
    }

    fn pairs(&self) -> &[SensorPair] {
        &self.sensors.pairs
    }

    fn holds(&self, at: Vec2) -> bool {
        self.sources.iter().any(|source| source.holds(at))
    }
}

impl SceneQuery for Solids {
    fn cast(
        &self,
        shape: CastShape,
        from: Vec3,
        along: Vec3,
        filter: CastFilter,
    ) -> Option<CastHit> {
        let (shape, rotation) = match shape {
            CastShape::Sphere { radius } => (ColliderShape::Sphere { radius }, Quat::IDENTITY),
            CastShape::Box {
                half_extents,
                rotation,
            } => (ColliderShape::Cuboid { half_extents }, rotation),
        };
        let filter = QueryFilter {
            layers: CollisionLayers {
                filter: filter.layers,
                ..CollisionLayers::default()
            },
            sensors: SensorFilter::Exclude,
            exclude: filter.ignore,
        };
        let options = ShapeCastOptions {
            stop_at_penetration: false,
            target_distance: 0.0,
        };
        cast(
            (&self.bvh, &self.entries, &self.sources),
            (self.limit, self.ready),
            (shape, Mat4::from_rotation_translation(rotation, from), along),
            filter,
            options,
        )
        .ok()
        .flatten()
        .map(|hit| CastHit {
            entity: hit.entity,
            distance: hit.fraction * along.length(),
            point: hit.point,
            normal: hit.normal,
        })
    }
}

fn cast(
    (bvh, entries, sources): (&Bvh, &[Entry], &[Arc<SolidField>]),
    (limit, ready): (usize, Result<(), Fault>),
    (shape, matrix, displacement): (ColliderShape, Mat4, Vec3),
    filter: QueryFilter,
    options: ShapeCastOptions,
) -> Result<Option<ShapeHit>, Fault> {
    let _profile = span("collision.query", "shape_cast");
    let mut precise = Aggregate::new("collision.narrow", "shape_cast_tests");
    let mut broad = Aggregate::new("collision.broad", "bvh_traversal");
    check_shape_cast(shape, matrix, displacement, options)?;
    ready?;
    let shape = AffineShape::new(shape, matrix)?;
    let swept = shape
        .bounds
        .swept(displacement)
        .expanded(options.target_distance);
    let mut nearest: Option<ShapeHit> = None;
    for entry in candidates(
        (bvh, entries),
        limit,
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
    let solid = placed::cast(sources, &shape, displacement, filter, options)?;
    let ground = terrain::cast(sources, &shape, displacement, filter, options)?;
    for hit in [solid, ground].into_iter().flatten() {
        if nearest.is_none_or(|old| (hit.fraction, hit.entity) < (old.fraction, old.entity)) {
            nearest = Some(hit);
        }
    }
    Ok(nearest)
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

pub(crate) struct SensorEvents {
    events: Feed<SensorEvent>,
    last: usize,
    pairs: Vec<SensorPair>,
    sequence: u64,
    tick: Option<u64>,
    committed_tick: Option<u64>,
    candidate_pairs: usize,
    error: Option<Fault>,
}

impl Default for SensorEvents {
    fn default() -> Self {
        Self {
            events: Feed::new(1),
            last: 0,
            pairs: Vec::new(),
            sequence: 0,
            tick: None,
            committed_tick: None,
            candidate_pairs: 0,
            error: None,
        }
    }
}

impl SensorEvents {
    fn push(&mut self, event: SensorEvent) {
        self.events.push(event);
        self.last += 1;
    }

    pub(crate) fn last(&self) -> &[SensorEvent] {
        let mut cursor = Cursor::default();
        match self.events.events(&mut cursor) {
            Events::Listed(all) => &all[all.len().saturating_sub(self.last)..],
            Events::Missed(_) => &[],
        }
    }

    pub(crate) fn feed(&self) -> &Feed<SensorEvent> {
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
        self.tick.is_some() && self.error.is_none()
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
    (colliders, edited): (Query<ColliderData>, Edited),
    globals: Res<GlobalTransforms>,
    config: Res<CollisionConfig>,
    (frame, tick): (Res<FrameTime>, Res<FixedTime>),
    mut collision: ResMut<Collision>,
    (mut stats, mut queries): (ResMut<CollisionStats>, ResMut<SceneQueries>),
) {
    let world = installed(&mut collision);
    world.refresh(
        *config,
        (frame.frame, tick.tick),
        (&colliders, &edited),
        &globals,
    );
    stats.set_if_neq(world.stats());
    queries.0 = Arc::new(world.sweep());
}

#[derive(SystemParam)]
pub(crate) struct Fresh<'w, 's> {
    sources: Query<'w, 's, Sources>,
    colliders: Query<'w, 's, ColliderData>,
    edited: Edited<'w, 's>,
    globals: Res<'w, GlobalTransforms>,
    config: Res<'w, CollisionConfig>,
    frame: Res<'w, FrameTime>,
    tick: Res<'w, FixedTime>,
    collision: ResMut<'w, Collision>,
    stats: ResMut<'w, CollisionStats>,
    queries: ResMut<'w, SceneQueries>,
}

impl Fresh<'_, '_> {
    pub(crate) fn world(&mut self) -> &CollisionWorld {
        let world = installed(&mut self.collision);
        world.sync_ground(&self.sources);
        world.refresh(
            *self.config,
            (self.frame.frame, self.tick.tick),
            (&self.colliders, &self.edited),
            &self.globals,
        );
        self.stats.set_if_neq(world.stats());
        self.queries.0 = Arc::new(world.sweep());
        world
    }
}

pub(crate) fn detect(mut collision: ResMut<Collision>, time: Res<FixedTime>) {
    installed(&mut collision).detect(time.tick);
}
