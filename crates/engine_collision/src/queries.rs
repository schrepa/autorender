use bevy_ecs::prelude::{Entity, World};
use engine_core::bounds::{self, MAX_SCAN_ENTITIES, default_limit};
use engine_core::{Error, ErrorCode, Operation, OperationScene};
use engine_settings::rule::range;
#[cfg(feature = "schema")]
use engine_settings::stated;
use engine_settings::{Rule, Setting};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::results::{
    CastHit, CollisionEventsResult, HitEntity, OverlapResult, PlacedHit, RayHit, RaycastResult,
    SensorEventKindView, SensorEventRow, ShapeCastResult,
};
use crate::{
    ColliderShape, CollisionError, CollisionLayers, CollisionRay, CollisionWorld,
    CollisionWorldView, MAX_COLLIDER_EXTENT, MIN_COLLIDER_EXTENT, PlacedPart, QueryFilter,
    SensorEvent, SensorEventKind, SensorEvents, SensorFilter, ShapeCastOptions,
    refresh_collision_world, validate_collider_transform, validate_shape_cast,
};

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct RaycastQuery {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
    #[cfg_attr(feature = "schema", schemars(transform = stated(bounds::DISTANCE)))]
    pub max_distance: f64,
    #[serde(default)]
    pub filter: CollisionFilterRequest,
}

impl Operation for RaycastQuery {
    const NAME: &'static str = "raycast";
    const CONTROL: bool = false;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        scan(world)?;
        let Self {
            origin,
            direction,
            max_distance,
            filter,
        } = request;
        let ray = ray(origin, direction, max_distance)?;
        let filter = query_filter(world, &filter)?;
        refresh_collision_world(world).map_err(CollisionError::query_error)?;
        let hit = world
            .resource_mut::<CollisionWorld>()
            .raycast(ray, filter)
            .map_err(CollisionError::query_error)?;
        let world = &*world;
        Ok(reply(RaycastResult {
            snapshot: CollisionWorldView::of(world.resource::<CollisionWorld>()),
            complete: true,
            hit: hit.map(|hit| RayHit {
                entity: hit_entity(world, hit.entity),
                distance: hit.distance,
                point: hit.point.to_array(),
                normal: hit.normal.to_array(),
                started_inside: hit.started_inside,
                placed: hit.placed.map(placed),
            }),
        }))
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct OverlapQuery {
    pub shape: CollisionShapeRequest,
    pub position: [f64; 3],
    #[serde(default = "identity_rotation")]
    pub rotation: [f64; 4],
    #[serde(default)]
    pub filter: CollisionFilterRequest,
    #[serde(default = "default_limit")]
    #[cfg_attr(feature = "schema", schemars(transform = stated(bounds::PAGE)))]
    pub limit: usize,
}

impl Operation for OverlapQuery {
    const NAME: &'static str = "overlap";
    const CONTROL: bool = false;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        scan(world)?;
        let Self {
            shape: source,
            position,
            rotation,
            filter,
            limit,
        } = request;
        page(limit)?;
        let shape = shape(&source)?;
        let matrix = pose(position, rotation)?;
        validate_collider_transform(shape, matrix).map_err(CollisionError::query_error)?;
        let filter = query_filter(world, &filter)?;
        refresh_collision_world(world).map_err(CollisionError::query_error)?;
        let mut entities = Vec::new();
        world
            .resource_mut::<CollisionWorld>()
            .overlap(shape, matrix, filter, &mut entities, limit)
            .map_err(CollisionError::query_error)?;
        let world = &*world;
        let mut hits: Vec<_> = entities
            .into_iter()
            .map(|entity| hit_entity(world, entity))
            .collect();
        hits.sort_by(|a, b| {
            (a.scene.as_deref(), a.id.as_deref()).cmp(&(b.scene.as_deref(), b.id.as_deref()))
        });
        Ok(reply(OverlapResult {
            snapshot: CollisionWorldView::of(world.resource::<CollisionWorld>()),
            complete: true,
            total: hits.len(),
            hits,
        }))
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct ShapeCastQuery {
    pub shape: CollisionShapeRequest,
    pub position: [f64; 3],
    #[serde(default = "identity_rotation")]
    pub rotation: [f64; 4],
    pub direction: [f64; 3],
    #[cfg_attr(feature = "schema", schemars(transform = stated(bounds::DISTANCE)))]
    pub max_distance: f64,
    #[serde(default)]
    pub filter: CollisionFilterRequest,
}

impl Operation for ShapeCastQuery {
    const NAME: &'static str = "shape_cast";
    const CONTROL: bool = false;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        scan(world)?;
        let Self {
            shape: source,
            position,
            rotation,
            direction,
            max_distance,
            filter,
        } = request;
        let shape = shape(&source)?;
        let matrix = pose(position, rotation)?;
        let ray = ray(position, direction, max_distance)?;
        let displacement = ray.direction * ray.max_distance;
        let options = ShapeCastOptions::default();
        validate_shape_cast(shape, matrix, displacement, options)
            .map_err(CollisionError::query_error)?;
        let filter = query_filter(world, &filter)?;
        refresh_collision_world(world).map_err(CollisionError::query_error)?;
        let world = &*world;
        let hit = world
            .resource::<CollisionWorld>()
            .cast_shape(shape, matrix, displacement, filter, options)
            .map_err(CollisionError::query_error)?;
        Ok(reply(ShapeCastResult {
            snapshot: CollisionWorldView::of(world.resource::<CollisionWorld>()),
            complete: true,
            hit: hit.map(|hit| CastHit {
                entity: hit_entity(world, hit.entity),
                fraction: hit.fraction,
                distance: hit.fraction * ray.max_distance,
                point: hit.point.to_array(),
                normal: hit.normal.to_array(),
                started_overlapping: hit.started_overlapping,
                placed: hit.placed.map(placed),
            }),
        }))
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct CollisionEventsQuery {
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default)]
    #[cfg_attr(feature = "schema", schemars(transform = stated(bounds::EVENT_OFFSET)))]
    pub offset: usize,
    #[serde(default = "default_limit")]
    #[cfg_attr(feature = "schema", schemars(transform = stated(bounds::PAGE)))]
    pub limit: usize,
}

impl Operation for CollisionEventsQuery {
    const NAME: &'static str = "collision_events";
    const CONTROL: bool = false;
    type Request = Self;

    fn run(world: &mut World, request: Self) -> Result<Value, Error> {
        scan(world)?;
        let Self {
            entity,
            offset,
            limit,
        } = request;
        page(limit)?;
        bounded(
            offset,
            bounds::EVENT_OFFSET,
            ErrorCode::InvalidPage,
            "$.offset",
        )?;
        let world = &*world;
        let entity = entity
            .as_deref()
            .map(|id| resolve_entity(world, id, false, "$.entity"))
            .transpose()?;
        let events = world
            .get_resource::<SensorEvents>()
            .ok_or_else(|| CollisionError::MissingPlugin.query_error())?;
        let matches = |event: &&SensorEvent| {
            entity.is_none_or(|entity| event.pair.first == entity || event.pair.second == entity)
        };
        let total = events.events().iter().filter(matches).count();
        let rows: Vec<_> = events
            .events()
            .iter()
            .filter(matches)
            .skip(offset)
            .take(limit)
            .map(|event| SensorEventRow {
                kind: match event.kind {
                    SensorEventKind::Enter => SensorEventKindView::Enter,
                    SensorEventKind::Exit => SensorEventKindView::Exit,
                },
                first: hit_entity(world, event.pair.first),
                second: hit_entity(world, event.pair.second),
            })
            .collect();
        let next = offset.saturating_add(rows.len());
        Ok(reply(CollisionEventsResult {
            sensor_only: true,
            sequence: events.sequence().to_string(),
            tick: events.tick().map(|tick| tick.to_string()),
            committed_tick: events.committed_tick().map(|tick| tick.to_string()),
            complete: events.complete(),
            candidate_pairs: events.candidate_pairs(),
            active_pairs: events.pairs().len(),
            error: events.error().map(CollisionError::query_error),
            total,
            events: rows,
            next_offset: (next < total).then_some(next),
        }))
    }

    #[cfg(feature = "schema")]
    fn schema() -> schemars::Schema {
        schemars::schema_for!(Self)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CollisionShapeRequest {
    Cuboid {
        #[cfg_attr(
            feature = "schema",
            schemars(transform = stated(range(MIN_COLLIDER_EXTENT, MAX_COLLIDER_EXTENT)))
        )]
        half_extents: [f64; 3],
    },
    Sphere {
        #[cfg_attr(
            feature = "schema",
            schemars(transform = stated(range(MIN_COLLIDER_EXTENT, MAX_COLLIDER_EXTENT)))
        )]
        radius: f64,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(default, deny_unknown_fields)]
pub struct CollisionFilterRequest {
    pub memberships: u32,
    pub filter: u32,
    pub sensors: CollisionSensorFilter,
    pub exclude: Option<String>,
}

impl Default for CollisionFilterRequest {
    fn default() -> Self {
        Self {
            memberships: 1,
            filter: u32::MAX,
            sensors: CollisionSensorFilter::Include,
            exclude: None,
        }
    }
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum CollisionSensorFilter {
    #[default]
    Include,
    Exclude,
    Only,
}

fn identity_rotation() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

pub(crate) fn reply(result: impl Serialize) -> Value {
    serde_json::to_value(result).expect("results_serialize")
}

pub(crate) fn bounded<T: Setting>(
    value: T,
    rule: Rule,
    code: ErrorCode,
    path: &str,
) -> Result<T, Error> {
    match value.check("value", rule) {
        Ok(()) => Ok(value),
        Err(invalid) => Err(Error::new(code, path, invalid.rule)),
    }
}

fn scan(world: &World) -> Result<(), Error> {
    if world
        .get_resource::<OperationScene>()
        .map(OperationScene::entity_count)
        .is_some_and(|count| count > MAX_SCAN_ENTITIES)
    {
        return Err(Error::new(
            ErrorCode::ScanLimit,
            "$.scene",
            "collision identity lookup admits at most 10000 authored entities",
        ));
    }
    Ok(())
}

fn hit_entity(world: &World, entity: Entity) -> HitEntity {
    let scene = world.get_resource::<OperationScene>();
    let authored = scene.and_then(|scene| scene.authored(world, entity));
    let in_active_scene = scene
        .zip(authored.as_ref())
        .is_some_and(|(scene, authored)| scene.holds(authored, entity));
    HitEntity {
        id: authored.as_ref().map(|authored| authored.id.clone()),
        scene: authored.as_ref().map(|authored| authored.scene.clone()),
        name: authored.map(|authored| authored.name),
        in_active_scene,
        alive: world.get_entity(entity).is_ok(),
    }
}

fn page(limit: usize) -> Result<(), Error> {
    bounded(limit, bounds::PAGE, ErrorCode::InvalidPage, "$.limit").map(|_| ())
}

fn shape(source: &CollisionShapeRequest) -> Result<ColliderShape, Error> {
    let dimension = |value: f64| -> Result<f32, Error> {
        let extents = range(MIN_COLLIDER_EXTENT, MAX_COLLIDER_EXTENT);
        bounded(value, extents, ErrorCode::InvalidShape, "$.shape").map(|value| value as f32)
    };
    let shape = match *source {
        CollisionShapeRequest::Cuboid {
            half_extents: [x, y, z],
        } => ColliderShape::Cuboid {
            half_extents: Vec3::new(dimension(x)?, dimension(y)?, dimension(z)?),
        },
        CollisionShapeRequest::Sphere { radius } => ColliderShape::Sphere {
            radius: dimension(radius)?,
        },
    };
    shape.validate().map_err(CollisionError::query_error)?;
    Ok(shape)
}

fn position(values: [f64; 3], path: &str) -> Result<Vec3, Error> {
    if values
        .iter()
        .any(|value| value.check("value", bounds::COORDINATE).is_err())
    {
        return Err(Error::new(
            ErrorCode::InvalidPosition,
            path,
            bounds::COORDINATE.describe(),
        ));
    }
    Ok(Vec3::from_array(values.map(|value| value as f32)))
}

fn pose(translation: [f64; 3], rotation: [f64; 4]) -> Result<Mat4, Error> {
    let translation = position(translation, "$.position")?;
    let length_squared: f64 = rotation.iter().map(|value| value * value).sum();
    if rotation.iter().any(|value| !value.is_finite())
        || !length_squared.is_finite()
        || (length_squared - 1.0).abs() > 0.001
    {
        return Err(Error::new(
            ErrorCode::InvalidRotation,
            "$.rotation",
            "rotation must be a finite normalized XYZW quaternion (squared length within 0.001 of one)",
        ));
    }
    let rotation = Quat::from_array(rotation.map(|value| value as f32)).normalize();
    Ok(Mat4::from_rotation_translation(rotation, translation))
}

fn ray(origin: [f64; 3], direction: [f64; 3], max_distance: f64) -> Result<CollisionRay, Error> {
    let position = position(origin, "$.origin")?;
    bounded(
        max_distance,
        bounds::DISTANCE,
        ErrorCode::InvalidRay,
        "$.max_distance",
    )?;
    let scale = direction
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if direction.iter().any(|value| !value.is_finite()) || scale == 0.0 {
        return Err(Error::new(
            ErrorCode::InvalidRay,
            "$.direction",
            "direction must be a finite nonzero vector",
        ));
    }
    let scaled = direction.map(|value| value / scale);
    let length = scaled.iter().map(|value| value * value).sum::<f64>().sqrt();
    let normalized = scaled.map(|value| value / length);
    if origin
        .into_iter()
        .zip(normalized)
        .any(|(origin, direction)| (origin + direction * max_distance).abs() > 1.0e6)
    {
        return Err(Error::new(
            ErrorCode::InvalidRay,
            "$.max_distance",
            "ray endpoint must remain within +/-1000000",
        ));
    }
    let ray = CollisionRay {
        origin: position,
        direction: Vec3::from_array(normalized.map(|value| value as f32)),
        max_distance: max_distance as f32,
    };
    ray.validate().map_err(CollisionError::query_error)?;
    Ok(ray)
}

fn placed(part: PlacedPart) -> PlacedHit {
    PlacedHit {
        layer: part.layer,
        asset: part.asset,
    }
}

fn query_filter(world: &World, filter: &CollisionFilterRequest) -> Result<QueryFilter, Error> {
    let exclude = filter
        .exclude
        .as_deref()
        .map(|id| resolve_entity(world, id, true, "$.filter.exclude"))
        .transpose()?;
    Ok(QueryFilter {
        layers: CollisionLayers {
            memberships: filter.memberships,
            filter: filter.filter,
        },
        sensors: match filter.sensors {
            CollisionSensorFilter::Include => SensorFilter::Include,
            CollisionSensorFilter::Exclude => SensorFilter::Exclude,
            CollisionSensorFilter::Only => SensorFilter::Only,
        },
        exclude,
    })
}

fn resolve_entity(
    world: &World,
    id: &str,
    require_alive: bool,
    path: &str,
) -> Result<Entity, Error> {
    if id.is_empty()
        || id.len() > 192
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err(Error::new(
            ErrorCode::InvalidSelector,
            path,
            "expected a bounded authored entity ID",
        ));
    }
    let scene = world.get_resource::<OperationScene>().ok_or_else(|| {
        Error::new(
            ErrorCode::NoScene,
            path,
            "entity selectors require an active scene",
        )
    })?;
    let entity = scene.entity(world, id).ok_or_else(|| {
        Error::new(
            ErrorCode::UnknownEntity,
            path,
            "entity is not in the active scene",
        )
    })?;
    if require_alive && world.get_entity(entity).is_err() {
        return Err(Error::new(
            ErrorCode::EntityRemoved,
            path,
            "the authored entity is no longer alive",
        ));
    }
    Ok(entity)
}
