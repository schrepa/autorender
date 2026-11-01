use bevy_ecs::prelude::*;
use engine_core::bounds::{self, MAX_SCAN_ENTITIES, default_limit};
use engine_core::rule::range;
use engine_core::{Error, ErrorCode, Expected, Operation, Rule, SceneIds, Setting, stated};
use engine_places::physics::{
    Collide, ColliderShape, Collision, CollisionLayers, CollisionRay, MAX_COLLIDER_EXTENT,
    MIN_COLLIDER_EXTENT, PlacedPart, QueryFilter, SensorEvent, SensorEventKind, SensorFilter,
    ShapeCastOptions,
};
use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::results::{
    CastHit, CollisionEventsResult, HitEntity, OverlapResult, PlacedHit, RayHit, RaycastResult,
    SensorEventKindView, SensorEventRow, ShapeCastResult,
};
use crate::types::{validate_collider_transform, validate_ray, validate_shape, validate_shape_cast};
use crate::world::{CollisionWorld, SensorEvents, world_of};

type Seen = (
    Res<'static, Collision>,
    SceneIds<'static, 'static>,
    Query<'static, 'static, Entity>,
);

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct RaycastQuery {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
    #[schemars(transform = stated(bounds::DISTANCE))]
    pub max_distance: f64,
    #[serde(default)]
    pub filter: CollisionFilterRequest,
}

impl Operation for RaycastQuery {
    const NAME: &'static str = "raycast";
    type Params = Seen;
    type Request = Self;

    fn run(
        (collision, ids, entities): SystemParamItem<'_, '_, Seen>,
        request: Self,
    ) -> Result<Value, Error> {
        scan(&ids)?;
        let Self {
            origin,
            direction,
            max_distance,
            filter,
        } = request;
        let ray = ray(origin, direction, max_distance)?;
        let filter = query_filter(&ids, &entities, &filter)?;
        let world = filled(&collision)?;
        let hit = world.raycast(ray, filter)?;
        Ok(reply(RaycastResult {
            snapshot: world.stats(),
            complete: true,
            hit: hit.map(|hit| RayHit {
                entity: hit_entity(&ids, &entities, hit.entity),
                distance: hit.distance,
                point: hit.point.to_array(),
                normal: hit.normal.to_array(),
                started_inside: hit.started_inside,
                placed: hit.placed.map(placed),
            }),
        }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct OverlapQuery {
    pub shape: CollisionShapeRequest,
    pub position: [f64; 3],
    #[serde(default = "identity_rotation")]
    pub rotation: [f64; 4],
    #[serde(default)]
    pub filter: CollisionFilterRequest,
    #[serde(default = "default_limit")]
    #[schemars(transform = stated(bounds::PAGE))]
    pub limit: usize,
}

impl Operation for OverlapQuery {
    const NAME: &'static str = "overlap";
    type Params = Seen;
    type Request = Self;

    fn run(
        (collision, ids, entities): SystemParamItem<'_, '_, Seen>,
        request: Self,
    ) -> Result<Value, Error> {
        scan(&ids)?;
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
        validate_collider_transform(shape, matrix)?;
        let filter = query_filter(&ids, &entities, &filter)?;
        let world = filled(&collision)?;
        let mut found = Vec::new();
        world.overlap(shape, matrix, filter, &mut found, limit)?;
        let mut hits: Vec<_> = found
            .into_iter()
            .map(|entity| hit_entity(&ids, &entities, entity))
            .collect();
        hits.sort_by(|a, b| {
            (a.scene.as_deref(), a.id.as_deref()).cmp(&(b.scene.as_deref(), b.id.as_deref()))
        });
        Ok(reply(OverlapResult {
            snapshot: world.stats(),
            complete: true,
            total: hits.len(),
            hits,
        }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ShapeCastQuery {
    pub shape: CollisionShapeRequest,
    pub position: [f64; 3],
    #[serde(default = "identity_rotation")]
    pub rotation: [f64; 4],
    pub direction: [f64; 3],
    #[schemars(transform = stated(bounds::DISTANCE))]
    pub max_distance: f64,
    #[serde(default)]
    pub filter: CollisionFilterRequest,
}

impl Operation for ShapeCastQuery {
    const NAME: &'static str = "shape_cast";
    type Params = Seen;
    type Request = Self;

    fn run(
        (collision, ids, entities): SystemParamItem<'_, '_, Seen>,
        request: Self,
    ) -> Result<Value, Error> {
        scan(&ids)?;
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
        validate_shape_cast(shape, matrix, displacement, options)?;
        let filter = query_filter(&ids, &entities, &filter)?;
        let world = filled(&collision)?;
        let hit = world.cast_shape(shape, matrix, displacement, filter, options)?;
        Ok(reply(ShapeCastResult {
            snapshot: world.stats(),
            complete: true,
            hit: hit.map(|hit| CastHit {
                entity: hit_entity(&ids, &entities, hit.entity),
                fraction: hit.fraction,
                distance: hit.fraction * ray.max_distance,
                point: hit.point.to_array(),
                normal: hit.normal.to_array(),
                started_overlapping: hit.started_overlapping,
                placed: hit.placed.map(placed),
            }),
        }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CollisionEventsQuery {
    #[serde(default)]
    pub entity: Option<String>,
    #[serde(default)]
    #[schemars(transform = stated(bounds::EVENT_OFFSET))]
    pub offset: usize,
    #[serde(default = "default_limit")]
    #[schemars(transform = stated(bounds::PAGE))]
    pub limit: usize,
}

impl Operation for CollisionEventsQuery {
    const NAME: &'static str = "collision_events";
    type Params = Seen;
    type Request = Self;

    fn run(
        (collision, ids, entities): SystemParamItem<'_, '_, Seen>,
        request: Self,
    ) -> Result<Value, Error> {
        scan(&ids)?;
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
        let entity = entity
            .as_deref()
            .map(|id| resolve_entity(&ids, &entities, id, false, "$.entity"))
            .transpose()?;
        let unsynced = SensorEvents::default();
        let events = world_of(&collision).map_or(&unsynced, |world| &world.sensors);
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
                first: hit_entity(&ids, &entities, event.pair.first),
                second: hit_entity(&ids, &entities, event.pair.second),
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
            error: events.error(),
            total,
            events: rows,
            next_offset: (next < total).then_some(next),
        }))
    }

    fn schema() -> Value {
        schema_of::<Self>()
    }
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum CollisionShapeRequest {
    Cuboid {
        #[schemars(transform = stated(range(MIN_COLLIDER_EXTENT, MAX_COLLIDER_EXTENT)))]
        half_extents: [f64; 3],
    },
    Sphere {
        #[schemars(transform = stated(range(MIN_COLLIDER_EXTENT, MAX_COLLIDER_EXTENT)))]
        radius: f64,
    },
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct CollisionFilterRequest {
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

#[derive(Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CollisionSensorFilter {
    #[default]
    Include,
    Exclude,
    Only,
}

fn identity_rotation() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

pub(crate) fn schema_of<T: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null)
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
        Err(invalid) => Err(Error {
            code,
            path: path.to_owned(),
            ..Error::from(invalid)
        }),
    }
}

pub(crate) fn filled(collision: &Collision) -> Result<&CollisionWorld, Error> {
    world_of(collision).ok_or_else(|| Error::new(ErrorCode::MissingResource, "$.collision"))
}

fn scan(ids: &SceneIds) -> Result<(), Error> {
    let count = ids.len();
    if count > MAX_SCAN_ENTITIES {
        return Err(Error::new(ErrorCode::ScanLimit, "$.scene")
            .at_most(MAX_SCAN_ENTITIES as u64)
            .got(count));
    }
    Ok(())
}

fn hit_entity(ids: &SceneIds, entities: &Query<Entity>, entity: Entity) -> HitEntity {
    let authored = ids.authored(entity);
    let in_active_scene = authored
        .as_ref()
        .is_some_and(|authored| ids.holds(authored, entity));
    HitEntity {
        id: authored.as_ref().map(|authored| authored.id.clone()),
        scene: authored.as_ref().map(|authored| authored.scene.clone()),
        name: authored.map(|authored| authored.name),
        in_active_scene,
        alive: entities.contains(entity),
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
    validate_shape(shape)?;
    Ok(shape)
}

fn position(values: [f64; 3], path: &str) -> Result<Vec3, Error> {
    for value in values {
        bounded(value, bounds::COORDINATE, ErrorCode::InvalidPosition, path)?;
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
        return Err(Error::new(ErrorCode::InvalidRotation, "$.rotation")
            .expected(Expected::Normalized)
            .got(rotation));
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
        return Err(Error::new(ErrorCode::InvalidRay, "$.direction")
            .expected(Expected::Finite)
            .got(direction));
    }
    let scaled = direction.map(|value| value / scale);
    let length = scaled.iter().map(|value| value * value).sum::<f64>().sqrt();
    let normalized = scaled.map(|value| value / length);
    let end: [f64; 3] = std::array::from_fn(|axis| origin[axis] + normalized[axis] * max_distance);
    if end.iter().any(|value| value.abs() > 1.0e6) {
        return Err(Error::new(ErrorCode::InvalidRay, "$.max_distance")
            .range(-1.0e6, 1.0e6)
            .got(end));
    }
    let ray = CollisionRay {
        origin: position,
        direction: Vec3::from_array(normalized.map(|value| value as f32)),
        max_distance: max_distance as f32,
    };
    validate_ray(ray)?;
    Ok(ray)
}

fn placed(part: PlacedPart) -> PlacedHit {
    PlacedHit {
        layer: part.layer,
        asset: part.asset,
    }
}

fn query_filter(
    ids: &SceneIds,
    entities: &Query<Entity>,
    filter: &CollisionFilterRequest,
) -> Result<QueryFilter, Error> {
    let exclude = filter
        .exclude
        .as_deref()
        .map(|id| resolve_entity(ids, entities, id, true, "$.filter.exclude"))
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

pub(crate) fn resolve_entity(
    ids: &SceneIds,
    entities: &Query<Entity>,
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
        return Err(Error::new(ErrorCode::InvalidSelector, path)
            .expected(Expected::Id)
            .got(id));
    }
    if !ids.loaded() {
        return Err(Error::new(ErrorCode::NoScene, path));
    }
    let entity = ids
        .entity(id)
        .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, path).got(id))?;
    if require_alive && !entities.contains(entity) {
        return Err(Error::new(ErrorCode::EntityRemoved, path).got(id));
    }
    Ok(entity)
}
