//! Read-only spatial observations. Cache synchronization never commits sensor events.
use std::collections::HashMap;

use engine_collision::{
    Collider, ColliderShape, CollisionError, CollisionLayers, CollisionRay, CollisionWorld,
    QueryFilter, SensorEventKind, SensorEvents, SensorFilter, ShapeCastOptions,
    refresh_collision_world, validate_collider_transform, validate_shape_cast,
};
use engine_core::{
    Engine, GlobalTransforms,
    bevy_ecs::prelude::{Entity, World},
    glam::{Mat4, Quat, Vec3},
};
use engine_scene::{SceneEntity, SceneInstance};
use serde_json::{Value, json};

use crate::{
    DevError, DevRequest, MAX_PAGE_SIZE, MAX_SCAN_ENTITIES,
    protocol::{CollisionFilterRequest, CollisionSensorFilter, CollisionShapeRequest},
    views::{ColliderShapeView, ColliderView, CollisionLimitsView, CollisionWorldView},
};

pub(crate) fn spatial_request(request: &DevRequest) -> bool {
    matches!(
        request,
        DevRequest::Raycast { .. }
            | DevRequest::Overlap { .. }
            | DevRequest::ShapeCast { .. }
            | DevRequest::CollisionEvents { .. }
    )
}

pub(crate) fn execute(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    request: DevRequest,
) -> Result<Value, DevError> {
    let identities = Identities::new(scene)?;
    match request {
        DevRequest::Raycast {
            origin,
            direction,
            max_distance,
            filter,
            ..
        } => {
            let ray = ray(origin, direction, max_distance)?;
            let filter = query_filter(engine.app.world(), scene, filter)?;
            refresh_collision_world(engine.app.world_mut()).map_err(native_error)?;
            let hit = engine
                .app
                .world_mut()
                .resource_mut::<CollisionWorld>()
                .raycast(ray, filter)
                .map_err(native_error)?;
            let world = engine.app.world();
            Ok(json!({
                "snapshot": world_value(world.resource::<CollisionWorld>()), "complete": true,
                "hit": hit.map(|hit| json!({"entity":identities.entity(world, hit.entity),
                    "distance":hit.distance, "point":hit.point.to_array(),
                    "normal":hit.normal.to_array(), "started_inside":hit.started_inside,
                    "placed":hit.placed.map(placed_json)})),
            }))
        }
        DevRequest::Overlap {
            shape: source,
            position,
            rotation,
            filter,
            limit,
            ..
        } => {
            page(limit)?;
            let shape = shape(source)?;
            let matrix = pose(position, rotation)?;
            validate_collider_transform(shape, matrix).map_err(native_error)?;
            let filter = query_filter(engine.app.world(), scene, filter)?;
            refresh_collision_world(engine.app.world_mut()).map_err(native_error)?;
            let mut entities = Vec::new();
            engine
                .app
                .world_mut()
                .resource_mut::<CollisionWorld>()
                .overlap(shape, matrix, filter, &mut entities, limit)
                .map_err(native_error)?;
            let world = engine.app.world();
            let mut hits: Vec<_> = entities
                .into_iter()
                .map(|entity| identities.entity(world, entity))
                .collect();
            hits.sort_by(|a, b| {
                (a["scene"].as_str(), a["id"].as_str())
                    .cmp(&(b["scene"].as_str(), b["id"].as_str()))
            });
            Ok(
                json!({"snapshot":world_value(world.resource::<CollisionWorld>()),
                "complete":true,"total":hits.len(),"hits":hits}),
            )
        }
        DevRequest::ShapeCast {
            shape: source,
            position,
            rotation,
            direction,
            max_distance,
            filter,
            ..
        } => {
            let shape = shape(source)?;
            let matrix = pose(position, rotation)?;
            let ray = ray(position, direction, max_distance)?;
            let displacement = ray.direction * ray.max_distance;
            let options = ShapeCastOptions::default();
            validate_shape_cast(shape, matrix, displacement, options).map_err(native_error)?;
            let filter = query_filter(engine.app.world(), scene, filter)?;
            refresh_collision_world(engine.app.world_mut()).map_err(native_error)?;
            let hit = engine
                .app
                .world_mut()
                .resource_mut::<CollisionWorld>()
                .cast_shape(shape, matrix, displacement, filter, options)
                .map_err(native_error)?;
            let world = engine.app.world();
            Ok(json!({
                "snapshot":world_value(world.resource::<CollisionWorld>()),"complete":true,
                "hit":hit.map(|hit| json!({"entity":identities.entity(world, hit.entity),
                    "fraction":hit.fraction,"distance":hit.fraction * ray.max_distance,
                    "point":hit.point.to_array(),"normal":hit.normal.to_array(),
                    "started_overlapping":hit.started_overlapping,
                    "placed":hit.placed.map(placed_json)})),
            }))
        }
        DevRequest::CollisionEvents {
            entity,
            offset,
            limit,
            ..
        } => {
            page(limit)?;
            if offset > 131_072 {
                return Err(DevError::new(
                    "invalid_page",
                    "$.offset",
                    "sensor event offset must be in 0..=131072",
                ));
            }
            let entity = entity
                .as_deref()
                .map(|id| resolve_entity(engine.app.world(), scene, id, false, "$.entity"))
                .transpose()?;
            let world = engine.app.world();
            let events = world
                .get_resource::<SensorEvents>()
                .ok_or_else(|| native_error(CollisionError::MissingPlugin))?;
            let matches = |event: &&engine_collision::SensorEvent| {
                entity
                    .is_none_or(|entity| event.pair.first == entity || event.pair.second == entity)
            };
            let total = events.events().iter().filter(matches).count();
            let rows: Vec<_> = events.events().iter().filter(matches).skip(offset).take(limit)
                .map(|event| json!({"kind":match event.kind { SensorEventKind::Enter => "enter", SensorEventKind::Exit => "exit" },
                    "first":identities.entity(world, event.pair.first),"second":identities.entity(world, event.pair.second)})).collect();
            let next = offset.saturating_add(rows.len());
            Ok(
                json!({"sensor_only":true,"sequence":events.sequence().to_string(),
                "tick":events.tick().map(|tick| tick.to_string()),
                "committed_tick":events.committed_tick().map(|tick| tick.to_string()),
                "complete":events.complete(),"candidate_pairs":events.candidate_pairs(),
                "active_pairs":events.pairs().len(),"error":events.error().map(|error| json!(native_error(error))),
                "total":total,"events":rows,"next_offset":(next < total).then_some(next)}),
            )
        }
        _ => Err(DevError::new(
            "invalid_request",
            "$.op",
            "expected a collision query",
        )),
    }
}

pub(crate) fn world_value(value: &CollisionWorld) -> Value {
    let report = value.report();
    let config = value.config();
    json!(CollisionWorldView {
        revision: report.revision.to_string(),
        frame: report.frame.to_string(),
        tick: report.tick.to_string(),
        colliders: report.colliders,
        active_colliders: report.active_colliders,
        invalid_colliders: report.invalid_colliders,
        overflow_colliders: report.overflow_colliders,
        bvh_nodes: report.bvh_nodes,
        cache_bytes: report.cache_bytes,
        complete: report.complete,
        placed_colliders: report.placed_colliders,
        limits: CollisionLimitsView {
            max_colliders: config.max_colliders,
            max_candidate_pairs: config.max_candidate_pairs,
            max_sensor_pairs: config.max_sensor_pairs,
            max_query_candidates: config.max_query_candidates,
            max_query_hits: config.max_query_hits
        },
    })
}

pub(crate) fn collider_value(world: &World, entity: Entity) -> Option<Value> {
    let collider = *world.get::<Collider>(entity)?;
    let valid = world
        .get_resource::<GlobalTransforms>()
        .and_then(|globals| globals.get(entity))
        .and_then(|global| global.matrix())
        .ok_or(CollisionError::InvalidTransform)
        .and_then(|matrix| validate_collider_transform(collider.shape(), matrix));
    let cache_status = world
        .get_resource::<CollisionWorld>()
        .map_or("unavailable", |cache| match cache.collider_status(entity) {
            Some(engine_collision::ColliderStatus::Active) => "active",
            Some(engine_collision::ColliderStatus::Invalid) => "invalid",
            Some(engine_collision::ColliderStatus::Overflow) => "overflow",
            None => "not_cached",
        });
    Some(json!(ColliderView {
        shape: match collider.shape() {
            ColliderShape::Cuboid { half_extents } => ColliderShapeView::Cuboid {
                half_extents: half_extents.to_array()
            },
            ColliderShape::Sphere { radius } => ColliderShapeView::Sphere { radius },
        },
        memberships: collider.layers().memberships,
        filter: collider.layers().filter,
        sensor: collider.is_sensor(),
        world_valid: valid.is_ok(),
        world_error: valid.err().map(|error| error.to_string()),
        cache_status: cache_status.to_owned(),
    }))
}

pub(crate) fn native_error(error: CollisionError) -> DevError {
    DevError::new(error.code(), "$.collision", error.to_string())
}

fn page(limit: usize) -> Result<(), DevError> {
    if !(1..=MAX_PAGE_SIZE).contains(&limit) {
        return Err(DevError::new(
            "invalid_page",
            "$.limit",
            "limit must be in 1..=100",
        ));
    }
    Ok(())
}

fn shape(source: CollisionShapeRequest) -> Result<ColliderShape, DevError> {
    let dimension = |value: f64| -> Result<f32, DevError> {
        if !value.is_finite() || !(1.0e-4..=1.0e4).contains(&value) {
            return Err(DevError::new(
                "invalid_shape",
                "$.shape",
                "shape dimensions must be finite and in [0.0001, 10000]",
            ));
        }
        Ok(value as f32)
    };
    let shape = match source {
        CollisionShapeRequest::Cuboid {
            half_extents: [x, y, z],
        } => ColliderShape::Cuboid {
            half_extents: Vec3::new(dimension(x)?, dimension(y)?, dimension(z)?),
        },
        CollisionShapeRequest::Sphere { radius } => ColliderShape::Sphere {
            radius: dimension(radius)?,
        },
    };
    shape.validate().map_err(native_error)?;
    Ok(shape)
}

fn position(values: [f64; 3], path: &str) -> Result<Vec3, DevError> {
    if values
        .iter()
        .any(|value| !value.is_finite() || value.abs() > 1.0e6)
    {
        return Err(DevError::new(
            "invalid_position",
            path,
            "coordinates must be finite and within +/-1000000",
        ));
    }
    Ok(Vec3::from_array(values.map(|value| value as f32)))
}

fn pose(translation: [f64; 3], rotation: [f64; 4]) -> Result<Mat4, DevError> {
    let translation = position(translation, "$.position")?;
    let length_squared: f64 = rotation.iter().map(|value| value * value).sum();
    if rotation.iter().any(|value| !value.is_finite())
        || !length_squared.is_finite()
        || (length_squared - 1.0).abs() > 0.001
    {
        return Err(DevError::new(
            "invalid_rotation",
            "$.rotation",
            "rotation must be a finite normalized XYZW quaternion (squared length within 0.001 of one)",
        ));
    }
    let rotation = Quat::from_array(rotation.map(|value| value as f32)).normalize();
    Ok(Mat4::from_rotation_translation(rotation, translation))
}

fn ray(origin: [f64; 3], direction: [f64; 3], max_distance: f64) -> Result<CollisionRay, DevError> {
    let position = position(origin, "$.origin")?;
    if !max_distance.is_finite() || !(0.0..=2.0e6).contains(&max_distance) {
        return Err(DevError::new(
            "invalid_ray",
            "$.max_distance",
            "distance must be finite and in [0, 2000000]",
        ));
    }
    let scale = direction
        .iter()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    if direction.iter().any(|value| !value.is_finite()) || scale == 0.0 {
        return Err(DevError::new(
            "invalid_ray",
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
        return Err(DevError::new(
            "invalid_ray",
            "$.max_distance",
            "ray endpoint must remain within +/-1000000",
        ));
    }
    let ray = CollisionRay {
        origin: position,
        direction: Vec3::from_array(normalized.map(|value| value as f32)),
        max_distance: max_distance as f32,
    };
    ray.validate().map_err(native_error)?;
    Ok(ray)
}

/// A placed object a hit names (decision 064): its terrain's placement
/// layer and asset indices.
fn placed_json(part: engine_collision::PlacedPart) -> serde_json::Value {
    json!({"layer": part.layer, "asset": part.asset})
}

fn query_filter(
    world: &World,
    scene: Option<&SceneInstance>,
    filter: CollisionFilterRequest,
) -> Result<QueryFilter, DevError> {
    let exclude = filter
        .exclude
        .as_deref()
        .map(|id| resolve_entity(world, scene, id, true, "$.filter.exclude"))
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
    scene: Option<&SceneInstance>,
    id: &str,
    require_alive: bool,
    path: &str,
) -> Result<Entity, DevError> {
    if id.is_empty()
        || id.len() > 192
        || !id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err(DevError::new(
            "invalid_selector",
            path,
            "expected a bounded authored entity ID",
        ));
    }
    let scene = scene.ok_or_else(|| {
        DevError::new(
            "no_active_scene",
            path,
            "entity selectors require an active scene",
        )
    })?;
    let entity = crate::inspect::authored_entity(world, scene, id).ok_or_else(|| {
        DevError::new("unknown_entity", path, "entity is not in the active scene")
    })?;
    if require_alive && world.get_entity(entity).is_err() {
        return Err(DevError::new(
            "entity_removed",
            path,
            "the authored entity is no longer alive",
        ));
    }
    Ok(entity)
}

struct Identities<'a> {
    scene: Option<&'a SceneInstance>,
    ids: HashMap<Entity, &'a str>,
}

impl<'a> Identities<'a> {
    fn new(scene: Option<&'a SceneInstance>) -> Result<Self, DevError> {
        if scene.is_some_and(|scene| scene.entity_count() > MAX_SCAN_ENTITIES) {
            return Err(DevError::new(
                "scan_limit",
                "$.scene",
                "collision identity lookup admits at most 10000 authored entities",
            ));
        }
        Ok(Self {
            scene,
            ids: scene
                .into_iter()
                .flat_map(SceneInstance::entities)
                .map(|(id, entity)| (entity, id))
                .collect(),
        })
    }

    fn entity(&self, world: &World, entity: Entity) -> Value {
        let id = self.ids.get(&entity).copied();
        let metadata = world.get::<SceneEntity>(entity);
        // A loaded streamed entity (decision 086) of the active scene.
        let streamed = id.is_none()
            && metadata.zip(self.scene).is_some_and(|(metadata, scene)| {
                scene.streamed().is_some() && metadata.scene_id == scene.id()
            });
        json!({"id":id.or_else(|| metadata.map(|metadata| metadata.authored_id.as_str())),
            "scene":if id.is_some() { self.scene.map(SceneInstance::id) } else { metadata.map(|metadata| metadata.scene_id.as_str()) },
            "name":metadata.map(|metadata| metadata.name.as_str()),
            "in_active_scene":id.is_some() || streamed,"alive":world.get_entity(entity).is_ok()})
    }
}
