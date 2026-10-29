use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::{Entity, World};
use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{
    AuthoredId, OperationScene, Transform, edit_transform, set_parent, smoke_check, smoke_eq,
};
use glam::{EulerRot, Quat, Vec3};
use serde_json::{Value, json};

use crate::{Collider, ColliderShape, CollisionLayers};
use engine_core::{Error, ErrorCode};

const SCENE: &str = "collision-fixture";
const TICK_DISTANCE: f32 = 6.5 / 60.0;

pub(crate) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let built = build(session.app().world_mut());
    smoke_check!(
        built.is_ok(),
        "fixture_builds",
        json!(built.as_ref().err().map(|error| format!("{error:?}"))),
        json!(null)
    );
    ask(session, "control", json!({"name": "pause"}))?;
    session.step(1);
    let ray = queries(session)?;
    sensor(session, &ray)?;
    sweep(session)?;
    changes(session, &ray)?;
    ask(session, "control", json!({"name": "resume"}))?;
    query(session, "raycast", ray.clone())?;
    ask(session, "control", json!({"name": "pause"}))?;
    session.restart(&["collision", "physics"]);
    let missing = refuse(session, "query", json!({"name": "raycast", "args": ray}))?;
    smoke_eq!(
        json!(missing["code"]),
        json!("missing_plugin"),
        "plugin_is_opt_in"
    );
    Ok(())
}

fn queries(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    for name in ["Collider", "CollisionWorld"] {
        ask(session, "describe", json!({"name": name}))?;
    }
    let world = ask(session, "inspect", json!({"component": "CollisionWorld"}))?["value"].clone();
    smoke_check!(
        world["complete"] == true
            && world["colliders"].as_u64().unwrap_or(0) > 0
            && world["invalid_colliders"] == 0
            && world["overflow_colliders"] == 0
            && world["bvh_nodes"].as_u64().unwrap_or(0) > 0,
        "collision_world_inspects",
        json!(world),
        json!({"complete": true, "invalid_colliders": 0, "overflow_colliders": 0})
    );
    let volume = component(session, "player", "Collider")?;
    smoke_check!(
        volume.is_object(),
        "collider_inspects",
        json!(volume),
        json!("object")
    );
    let before = state(session)?;

    let filter = json!({"memberships": 2, "filter": 1, "sensors": "exclude"});
    let ray =
        json!({"origin": [4, 4, 0], "direction": [0, -7, 0], "max_distance": 5, "filter": filter});
    let hit = query(session, "raycast", ray.clone())?["hit"].clone();
    smoke_eq!(
        json!(hit["entity"]["id"]),
        json!("target"),
        "closest_raycast"
    );
    smoke_check!(
        near(&hit["distance"], 3.0),
        "closest_raycast_distance",
        json!(hit["distance"]),
        json!(3.0)
    );
    smoke_check!(
        near(&hit["normal"][1], 1.0),
        "closest_raycast_normal",
        json!(hit["normal"]),
        json!([0, 1, 0])
    );
    let layered = query(
        session,
        "raycast",
        with(&ray, "filter", json!({"memberships": 8, "filter": 1})),
    )?;
    smoke_check!(
        layered["hit"].is_null(),
        "symmetric_layers",
        json!(layered["hit"]),
        json!(null)
    );

    let sensor_filter = json!({"memberships": 2, "filter": 4, "sensors": "only"});
    let sensor_ray = json!({"origin": [0, 0.6, 0], "direction": [0, 0, -1], "max_distance": 3, "filter": sensor_filter});
    let sensed = query(session, "raycast", sensor_ray.clone())?;
    smoke_eq!(
        json!(sensed["hit"]["entity"]["id"]),
        json!("sensor"),
        "sensor_filter_only"
    );
    let excluded = query(
        session,
        "raycast",
        with(
            &sensor_ray,
            "filter",
            with(&sensor_filter, "sensors", json!("exclude")),
        ),
    )?;
    smoke_check!(
        excluded["hit"].is_null(),
        "sensor_filter_exclude",
        json!(excluded["hit"]),
        json!(null)
    );
    let skipped = query(
        session,
        "raycast",
        with(
            &sensor_ray,
            "filter",
            with(&sensor_filter, "exclude", json!("sensor")),
        ),
    )?;
    smoke_check!(
        skipped["hit"].is_null(),
        "sensor_exclusion",
        json!(skipped["hit"]),
        json!(null)
    );

    shapes(session, &ray, &filter)?;
    refuse(
        session,
        "query",
        json!({"name": "raycast", "args": with(&ray, "direction", json!([0, 0, 0]))}),
    )?;
    refuse(
        session,
        "query",
        json!({"name": "overlap", "args": {"position": [0, 0, 0], "shape": {"kind": "sphere", "radius": 20}, "filter": filter, "limit": 1}}),
    )?;
    smoke_eq!(state(session)?, before, "invalid_and_overflow_nonmutating");
    Ok(ray)
}

fn shapes(session: &mut dyn SmokeSession, ray: &Value, filter: &Value) -> SmokeResult {
    let inside = query(session, "raycast", with(ray, "origin", json!([4, 0.5, 0])))?["hit"].clone();
    smoke_eq!(json!(inside["started_inside"]), json!(true), "inside_hit");
    smoke_check!(
        near(&inside["distance"], 0.0),
        "inside_hit_distance",
        json!(inside["distance"]),
        json!(0.0)
    );

    let overlap = query(
        session,
        "overlap",
        json!({"position": [4, 0.5, 0], "shape": {"kind": "sphere", "radius": 0.2}, "filter": filter}),
    )?;
    let overlapped: Vec<Value> = overlap["hits"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|hit| hit["id"].clone())
        .collect();
    smoke_eq!(json!(overlapped), json!(["target"]), "overlap");

    let cast = query(
        session,
        "shape_cast",
        json!({"position": [4, 4, 0], "shape": cuboid(0.25), "direction": [0, -1, 0], "max_distance": 5, "filter": filter}),
    )?["hit"]
        .clone();
    smoke_eq!(json!(cast["entity"]["id"]), json!("target"), "shape_cast");
    smoke_check!(
        near(&cast["distance"], 2.75),
        "shape_cast_distance",
        json!(cast["distance"]),
        json!(2.75)
    );

    for (id, x) in [("sheared", 8), ("ellipsoid", -4)] {
        let hit = query(
            session,
            "raycast",
            with(
                &with(ray, "origin", json!([x, 5, 0])),
                "max_distance",
                json!(6),
            ),
        )?["hit"]
            .clone();
        smoke_eq!(json!(hit["entity"]["id"]), json!(id), "shear_and_ellipsoid");
        smoke_check!(
            near(&hit["distance"], 3.5),
            "shear_and_ellipsoid_distance",
            json!(hit["distance"]),
            json!(3.5)
        );
    }
    let sheared_cast = query(
        session,
        "shape_cast",
        json!({"position": [8, 5, 0], "shape": cuboid(0.25), "direction": [0, -1, 0], "max_distance": 5, "filter": filter}),
    )?["hit"]
        .clone();
    smoke_eq!(
        json!(sheared_cast["entity"]["id"]),
        json!("sheared"),
        "sheared_cast"
    );
    smoke_check!(
        near(&sheared_cast["distance"], 3.25),
        "sheared_cast_distance",
        json!(sheared_cast["distance"]),
        json!(3.25)
    );
    Ok(())
}

fn sensor(session: &mut dyn SmokeSession, ray: &Value) -> SmokeResult {
    let mut transitions = Vec::new();
    for tick in 0..36 {
        let player = entity(session.app().world(), "player");
        let moved = player.is_some_and(|player| {
            edit_transform(session.app().world_mut(), player, |transform| {
                transform.translation.z -= TICK_DISTANCE;
            })
            .is_ok()
        });
        smoke_check!(moved, "player_moves", json!(tick), json!("moved"));
        session.step(1);
        let events = sensor_events(session)?;
        let rows = events["events"].as_array().cloned().unwrap_or_default();
        for event in &rows {
            transitions.push(json!({"tick": tick, "kind": event["kind"]}));
        }
        if rows.is_empty() {
            continue;
        }
        let exact = state(session)?;
        query(session, "raycast", ray.clone())?;
        smoke_eq!(sensor_events(session)?, events, "events_nonconsuming");
        smoke_eq!(state(session)?, exact, "queries_preserve_clock");
        ask(
            session,
            "control",
            json!({"name": "present", "args": {"alpha": 0.5}}),
        )?;
        let preview = state(session)?;
        query(session, "raycast", ray.clone())?;
        smoke_eq!(state(session)?, preview, "queries_preserve_preview");
        smoke_eq!(sensor_events(session)?, events, "events_survive_preview");
    }
    let kinds: Vec<Value> = transitions
        .iter()
        .map(|transition| transition["kind"].clone())
        .collect();
    smoke_eq!(json!(kinds), json!(["enter", "exit"]), "sensor_enter_exit");
    Ok(())
}

fn sweep(session: &mut dyn SmokeSession) -> SmokeResult {
    let world = session.app().world_mut();
    let placed = entity(world, "player").is_some_and(|player| {
        edit_transform(world, player, |transform| {
            transform.translation = Vec3::new(0.0, 0.6, 0.0);
        })
        .is_ok()
    });
    smoke_check!(placed, "player_returns", json!(placed), json!(true));
    let wall = collider(cuboid_shape([3.0, 2.0, 0.01]), 1, 2, false);
    smoke_check!(
        wall.is_ok(),
        "thin_wall_builds",
        json!(format!("{wall:?}")),
        json!("collider")
    );
    if let Ok(wall) = wall {
        let world = session.app().world_mut();
        let spawned = world
            .spawn((
                placed_at([0.66, 0.6, 0.0], [0.0, 90.0, 0.0], [1.0; 3]),
                wall,
            ))
            .id();
        rename(world, |named| {
            named.insert("thin-wall".to_owned(), spawned);
        });
    }
    session.step(1);
    let swept = query(
        session,
        "shape_cast",
        json!({
            "position": [0, 0.65, 0], "shape": cuboid(0.6), "direction": [1, 0, 0], "max_distance": 2,
            "filter": {"memberships": 2, "filter": 1, "sensors": "exclude", "exclude": "player"},
        }),
    )?["hit"]
        .clone();
    smoke_eq!(
        json!(swept["entity"]["id"]),
        json!("thin-wall"),
        "swept_thin_wall"
    );
    smoke_check!(
        near(&swept["distance"], 0.05),
        "swept_thin_wall_distance",
        json!(swept["distance"]),
        json!(0.05)
    );
    Ok(())
}

fn changes(session: &mut dyn SmokeSession, ray: &Value) -> SmokeResult {
    let invalid_before = state(session)?;
    let flat = collider(cuboid_shape([0.5; 3]), 1, 2, false);
    smoke_check!(
        flat.is_ok(),
        "flat_collider_builds",
        json!(format!("{flat:?}")),
        json!("collider")
    );
    let flat = flat.ok().map(|flat| {
        session
            .app()
            .world_mut()
            .spawn((placed_at([12.0, 0.5, 0.0], [0.0; 3], [0.0, 1.0, 1.0]), flat))
            .id()
    });
    session.step(1);
    let counted = ask(session, "inspect", json!({"component": "CollisionWorld"}))?["value"].clone();
    smoke_eq!(
        json!(counted["invalid_colliders"]),
        json!(1),
        "invalid_collider_counted"
    );
    let still = query(session, "raycast", ray.clone())?;
    smoke_eq!(
        json!(still["hit"]["entity"]["id"]),
        json!("target"),
        "invalid_collider_preserves_world"
    );
    smoke_eq!(
        component(session, "player", "Transform")?,
        invalid_before["transform"].clone(),
        "invalid_collider_preserves_player"
    );
    if let Some(flat) = flat {
        session.app().world_mut().despawn(flat);
    }
    let world = session.app().world_mut();
    if let Some(target) = entity(world, "target") {
        world.despawn(target);
    }
    rename(world, |named| {
        named.remove("target");
    });
    session.step(1);
    let retired = query(session, "raycast", with(ray, "max_distance", json!(3.5)))?;
    smoke_check!(
        retired["hit"].is_null(),
        "removed_colliders_retire",
        json!(retired["hit"]),
        json!(null)
    );
    let cleared = ask(session, "inspect", json!({"component": "CollisionWorld"}))?["value"].clone();
    smoke_eq!(
        json!(cleared["invalid_colliders"]),
        json!(0),
        "invalid_collider_retires"
    );
    Ok(())
}

fn build(world: &mut World) -> Result<(), Error> {
    let fixture = [
        (
            "player",
            placed_at([0.0, 0.6, 0.0], [0.0; 3], [1.2; 3]),
            Some(collider(cuboid_shape([0.5; 3]), 2, 5, false)?),
        ),
        (
            "floor",
            placed_at([0.0, -0.5, 0.0], [0.0; 3], [1.0; 3]),
            Some(collider(cuboid_shape([20.0, 0.5, 20.0]), 1, 2, false)?),
        ),
        (
            "target",
            placed_at([4.0, 0.5, 0.0], [0.0; 3], [1.0; 3]),
            Some(collider(cuboid_shape([0.5; 3]), 1, 2, false)?),
        ),
        (
            "sensor",
            placed_at([0.0, 0.6, -2.0], [0.0; 3], [1.0; 3]),
            Some(collider(ColliderShape::Sphere { radius: 0.4 }, 4, 2, true)?),
        ),
        (
            "parent",
            placed_at([8.0, 1.0, 0.0], [0.0, 30.0, 0.0], [2.0, 1.0, 0.5]),
            None,
        ),
        (
            "sheared",
            placed_at([0.0; 3], [0.0, 40.0, 0.0], [1.0; 3]),
            Some(collider(cuboid_shape([0.5; 3]), 1, 2, false)?),
        ),
        (
            "ellipsoid",
            placed_at([-4.0, 1.0, 0.0], [0.0; 3], [2.0, 1.0, 0.5]),
            Some(collider(
                ColliderShape::Sphere { radius: 0.5 },
                1,
                2,
                false,
            )?),
        ),
    ];
    let mut named = BTreeMap::new();
    for (id, transform, collider) in fixture {
        let mut spawned = world.spawn(transform);
        if let Some(collider) = collider {
            spawned.insert(collider);
        }
        named.insert(id.to_owned(), spawned.id());
    }
    if set_parent(world, named["sheared"], Some(named["parent"])).is_err() {
        return Err(Error::new(ErrorCode::CollisionInvalidTransform, "$"));
    }
    world.insert_resource(OperationScene {
        id: SCENE.to_owned(),
        entities: Arc::new(named),
        streamed: false,
        authored,
    });
    Ok(())
}

fn collider(
    shape: ColliderShape,
    memberships: u32,
    filter: u32,
    sensor: bool,
) -> Result<Collider, Error> {
    Ok(Collider::new(shape)?
        .with_layers(CollisionLayers::new(memberships, filter))
        .with_sensor(sensor))
}

fn cuboid_shape(half_extents: [f32; 3]) -> ColliderShape {
    ColliderShape::Cuboid {
        half_extents: Vec3::from_array(half_extents),
    }
}

fn cuboid(half: f64) -> Value {
    json!({"kind": "cuboid", "half_extents": [half, half, half]})
}

fn placed_at(translation: [f32; 3], degrees: [f32; 3], scale: [f32; 3]) -> Transform {
    let [x, y, z] = degrees.map(f32::to_radians);
    Transform {
        translation: Vec3::from_array(translation),
        rotation: Quat::from_euler(EulerRot::XYZ, x, y, z),
        scale: Vec3::from_array(scale),
    }
}

fn authored(world: &World, entity: Entity) -> Option<AuthoredId> {
    let scene = world.get_resource::<OperationScene>()?;
    let (id, _) = scene.entities.iter().find(|&(_, &named)| named == entity)?;
    Some(AuthoredId {
        scene: scene.id.clone(),
        id: id.clone(),
        name: id.clone(),
    })
}

fn entity(world: &World, id: &str) -> Option<Entity> {
    world
        .get_resource::<OperationScene>()
        .and_then(|scene| scene.entities.get(id).copied())
}

fn rename(world: &mut World, change: impl FnOnce(&mut BTreeMap<String, Entity>)) {
    let mut named = world
        .get_resource::<OperationScene>()
        .map(|scene| (*scene.entities).clone())
        .unwrap_or_default();
    change(&mut named);
    world.insert_resource(OperationScene {
        id: SCENE.to_owned(),
        entities: Arc::new(named),
        streamed: false,
        authored,
    });
}

fn near(value: &Value, expected: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|value| (value - expected).abs() < 0.001)
}

fn with(base: &Value, key: &str, value: Value) -> Value {
    let mut changed = base.clone();
    changed[key] = value;
    changed
}

fn state(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let status = ask(session, "query", json!({"name": "status"}))?;
    Ok(json!({
        "frame": status["observation"]["frame"],
        "tick": status["observation"]["tick"],
        "presentation": status["observation"]["presentation"],
        "transform": component(session, "player", "Transform")?,
        "global": component(session, "player", "GlobalTransform")?,
    }))
}

fn sensor_events(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let mut events = query(session, "collision_events", json!({"entity": "sensor"}))?;
    if let Some(fields) = events.as_object_mut() {
        for key in ["observation", "rendering", "paused", "source"] {
            fields.remove(key);
        }
    }
    Ok(events)
}

fn component(
    session: &mut dyn SmokeSession,
    id: &str,
    component: &str,
) -> Result<Value, SmokeFailure> {
    let inspected = ask(
        session,
        "inspect",
        json!({"entity": id, "component": component}),
    )?;
    Ok(inspected["entity"]["components"][component].clone())
}

fn query(session: &mut dyn SmokeSession, name: &str, args: Value) -> Result<Value, SmokeFailure> {
    ask(session, "query", json!({"name": name, "args": args}))
}

fn ask(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Result<Value, SmokeFailure> {
    let sent = json!({"verb": verb, "body": body});
    let reply = session.request(verb, body);
    smoke_check!(
        reply.is_ok(),
        "request_accepted",
        json!(reply.as_ref().err()),
        sent
    );
    Ok(reply
        .map(|reply| reply["result"].clone())
        .unwrap_or_default())
}

fn refuse(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Result<Value, SmokeFailure> {
    let sent = json!({"verb": verb, "body": body});
    let reply = session.request(verb, body);
    smoke_check!(
        reply.is_err(),
        "request_refused",
        json!(reply.as_ref().ok()),
        sent
    );
    Ok(reply
        .err()
        .map(|reply| reply["error"].clone())
        .unwrap_or_default())
}
