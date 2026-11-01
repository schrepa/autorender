use engine_core::smoke::{Checked, ask, component, control, inspect, query, refuse};
use engine_core::{ErrorCode, Section, SmokeFailure, SmokeSession, smoke_eq};
use engine_places::physics::{Collider, ColliderShape, MAX_COLLISION_COORDINATE};
use glam::Vec3;
use serde_json::{Value, json};

use crate::types::CollisionStats;

const SCENE: &str = "collision-fixture";
const TICK_DISTANCE: f32 = 6.5 / 60.0;

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    load(session, &fixture(true, &[]))?;
    let ray = queries(session)?;
    sensor(session, &ray)?;
    sweep(session)?;
    changes(session, &ray)?;
    control(session, "resume", json!({}))?;
    query(session, "raycast", ray)?;
    control(session, "pause", json!({}))?;
    Ok(())
}

fn queries(session: &mut dyn SmokeSession) -> Checked<Value> {
    for name in ["collider", CollisionStats::KEY] {
        ask(session, "describe", json!({"name": name}))?;
    }
    let world = inspect(session, CollisionStats::KEY)?;
    smoke_eq!(
        json!({
            "complete": world["complete"],
            "colliders": world["colliders"].as_u64().is_some_and(|count| count > 0),
            "invalid_colliders": world["invalid_colliders"],
            "overflow_colliders": world["overflow_colliders"],
            "bvh_nodes": world["bvh_nodes"].as_u64().is_some_and(|count| count > 0),
        }),
        json!({
            "complete": true,
            "colliders": true,
            "invalid_colliders": 0,
            "overflow_colliders": 0,
            "bvh_nodes": true,
        }),
        "collision_world_inspects"
    );
    let volume = query(session, "collider", json!({"entity": "player"}))?;
    smoke_eq!(volume.is_object(), true, "collider_inspects");
    let before = state(session)?;

    let filter = json!({"memberships": 2, "filter": 1, "sensors": "exclude"});
    let ray =
        json!({"origin": [4, 4, 0], "direction": [0, -7, 0], "max_distance": 5, "filter": filter});
    let hit = query(session, "raycast", ray.clone())?["hit"].clone();
    smoke_eq!(hit["entity"]["id"], json!("target"), "closest_raycast");
    smoke_eq!(
        near(&hit["distance"], 3.0),
        json!(3.0),
        "closest_raycast_distance"
    );
    smoke_eq!(
        near(&hit["normal"][1], 1.0),
        json!(1.0),
        "closest_raycast_normal"
    );
    let layered = query(
        session,
        "raycast",
        with(&ray, "filter", json!({"memberships": 8, "filter": 1})),
    )?;
    smoke_eq!(layered["hit"], Value::Null, "symmetric_layers");

    let sensor_filter = json!({"memberships": 2, "filter": 4, "sensors": "only"});
    let sensor_ray = json!({"origin": [0, 0.6, 0], "direction": [0, 0, -1], "max_distance": 3, "filter": sensor_filter});
    let sensed = query(session, "raycast", sensor_ray.clone())?;
    smoke_eq!(
        sensed["hit"]["entity"]["id"],
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
    smoke_eq!(excluded["hit"], Value::Null, "sensor_filter_exclude");
    let skipped = query(
        session,
        "raycast",
        with(
            &sensor_ray,
            "filter",
            with(&sensor_filter, "exclude", json!("sensor")),
        ),
    )?;
    smoke_eq!(skipped["hit"], Value::Null, "sensor_exclusion");

    shapes(session, &ray, &filter)?;
    refuse(
        session,
        "query",
        "raycast",
        with(&ray, "direction", json!([0, 0, 0])),
    )?;
    refuse(
        session,
        "query",
        "overlap",
        json!({"position": [0, 0, 0], "shape": {"kind": "sphere", "radius": 20}, "filter": filter, "limit": 1}),
    )?;
    smoke_eq!(state(session)?, before, "invalid_and_overflow_nonmutating");
    Ok(ray)
}

fn shapes(session: &mut dyn SmokeSession, ray: &Value, filter: &Value) -> Checked<()> {
    let inside = query(session, "raycast", with(ray, "origin", json!([4, 0.5, 0])))?["hit"].clone();
    smoke_eq!(inside["started_inside"], json!(true), "inside_hit");
    smoke_eq!(
        near(&inside["distance"], 0.0),
        json!(0.0),
        "inside_hit_distance"
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
    smoke_eq!(cast["entity"]["id"], json!("target"), "shape_cast");
    smoke_eq!(
        near(&cast["distance"], 2.75),
        json!(2.75),
        "shape_cast_distance"
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
        smoke_eq!(hit["entity"]["id"], json!(id), "shear_and_ellipsoid");
        smoke_eq!(
            near(&hit["distance"], 3.5),
            json!(3.5),
            "shear_and_ellipsoid_distance"
        );
    }
    let sheared_cast = query(
        session,
        "shape_cast",
        json!({"position": [8, 5, 0], "shape": cuboid(0.25), "direction": [0, -1, 0], "max_distance": 5, "filter": filter}),
    )?["hit"]
        .clone();
    smoke_eq!(
        sheared_cast["entity"]["id"],
        json!("sheared"),
        "sheared_cast"
    );
    smoke_eq!(
        near(&sheared_cast["distance"], 3.25),
        json!(3.25),
        "sheared_cast_distance"
    );
    Ok(())
}

fn sensor(session: &mut dyn SmokeSession, ray: &Value) -> Checked<()> {
    let mut kinds = Vec::new();
    let mut z = 0.0_f32;
    for tick in 0..36 {
        z -= TICK_DISTANCE;
        let moved = edit(session, "player", [0.0, 0.6, z]);
        smoke_eq!(
            if moved { json!("moved") } else { json!(tick) },
            json!("moved"),
            "player_moves"
        );
        session.step(1);
        let events = sensor_events(session)?;
        let rows = events["events"].as_array().cloned().unwrap_or_default();
        kinds.extend(rows.iter().map(|event| event["kind"].clone()));
        if rows.is_empty() {
            continue;
        }
        let exact = state(session)?;
        query(session, "raycast", ray.clone())?;
        smoke_eq!(sensor_events(session)?, events, "events_nonconsuming");
        smoke_eq!(state(session)?, exact, "queries_preserve_clock");
        control(session, "present", json!({"alpha": 0.5}))?;
        let preview = state(session)?;
        query(session, "raycast", ray.clone())?;
        smoke_eq!(state(session)?, preview, "queries_preserve_preview");
        smoke_eq!(sensor_events(session)?, events, "events_survive_preview");
    }
    smoke_eq!(json!(kinds), json!(["enter", "exit"]), "sensor_enter_exit");
    Ok(())
}

fn sweep(session: &mut dyn SmokeSession) -> Checked<()> {
    smoke_eq!(
        edit(session, "player", [0.0, 0.6, 0.0]),
        true,
        "player_returns"
    );
    smoke_eq!(
        Collider::cuboid(Vec3::new(3.0, 2.0, 0.01)).is_ok(),
        true,
        "thin_wall_builds"
    );
    let wall = json!({
        "id": "thin-wall",
        "transform": placed([0.66, 0.6, 0.0], [0.0, 90.0, 0.0], [1.0; 3]),
        "collider": collider(cuboid_shape([3.0, 2.0, 0.01]), 1, 2, false),
    });
    load(session, &fixture(true, &[wall]))?;
    let swept = query(
        session,
        "shape_cast",
        json!({
            "position": [0, 0.65, 0], "shape": cuboid(0.6), "direction": [1, 0, 0], "max_distance": 2,
            "filter": {"memberships": 2, "filter": 1, "sensors": "exclude", "exclude": "player"},
        }),
    )?["hit"]
        .clone();
    smoke_eq!(swept["entity"]["id"], json!("thin-wall"), "swept_thin_wall");
    smoke_eq!(
        near(&swept["distance"], 0.05),
        json!(0.05),
        "swept_thin_wall_distance"
    );
    Ok(())
}

fn changes(session: &mut dyn SmokeSession, ray: &Value) -> Checked<()> {
    let before = state(session)?;
    let refused = Collider::new(ColliderShape::Cuboid {
        half_extents: Vec3::new(0.0, 0.5, 0.5),
    });
    smoke_eq!(
        json!(refused.as_ref().err().map(|error| error.code)),
        json!(ErrorCode::CollisionInvalidShape),
        "invalid_shape_refused"
    );
    smoke_eq!(
        Collider::cuboid(Vec3::splat(0.5)).is_ok(),
        true,
        "flat_collider_builds"
    );
    let flat = json!({
        "id": "flat",
        "transform": placed([12.0, 0.5, 0.0], [0.0; 3], [1.0; 3]),
        "collider": collider(cuboid_shape([0.5; 3]), 1, 2, false),
    });
    load(session, &fixture(true, &[flat]))?;
    control(
        session,
        "edit_transform",
        json!({"entity": "flat", "translation": [MAX_COLLISION_COORDINATE + 1000.0, 0.5, 0.0]}),
    )?;
    session.step(1);
    let counted = inspect(session, CollisionStats::KEY)?;
    smoke_eq!(
        counted["invalid_colliders"],
        json!(1),
        "invalid_collider_counted"
    );
    let incomplete = refuse(session, "query", "raycast", ray.clone())?;
    smoke_eq!(
        incomplete["code"],
        json!("collision_incomplete_world"),
        "invalid_collider_blocks_queries"
    );
    smoke_eq!(
        component(session, "player", "Transform")?,
        before["transform"],
        "invalid_collider_preserves_player"
    );
    load(session, &fixture(false, &[]))?;
    let retired = query(session, "raycast", with(ray, "max_distance", json!(3.5)))?;
    smoke_eq!(retired["hit"], Value::Null, "removed_colliders_retire");
    let cleared = inspect(session, CollisionStats::KEY)?;
    smoke_eq!(
        cleared["invalid_colliders"],
        json!(0),
        "invalid_collider_retires"
    );
    Ok(())
}

fn load(session: &mut dyn SmokeSession, scene: &Value) -> Checked<()> {
    session.source(&scene.to_string());
    control(session, "pause", json!({}))?;
    session.step(1);
    Ok(())
}

fn edit(session: &mut dyn SmokeSession, id: &str, translation: [f32; 3]) -> bool {
    session
        .request(
            "control",
            json!({"name": "edit_transform", "args": {"entity": id, "translation": translation}}),
        )
        .is_ok()
}

fn fixture(target: bool, extra: &[Value]) -> Value {
    let mut entities = vec![
        json!({
            "id": "player",
            "transform": placed([0.0, 0.6, 0.0], [0.0; 3], [1.2; 3]),
            "collider": collider(cuboid_shape([0.5; 3]), 2, 5, false),
        }),
        json!({
            "id": "floor",
            "transform": placed([0.0, -0.5, 0.0], [0.0; 3], [1.0; 3]),
            "collider": collider(cuboid_shape([20.0, 0.5, 20.0]), 1, 2, false),
        }),
    ];
    if target {
        entities.push(json!({
            "id": "target",
            "transform": placed([4.0, 0.5, 0.0], [0.0; 3], [1.0; 3]),
            "collider": collider(cuboid_shape([0.5; 3]), 1, 2, false),
        }));
    }
    entities.extend([
        json!({
            "id": "sensor",
            "transform": placed([0.0, 0.6, -2.0], [0.0; 3], [1.0; 3]),
            "collider": collider(json!({"kind": "sphere", "radius": 0.4}), 4, 2, true),
        }),
        json!({
            "id": "parent",
            "transform": placed([8.0, 1.0, 0.0], [0.0, 30.0, 0.0], [2.0, 1.0, 0.5]),
        }),
        json!({
            "id": "sheared",
            "parent": "parent",
            "transform": placed([0.0; 3], [0.0, 40.0, 0.0], [1.0; 3]),
            "collider": collider(cuboid_shape([0.5; 3]), 1, 2, false),
        }),
        json!({
            "id": "ellipsoid",
            "transform": placed([-4.0, 1.0, 0.0], [0.0; 3], [2.0, 1.0, 0.5]),
            "collider": collider(json!({"kind": "sphere", "radius": 0.5}), 1, 2, false),
        }),
    ]);
    entities.extend_from_slice(extra);
    json!({
        "id": SCENE,
        "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": entities,
    })
}

fn collider(shape: Value, memberships: u32, filter: u32, sensor: bool) -> Value {
    json!({"shape": shape, "memberships": memberships, "filter": filter, "sensor": sensor})
}

fn cuboid_shape(half_extents: [f32; 3]) -> Value {
    json!({"kind": "cuboid", "half_extents": half_extents})
}

fn cuboid(half: f64) -> Value {
    json!({"kind": "cuboid", "half_extents": [half, half, half]})
}

fn placed(translation: [f32; 3], degrees: [f32; 3], scale: [f32; 3]) -> Value {
    json!({"translation": translation, "rotation_degrees": degrees, "scale": scale})
}

fn state(session: &mut dyn SmokeSession) -> Checked<Value> {
    let status = ask(session, "query", json!({"name": "status"}))?;
    Ok(json!({
        "frame": status["observation"]["frame"],
        "tick": status["observation"]["tick"],
        "presentation": status["observation"]["presentation"],
        "transform": component(session, "player", "Transform")?,
        "global": component(session, "player", "GlobalTransform")?,
    }))
}

fn sensor_events(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut events = query(session, "collision_events", json!({"entity": "sensor"}))?;
    if let Some(fields) = events.as_object_mut() {
        for key in ["observation", "rendering", "paused", "source"] {
            fields.remove(key);
        }
    }
    Ok(events)
}

fn near(value: &Value, expected: f64) -> Value {
    if value
        .as_f64()
        .is_some_and(|value| (value - expected).abs() < 0.001)
    {
        json!(expected)
    } else {
        value.clone()
    }
}

fn with(base: &Value, key: &str, value: Value) -> Value {
    let mut changed = base.clone();
    changed[key] = value;
    changed
}
