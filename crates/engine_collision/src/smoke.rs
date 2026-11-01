use engine_core::smoke::{Checked, control, inspect, query, refuse};
use engine_core::{Section, SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use crate::types::CollisionStats;

const SCENE: &str = "collision-fixture";

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    load(session, &fixture(0.0, true, &[]))?;
    let ray = queries(session)?;
    sensor(session)?;
    sweep(session)?;
    changes(session, &ray)?;
    control(session, "resume", json!({}))?;
    query(session, "raycast", ray)?;
    control(session, "pause", json!({}))?;
    Ok(())
}

fn queries(session: &mut dyn SmokeSession) -> Checked<Value> {
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
    smoke_eq!(
        json!([volume["world_valid"], volume["cache_status"]]),
        json!([true, "active"]),
        "collider_inspects"
    );

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

fn sensor(session: &mut dyn SmokeSession) -> Checked<()> {
    load(session, &fixture(-2.0, true, &[]))?;
    let entered = query(session, "collision_events", json!({"entity": "sensor"}))?;
    smoke_eq!(kinds(&entered), json!(["enter"]), "sensor_enters");
    load(session, &fixture(-4.0, true, &[]))?;
    let left = query(session, "collision_events", json!({}))?;
    smoke_eq!(kinds(&left), json!(["exit"]), "sensor_exits");
    Ok(())
}

fn sweep(session: &mut dyn SmokeSession) -> Checked<()> {
    let wall = json!({
        "id": "thin-wall",
        "transform": placed([0.66, 0.6, 0.0], [0.0, 90.0, 0.0], [1.0; 3]),
        "collider": collider(cuboid_shape([3.0, 2.0, 0.01]), 1, 2, false),
    });
    load(session, &fixture(0.0, true, &[wall]))?;
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
    let flat = json!({
        "id": "flat",
        "transform": placed([12.0, 0.5, 0.0], [0.0; 3], [0.0001, 1.0, 1.0]),
        "collider": collider(cuboid_shape([0.5; 3]), 1, 2, false),
    });
    load(session, &fixture(0.0, true, &[flat]))?;
    let counted = inspect(session, CollisionStats::KEY)?;
    smoke_eq!(
        counted["invalid_colliders"],
        json!(1),
        "invalid_collider_counted"
    );
    let incomplete = refuse(session, "query", "raycast", ray.clone())?;
    smoke_eq!(
        incomplete["code"],
        json!("invalid_value"),
        "invalid_collider_blocks_queries"
    );
    load(session, &fixture(0.0, false, &[]))?;
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

fn fixture(player: f32, target: bool, extra: &[Value]) -> Value {
    let mut entities = vec![
        json!({
            "id": "player",
            "transform": placed([0.0, 0.6, player], [0.0; 3], [1.2; 3]),
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

fn kinds(events: &Value) -> Value {
    events["events"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|event| event["kind"].clone())
        .collect()
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
