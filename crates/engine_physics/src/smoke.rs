use std::collections::BTreeMap;
use std::sync::Arc;

use bevy_ecs::prelude::{Entity, World};
use engine_collision::{Collider, ColliderShape, CollisionError};
use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{AuthoredId, OperationScene, Transform, set_parent, smoke_check, smoke_eq};
use glam::Vec3;
use serde_json::{Value, json};

use crate::RigidBody;

const SCENE: &str = "physics-fixture";

pub(crate) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let built = build(session.app().world_mut());
    smoke_check!(
        built.is_ok(),
        "fixture_builds",
        json!(built.as_ref().err().map(|error| format!("{error:?}"))),
        json!(null)
    );
    ask(session, "control", json!({"name": "pause"}))?;
    ask(
        session,
        "control",
        json!({"name": "environment", "args": {"merge": true, "physics": {"water_density": 1000}}}),
    )?;
    for name in ["RigidBody", "PhysicsWorld", "Physics"] {
        ask(session, "describe", json!({"name": name}))?;
    }
    let settings = ask(session, "inspect", json!({"component": "Physics"}))?["value"].clone();
    smoke_check!(
        settings["enabled"] == true
            && settings["gravity"][1]
                .as_f64()
                .is_some_and(|gravity| (gravity + 9.81).abs() < 1e-4),
        "settings_inspect",
        json!(settings),
        json!({"enabled": true, "gravity": [0, -9.81, 0]})
    );
    rests(session)?;
    toggles(session)?;
    parented(session)?;
    session.restart(&["collision", "physics"]);
    let missing = refuse(session, "inspect", json!({"component": "PhysicsWorld"}))?;
    smoke_eq!(json!(missing["code"]), json!("missing_plugin"), "left_out_without_collision");
    Ok(())
}

fn rests(session: &mut dyn SmokeSession) -> SmokeResult {
    seconds(session, 3);
    let crate_at = component(session, "crate", "Transform")?;
    smoke_check!(
        crate_at["translation"][1]
            .as_f64()
            .is_some_and(|height| (height - 10.5).abs() < 0.05),
        "falls_and_rests",
        json!(crate_at),
        json!({"translation": [0, 10.5, 0]})
    );
    let body = component(session, "crate", "RigidBody")?;
    smoke_check!(
        body["simulated"] == true
            && body["problem"].is_null()
            && body["mass"]
                .as_f64()
                .is_some_and(|mass| (mass - 600.0).abs() < 1.0),
        "body_inspects",
        json!(body),
        json!({"simulated": true, "problem": null, "mass": 600})
    );
    let stone = component(session, "stone", "Transform")?;
    smoke_check!(
        stone["translation"][1]
            .as_f64()
            .is_some_and(|height| height < -2.0),
        "sinks",
        json!(stone),
        json!({"translation": [32, -2, 0]})
    );
    let world = ask(session, "inspect", json!({"component": "PhysicsWorld"}))?["value"].clone();
    smoke_check!(
        world["bodies"] == 2 && world["held"] == 0 && world["solids"].as_u64().unwrap_or(0) >= 2,
        "world_inspects",
        json!(world),
        json!({"bodies": 2, "held": 0, "solids": 2})
    );
    let status = ask(session, "query", json!({"name": "status"}))?;
    smoke_eq!(json!(status["rendering"]["physics"]["bodies"]), json!(2), "frames_observe");
    Ok(())
}

fn toggles(session: &mut dyn SmokeSession) -> SmokeResult {
    ask(
        session,
        "control",
        json!({"name": "environment", "args": {"merge": true, "physics": {"enabled": false}}}),
    )?;
    let held = component(session, "stone", "Transform")?;
    seconds(session, 1);
    smoke_eq!(component(session, "stone", "Transform")?, held, "physics_off_holds");
    ask(
        session,
        "control",
        json!({"name": "environment", "args": {"merge": true, "physics": {"enabled": true}}}),
    )?;
    seconds(session, 1);
    let moved = component(session, "stone", "Transform")?;
    smoke_check!(moved != held, "physics_on_moves", json!(moved), json!(held));
    Ok(())
}

fn parented(session: &mut dyn SmokeSession) -> SmokeResult {
    let before = component(session, "crate", "Transform")?;
    let hung = hang(session.app().world_mut());
    smoke_check!(
        hung.is_ok(),
        "parented_body_builds",
        json!(hung.as_ref().err().map(|error| format!("{error:?}"))),
        json!(null)
    );
    session.step(1);
    let parented = component(session, "hung", "RigidBody")?;
    smoke_check!(
        parented["problem"] == "not_root" && parented["simulated"] == false,
        "parented_body_refused",
        json!(parented),
        json!({"problem": "not_root", "simulated": false})
    );
    smoke_eq!(component(session, "crate", "Transform")?, before, "parented_body_leaves_world");
    Ok(())
}

fn build(world: &mut World) -> Result<(), CollisionError> {
    let solid = |half: [f32; 3]| {
        Collider::new(ColliderShape::Cuboid {
            half_extents: Vec3::from_array(half),
        })
    };
    let floor = world
        .spawn((
            Transform::from_xyz(0.0, 9.5, 0.0),
            solid([10.0, 0.5, 10.0])?,
        ))
        .id();
    let crate_body = world
        .spawn((
            Transform::from_xyz(0.0, 14.0, 0.0),
            solid([0.5; 3])?,
            RigidBody::default(),
        ))
        .id();
    let seabed = world
        .spawn((
            Transform::from_xyz(30.0, -12.5, 0.0),
            solid([20.0, 0.5, 20.0])?,
        ))
        .id();
    let stone = world
        .spawn((
            Transform::from_xyz(32.0, 40.0, 0.0),
            Collider::new(ColliderShape::Sphere { radius: 0.5 })?,
            RigidBody {
                density: 2600.0,
                ..RigidBody::default()
            },
        ))
        .id();
    name(
        world,
        [
            ("floor", floor),
            ("crate", crate_body),
            ("seabed", seabed),
            ("stone", stone),
        ]
        .into_iter()
        .map(|(id, entity)| (id.to_owned(), entity))
        .collect(),
    );
    Ok(())
}

fn hang(world: &mut World) -> Result<(), CollisionError> {
    let Some(floor) = entity(world, "floor") else {
        return Err(CollisionError::InvalidTransform);
    };
    let hung = world
        .spawn((
            Transform::from_xyz(0.0, 4.0, 0.0),
            Collider::new(ColliderShape::Sphere { radius: 0.5 })?,
            RigidBody::default(),
        ))
        .id();
    if set_parent(world, hung, Some(floor)).is_err() {
        return Err(CollisionError::InvalidTransform);
    }
    let mut named = world
        .get_resource::<OperationScene>()
        .map(|scene| (*scene.entities).clone())
        .unwrap_or_default();
    named.insert("hung".to_owned(), hung);
    name(world, named);
    Ok(())
}

fn name(world: &mut World, entities: BTreeMap<String, Entity>) {
    world.insert_resource(OperationScene {
        id: SCENE.to_owned(),
        entities: Arc::new(entities),
        streamed: false,
        authored,
    });
}

fn authored(world: &World, entity: Entity) -> Option<AuthoredId> {
    let scene = world.get_resource::<OperationScene>()?;
    let (id, _) = scene
        .entities
        .iter()
        .find(|&(_, &named)| named == entity)?;
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

fn seconds(session: &mut dyn SmokeSession, count: u32) {
    for _ in (0..count * 60).step_by(120) {
        session.step(120);
    }
}

fn component(
    session: &mut dyn SmokeSession,
    id: &str,
    component: &str,
) -> Result<Value, SmokeFailure> {
    let inspected = ask(session, "inspect", json!({"entity": id, "component": component}))?;
    Ok(inspected["entity"]["components"][component].clone())
}

fn ask(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Result<Value, SmokeFailure> {
    let sent = json!({"verb": verb, "body": body});
    let reply = session.request(verb, body);
    smoke_check!(reply.is_ok(), "request_accepted", json!(reply.as_ref().err()), sent);
    Ok(reply.map(|reply| reply["result"].clone()).unwrap_or_default())
}

fn refuse(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Result<Value, SmokeFailure> {
    let sent = json!({"verb": verb, "body": body});
    let reply = session.request(verb, body);
    smoke_check!(reply.is_err(), "request_refused", json!(reply.as_ref().ok()), sent);
    Ok(reply.err().map(|reply| reply["error"].clone()).unwrap_or_default())
}
