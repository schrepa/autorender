use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, component, control, near, query, render, settled};
use serde_json::{Value, json};

use super::requests::seen;

const BOX: &str = "box";
const TARGET: [f64; 3] = [5.0, 1.0, -2.0];
const SPAWNED_AT: [f64; 3] = [-3.0, 0.5, 2.0];
const EPSILON: f64 = 1e-4;

fn fixture() -> Value {
    json!({
        "id": "unpaused-calls", "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {"id": BOX, "transform": {"translation": [0.0, 0.5, 0.0]}, "node": {"mobility": "moving"}},
            {"id": "camera", "transform": {"translation": [0.0, 8.0, 14.0], "rotation_degrees": [-30.0, 0.0, 0.0]},
             "node": {"clock": "real", "mobility": "moving"},
             "camera": {"near": 0.1, "far": 200.0, "projection": {"Perspective": {"vertical_fov_radians": 0.9}}},
             "view": {}},
        ],
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&fixture().to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    let resumed = control(session, "resume", json!({}))?;
    smoke_eq!(resumed["paused"], json!(false), "the_session_runs");
    let spawned = called(session)?;
    control(session, "pause", json!({}))?;
    render(session)?;
    landed(session, &spawned)
}

fn called(session: &mut dyn SmokeSession) -> Checked<String> {
    let teleported = control(
        session,
        "teleport",
        json!({ "moves": [{"entity": BOX, "translation": TARGET}] }),
    )?;
    let spawned = control(
        session,
        "spawn",
        json!({ "entities": [{"node": {"mobility": "moving"}, "transform": {"translation": SPAWNED_AT}}] }),
    )?;
    let collider = json!({"shape": {"kind": "cuboid", "half_extents": [0.5, 0.5, 0.5]}});
    let changed = control(
        session,
        "change",
        json!({ "changes": [{"entity": BOX, "component": "collider", "data": collider}] }),
    )?;
    let got = json!({
        "paused": [teleported["paused"], spawned["paused"], changed["paused"]],
        "done": [teleported["done"], spawned["done"], changed["done"]],
        "lands": teleported["rows"][0]["done"]["lands"],
    });
    let wanted = json!({
        "paused": [false, false, false],
        "done": [1, 1, 1],
        "lands": "next_frame",
    });
    smoke_eq!(got, wanted, "calls_by_name_are_taken_while_running");
    Ok(spawned["rows"][0]["done"]["entity"]
        .as_str()
        .unwrap_or_default()
        .to_owned())
}

fn translation(session: &mut dyn SmokeSession, id: &str) -> Checked<Value> {
    let mut placed = component(session, id, "GlobalTransform")?;
    Ok(placed["translation"].take())
}

fn near_all(values: &Value, expected: &[f64]) -> bool {
    values.as_array().is_some_and(|values| {
        values.len() == expected.len()
            && values.iter().zip(expected).all(|(value, &expected)| {
                value
                    .as_f64()
                    .is_some_and(|value| (value - expected).abs() <= EPSILON)
            })
    })
}

fn landed(session: &mut dyn SmokeSession, spawned: &str) -> Checked<()> {
    let moved = translation(session, BOX)?;
    let placed = translation(session, spawned)?;
    let collider = query(session, "collider", json!({ "entity": BOX }))?;
    let got = json!({
        "teleported": moved,
        "spawned": placed,
        "collider": collider["cache_status"],
    });
    let wanted = json!({
        "teleported": near(TARGET, EPSILON),
        "spawned": near(SPAWNED_AT, EPSILON),
        "collider": "active",
    });
    let held = near_all(&moved, &TARGET)
        && near_all(&placed, &SPAWNED_AT)
        && collider["cache_status"] == "active";
    smoke_eq!(
        seen(held, got, &wanted),
        wanted,
        "calls_made_while_running_land_at_the_next_frame"
    );
    Ok(())
}
