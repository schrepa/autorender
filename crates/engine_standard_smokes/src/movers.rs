use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, component, control, near, number, query, render, settled};
use serde_json::{Value, json};

use super::requests::seen;

const CRATE: &str = "crate";
const HELD_AT: f64 = 6.0;
const OFFSET: f64 = 0.5;
const EPSILON: f64 = 1e-3;
const TICKS: u32 = 4;
const MOVER_PART: &str = "mover";

fn fixture() -> Value {
    json!({
        "id": "movers", "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {"id": "floor", "transform": {"translation": [0.0, -0.5, 0.0]},
             "collider": {"shape": {"kind": "cuboid", "half_extents": [16.0, 0.5, 16.0]}}},
            {"id": CRATE, "transform": {"translation": [3.0, HELD_AT, 0.0]},
             "node": {"mobility": "moving"},
             "collider": {"shape": {"kind": "cuboid", "half_extents": [0.5, 0.5, 0.5]}},
             "rigid_body": {"motion": "kinematic"}},
            {"id": "camera", "transform": {"translation": [0.0, 12.0, 16.0], "rotation_degrees": [-35.0, 0.0, 0.0]},
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
    let mut held = why(session)?;
    let holder = held["mover"]["by"].take();
    let grounded = json!({"offset": OFFSET});
    changed(session, "grounded", &grounded)?;
    session.step(TICKS);
    let claimant = refused(session, &holder)?;
    changed(session, "rigid_body", &Value::Null)?;
    session.step(TICKS);
    render(session)?;
    released(session, &claimant)
}

fn why(session: &mut dyn SmokeSession) -> Checked<Value> {
    query(session, "why", json!({ "entity": CRATE }))
}

fn mover_rows(why: &Value) -> Vec<Value> {
    why["refused"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|row| row["part"] == MOVER_PART)
        .cloned()
        .collect()
}

fn height(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut placed = component(session, CRATE, "GlobalTransform")?;
    Ok(placed["translation"][1].take())
}

fn changed(session: &mut dyn SmokeSession, kind: &str, data: &Value) -> Checked<()> {
    let change = json!({"entity": CRATE, "component": kind, "data": data});
    let reply = control(session, "change", json!({ "changes": [change] }))?;
    let got = json!({"done": reply["done"], "refused": reply["refused"]});
    let wanted = json!({"done": 1, "refused": 0});
    smoke_eq!(
        seen(got == wanted, reply["rows"].clone(), &wanted),
        wanted,
        "change_lands"
    );
    Ok(())
}

fn refused(session: &mut dyn SmokeSession, holder: &Value) -> Checked<Value> {
    let answer = why(session)?;
    let rows = mover_rows(&answer);
    let claimant = rows
        .first()
        .map(|row| row["reason"]["got"].clone())
        .unwrap_or_default();
    let stood = height(session)?;
    let got = json!({
        "holder": answer["mover"]["by"],
        "rows": rows.len(),
        "reason": rows.first().map(|row| row["reason"]["reason"].clone()),
        "nearest": rows.first().map(|row| row["reason"]["nearest"].clone()),
        "claimant": claimant,
        "height": stood,
    });
    let wanted = json!({
        "holder": holder,
        "rows": 1,
        "reason": "usage_conflict",
        "nearest": holder,
        "claimant": "a second mover",
        "height": near(HELD_AT, EPSILON),
    });
    let held = holder.is_string()
        && answer["mover"]["by"] == *holder
        && rows.len() == 1
        && got["reason"] == "usage_conflict"
        && got["nearest"] == *holder
        && claimant.is_string()
        && claimant != *holder
        && (number(&stood) - HELD_AT).abs() < EPSILON;
    smoke_eq!(
        seen(held, got, &wanted),
        wanted,
        "why_names_the_refused_second_mover"
    );
    Ok(claimant)
}

fn released(session: &mut dyn SmokeSession, claimant: &Value) -> Checked<()> {
    let answer = why(session)?;
    let rows = mover_rows(&answer);
    let stood = height(session)?;
    let got = json!({
        "mover": answer["mover"]["by"],
        "refused": rows,
        "height": stood,
    });
    let wanted = json!({
        "mover": claimant,
        "refused": [],
        "height": near(OFFSET, EPSILON),
    });
    let held = answer["mover"]["by"] == *claimant
        && rows.is_empty()
        && (number(&stood) - OFFSET).abs() < EPSILON;
    smoke_eq!(
        seen(held, got, &wanted),
        wanted,
        "removing_the_holder_hands_the_entity_to_the_other_mover"
    );
    Ok(())
}
