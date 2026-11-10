use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, control, environment, query, render, settled};
use serde_json::{Value, json};

use super::requests::seen;

const BOXES: usize = 5;
const LIMIT: usize = 3;
const COLLIDER_PART: &str = "collider";
const OVER_LIMIT: &str = "collision_collider_budget";
const ACTIVE: &str = "active";
const LEFT_OUT: &str = "over_budget";

fn named(index: usize) -> String {
    format!("box-{index}")
}

fn across(index: usize) -> f64 {
    index as f64 * 4.0 - 8.0
}

fn fixture() -> Value {
    let mut entities: Vec<Value> = (0..BOXES)
        .map(|index| {
            json!({
                "id": named(index),
                "transform": {"translation": [across(index), 0.5, 0.0]},
                "collider": {"shape": {"kind": "cuboid", "half_extents": [0.5, 0.5, 0.5]}},
            })
        })
        .collect();
    entities.push(json!({
        "id": "camera", "transform": {"translation": [0.0, 10.0, 18.0], "rotation_degrees": [-30.0, 0.0, 0.0]},
        "node": {"clock": "real", "mobility": "moving"},
        "camera": {"near": 0.1, "far": 200.0, "projection": {"Perspective": {"vertical_fov_radians": 0.9}}},
        "view": {},
    }));
    json!({
        "id": "collider-limit", "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": entities,
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&fixture().to_string());
    control(session, "pause", json!({}))?;
    environment(
        session,
        json!({"merge": true, "collision": {"max_colliders": LIMIT}}),
    )?;
    settled(session)?;
    session.step(1);
    render(session)?;
    let placed = places(session)?;
    let left_out: Vec<usize> = placed
        .iter()
        .enumerate()
        .filter(|(_, status)| *status == LEFT_OUT)
        .map(|(index, _)| index)
        .collect();
    let kept: Vec<usize> = placed
        .iter()
        .enumerate()
        .filter(|(_, status)| *status == ACTIVE)
        .map(|(index, _)| index)
        .collect();
    let wanted = json!({"active": LIMIT, "over_budget": BOXES - LIMIT});
    smoke_eq!(
        seen(
            kept.len() == LIMIT && left_out.len() == BOXES - LIMIT,
            json!(placed),
            &wanted,
        ),
        wanted,
        "only_the_colliders_past_the_limit_are_left_out"
    );
    reasons(
        session,
        &left_out,
        "why_names_the_limit_for_the_left_out_alone",
    )?;
    rays(session, &kept, &left_out)?;
    environment(
        session,
        json!({"merge": true, "collision": {"max_colliders": null}}),
    )?;
    session.step(1);
    render(session)?;
    let placed = places(session)?;
    let wanted = Value::from(vec![ACTIVE; BOXES]);
    smoke_eq!(
        json!(placed),
        wanted,
        "lifting_the_limit_takes_every_collider_back"
    );
    reasons(session, &[], "lifting_the_limit_clears_the_refusals")
}

fn places(session: &mut dyn SmokeSession) -> Checked<Vec<String>> {
    let mut placed = Vec::with_capacity(BOXES);
    for index in 0..BOXES {
        let view = query(session, "collider", json!({ "entity": named(index) }))?;
        placed.push(view["cache_status"].as_str().unwrap_or_default().to_owned());
    }
    Ok(placed)
}

fn reasons(session: &mut dyn SmokeSession, left_out: &[usize], check: &'static str) -> Checked<()> {
    let mut got = Vec::with_capacity(BOXES);
    let mut wanted = Vec::with_capacity(BOXES);
    for index in 0..BOXES {
        let why = query(session, "why", json!({ "entity": named(index) }))?;
        let refused: Vec<Value> = why["refused"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|row| row["part"] == COLLIDER_PART)
            .map(|row| row["reason"]["reason"].clone())
            .collect();
        got.push(json!({ "entity": named(index), "refused": refused }));
        let expected: Vec<&str> = if left_out.contains(&index) {
            vec![OVER_LIMIT]
        } else {
            Vec::new()
        };
        wanted.push(json!({ "entity": named(index), "refused": expected }));
    }
    smoke_eq!(json!(got), json!(wanted), check);
    Ok(())
}

fn rays(session: &mut dyn SmokeSession, kept: &[usize], left_out: &[usize]) -> Checked<()> {
    let mut got = Vec::new();
    let mut wanted = Vec::new();
    for (index, hits) in kept
        .iter()
        .map(|&index| (index, true))
        .chain(left_out.iter().map(|&index| (index, false)))
    {
        let ray = query(
            session,
            "raycast",
            json!({"origin": [across(index), 10.0, 0.0], "direction": [0.0, -1.0, 0.0], "max_distance": 50.0}),
        )?;
        let hit = if hits {
            json!(named(index))
        } else {
            Value::Null
        };
        got.push(json!({
            "entity": named(index),
            "hit": ray["hit"]["entity"]["id"],
            "overflow": ray["snapshot"]["overflow_colliders"],
            "complete": ray["snapshot"]["complete"],
        }));
        wanted.push(json!({
            "entity": named(index),
            "hit": hit,
            "overflow": BOXES - LIMIT,
            "complete": true,
        }));
    }
    smoke_eq!(
        json!(got),
        json!(wanted),
        "queries_keep_working_over_the_limit"
    );
    Ok(())
}
