use engine_core::{SmokeFailure, SmokeSession, TargetName, smoke_check, smoke_eq};
use engine_smoke::{Checked, control, environment, query, render, settled};
use serde_json::{Value, json};

use super::image::image;
use super::requests::{grid, seen};

pub(super) const LEFT_OUT: &str = "engine_prepass";
pub(super) const GROUND_FILLER: &str = "engine_collision";
const OCCLUSION: &str = "engine_ambient_occlusion";
const GROUNDING: &str = "engine_grounding";
const GROUND: &str = "ground";
const OCCLUSION_PASSES: [&str; 2] = ["ambient_occlusion", "effects_filter"];
const UNWRITTEN: &str = "read_before_write";
const UNFILLED: &str = "place_not_filled";

fn brick_of(row: &Value) -> Option<&str> {
    row["path"]
        .as_str()?
        .strip_prefix("$.bricks.")?
        .split('.')
        .next()
        .filter(|brick| !brick.is_empty())
}

fn names_a_pass(row: &Value) -> bool {
    row["path"]
        .as_str()
        .is_some_and(|path| path.contains(".passes."))
}

fn named_by_why(session: &mut dyn SmokeSession, row: &Value, brick: &str) -> Checked<()> {
    let why = query(session, "why", json!({ "brick": brick }))?;
    let named = why["problems"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|problem| problem == row);
    smoke_eq!(
        seen(named, why["problems"].clone(), row),
        row.clone(),
        "why_names_the_missing_partner"
    );
    if row["reason"] == UNWRITTEN {
        smoke_check!(
            names_a_pass(row) && why["state"]["off"]["because"] != "unmet",
            "only_the_reading_pass_turns_off",
            json!({ "path": row["path"], "state": why["state"] }),
            json!({
                "path": "$.bricks.<brick>.passes.<pass>.reads.<target>",
                "state": { "off": { "because": { "not": "unmet" } } },
            })
        );
    }
    Ok(())
}

fn suggesting(brick: &str) -> Value {
    Value::String(json!([brick]).to_string())
}

fn reads_depth(row: &Value) -> bool {
    row["path"]
        .as_str()
        .is_some_and(|path| path.ends_with(&format!(".reads.{}", TargetName::DEPTH.name())))
}

fn unmet_rows(status: &Value) -> Vec<Value> {
    status["unmet"].as_array().cloned().unwrap_or_default()
}

fn empty() -> Value {
    json!({"id": "partners", "revision": 1, "materials": [], "meshes": [], "entities": []})
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&empty().to_string());
    render(session)?;
    let status = query(session, "status", json!({}))?;
    let listed = |brick: &str| {
        status["features"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|name| name == brick)
    };
    smoke_eq!(
        json!({ "occlusion": listed(OCCLUSION), "prepass": listed(LEFT_OUT) }),
        json!({ "occlusion": true, "prepass": false }),
        "the_list_keeps_occlusion_and_leaves_out_the_prepass"
    );
    let unmet = unmet_rows(&status);
    for row in &unmet {
        if let Some(brick) = brick_of(row) {
            named_by_why(session, row, brick)?;
        }
    }
    let reported = unmet.iter().any(|row| {
        brick_of(row) == Some(OCCLUSION) && row["reason"] == UNWRITTEN && names_a_pass(row)
    });
    smoke_eq!(
        json!({ "reported": reported }),
        json!({ "reported": true }),
        "the_occlusion_pass_without_prepass_is_reported"
    );
    let suggested: Vec<Value> = unmet
        .iter()
        .filter(|row| brick_of(row) == Some(OCCLUSION) && reads_depth(row))
        .map(|row| row["nearest"].clone())
        .collect();
    let wanted = suggesting(LEFT_OUT);
    smoke_check!(
        !suggested.is_empty() && suggested.iter().all(|nearest| *nearest == wanted),
        "the_unwritten_depth_names_only_the_left_out_prepass",
        json!(suggested),
        json!([wanted])
    );
    occlusion_off(session)
}

pub(super) fn ground(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&empty().to_string());
    render(session)?;
    let status = query(session, "status", json!({}))?;
    let path = format!("$.bricks.{GROUNDING}.reads.{GROUND}");
    let row = unmet_rows(&status)
        .into_iter()
        .find(|row| row["path"] == path.as_str())
        .unwrap_or(Value::Null);
    smoke_eq!(
        json!({ "path": row["path"], "reason": row["reason"], "nearest": row["nearest"] }),
        json!({ "path": path, "reason": UNFILLED, "nearest": suggesting(GROUND_FILLER) }),
        "the_missing_ground_names_the_brick_to_add"
    );
    named_by_why(session, &row, GROUNDING)?;
    let why = query(session, "why", json!({ "brick": GROUNDING }))?;
    smoke_eq!(
        why["state"]["off"]["because"].clone(),
        json!("unmet"),
        "grounding_without_ground_is_held_off"
    );
    Ok(())
}

fn occlusion_off(session: &mut dyn SmokeSession) -> Checked<()> {
    session.source(&grid(2).to_string());
    control(session, "pause", json!({}))?;
    let shown = settled(session)?;
    let ran: Vec<&Value> = shown["render_passes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|pass| OCCLUSION_PASSES.iter().any(|name| *pass == *name))
        .collect();
    smoke_eq!(json!(ran), json!([]), "no_occlusion_pass_runs");
    let starved = image(session)?;
    environment(
        session,
        json!({ "ambient_occlusion": { "enabled": false } }),
    )?;
    let without = image(session)?;
    let wanted = json!({ "sha256": without });
    smoke_eq!(
        seen(starved == without, json!({ "sha256": starved }), &wanted),
        wanted,
        "the_boxes_look_as_without_occlusion"
    );
    Ok(())
}
