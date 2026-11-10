use engine_core::{SmokeFailure, SmokeSession, smoke_check, smoke_eq};
use engine_smoke::{Checked, query, render};
use serde_json::{Value, json};

use super::requests::seen;

pub(super) const LEFT_OUT: &str = "engine_prepass";
const OCCLUSION: &str = "engine_ambient_occlusion";
const UNWRITTEN: &str = "read_before_write";

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
    let unmet: Vec<Value> = status["unmet"].as_array().cloned().unwrap_or_default();
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
    Ok(())
}
