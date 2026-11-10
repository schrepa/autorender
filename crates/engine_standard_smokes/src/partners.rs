use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, query, render};
use serde_json::{Value, json};

use super::requests::seen;

const OCCLUSION: &str = "engine_ambient_occlusion";
const PREPASS: &str = "engine_prepass";
const UNWRITTEN: &str = "read_before_write";

fn brick_of(row: &Value) -> Option<&str> {
    row["path"]
        .as_str()?
        .strip_prefix("$.bricks.")?
        .split('.')
        .next()
        .filter(|brick| !brick.is_empty())
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
        let wanted = json!({ "state": "off", "because": "unmet" });
        let got = json!({
            "state": why["state"]["state"],
            "because": why["state"]["off"]["because"],
        });
        smoke_eq!(got, wanted, "a_brick_reading_an_unwritten_target_is_off");
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
    let unmet: Vec<Value> = status["unmet"].as_array().cloned().unwrap_or_default();
    for row in &unmet {
        if let Some(brick) = brick_of(row) {
            named_by_why(session, row, brick)?;
        }
    }
    let listed = |brick: &str| {
        status["features"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|name| name == brick)
    };
    let wanted = json!({ "reported": listed(OCCLUSION) && !listed(PREPASS) });
    let reported = unmet
        .iter()
        .any(|row| brick_of(row) == Some(OCCLUSION) && row["reason"] == UNWRITTEN);
    smoke_eq!(
        json!({ "reported": reported }),
        wanted,
        "ambient_occlusion_without_prepass_is_reported"
    );
    Ok(())
}
