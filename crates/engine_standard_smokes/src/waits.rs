use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, control, number, query, render};
use serde_json::{Value, json};

use super::ground::fixture;
use super::requests::seen;

fn cells<'a>(rendering: &'a Value, state: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
    rendering["kind_passes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(move |cell| cell["state"]["state"] == state)
}

fn named_by_why(session: &mut dyn SmokeSession, cell: &Value) -> Checked<()> {
    let brick = cell["brick"].as_str().unwrap_or_default();
    let why = query(session, "why", json!({ "brick": brick }))?;
    let listed = why["drawing"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|row| row == cell);
    let unsettled = why["waiting"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|row| number(&row["left"]) > 0.0);
    let wanted = json!({ "drawing": [cell], "waiting": "a row with work left" });
    smoke_eq!(
        seen(
            listed && unsettled,
            json!({ "drawing": why["drawing"], "waiting": why["waiting"] }),
            &wanted,
        ),
        wanted,
        "why_names_what_a_kind_waits_for"
    );
    Ok(())
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&fixture().to_string());
    control(session, "pause", json!({}))?;
    loop {
        let rendering = render(session)?;
        let readiness = &rendering["readiness"];
        smoke_eq!(
            readiness["failed"],
            json!([]),
            "nothing_fails_while_settling"
        );
        let hidden: Vec<&Value> = cells(&rendering, "skipped")
            .filter(|cell| {
                cell["state"]["reason"]
                    .as_str()
                    .is_some_and(|reason| reason.contains("resident"))
            })
            .collect();
        smoke_eq!(json!(hidden), json!([]), "no_skip_hides_a_wait");
        let waiting: Vec<Value> = cells(&rendering, "waiting").cloned().collect();
        for cell in &waiting {
            named_by_why(session, cell)?;
        }
        let settled = readiness["loading"]["total"] == 0
            && readiness["compiling"] == 0
            && readiness["settling"] == 0;
        if settled {
            smoke_eq!(json!(waiting), json!([]), "no_kind_waits_once_settled");
            return Ok(());
        }
    }
}
