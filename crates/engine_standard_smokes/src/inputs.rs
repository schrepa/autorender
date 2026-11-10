use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{query, render};
use serde_json::{Value, json};

pub(super) const LEFT_OUT: &str = "engine_lighting";

fn names_the_left_out(row: &Value) -> bool {
    row["nearest"]
        .as_str()
        .is_some_and(|nearest| nearest.contains(LEFT_OUT))
}

fn empty() -> Value {
    json!({"id": "inputs", "revision": 1, "materials": [], "meshes": [], "entities": []})
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&empty().to_string());
    render(session)?;
    let status = query(session, "status", json!({}))?;
    let notices: Vec<Value> = status["notices"].as_array().cloned().unwrap_or_default();
    let paths: Vec<&str> = notices
        .iter()
        .filter_map(|row| row["path"].as_str())
        .collect();
    let once = paths
        .iter()
        .enumerate()
        .all(|(at, path)| !paths[..at].contains(path));
    let noticed = |reason: &str| notices.iter().any(|row| row["reason"] == reason);
    smoke_eq!(
        json!({
            "unmet": status["unmet"],
            "unread": noticed("input_unread"),
            "unopened": noticed("slot_unopened"),
            "once": once,
            "named": notices.iter().all(names_the_left_out),
        }),
        json!({ "unmet": [], "unread": true, "unopened": true, "once": true, "named": true }),
        "unread_inputs_and_unopened_slots_are_noticed_once_naming_the_brick_that_reads_them"
    );
    Ok(())
}
