use engine_core::smoke::{Checked, ask, control, query, refuse};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

const DESCRIBED: [&str; 6] = [
    "screen",
    "screens",
    "level",
    "level_progress",
    "time",
    "time_state",
];

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene("screens-empty", &json!([])).to_string());
    control(session, "pause", json!({}))?;
    for name in DESCRIBED {
        ask(session, "describe", json!({"name": name}))?;
    }
    undeclared(session)?;
    session.source(&scene("screens-fixture", &entities()).to_string());
    declared(session)?;
    timed(session)?;
    let progress = query(session, "level_progress", json!({}))?;
    smoke_eq!(progress["state"], json!("idle"), "level_starts_idle");
    Ok(())
}

fn paused(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut state = query(session, "time_state", json!({}))?;
    Ok(state["time_paused"].take())
}

fn tick(session: &mut dyn SmokeSession) -> Value {
    session
        .request("query", json!({"name": "status"}))
        .map(|mut answered| answered["observation"]["tick"].take())
        .unwrap_or_default()
}

fn undeclared(session: &mut dyn SmokeSession) -> Checked<()> {
    control(session, "screen", json!({"action": "push", "name": "any"}))?;
    session.step(1);
    smoke_eq!(
        paused(session)?,
        json!(false),
        "undeclared_screens_do_nothing"
    );
    control(session, "screen", json!({"action": "pop"}))?;
    session.step(1);
    Ok(())
}

fn declared(session: &mut dyn SmokeSession) -> Checked<()> {
    control(session, "screen", json!({"action": "go", "name": "play"}))?;
    session.step(1);
    let before = tick(session);
    session.step(4);
    smoke_eq!(tick(session) != before, true, "playing_screen_ticks");
    control(session, "screen", json!({"action": "push", "name": "menu"}))?;
    session.step(1);
    smoke_eq!(paused(session)?, json!(true), "pausing_screen_pauses");
    let held = tick(session);
    session.step(4);
    smoke_eq!(tick(session), held, "paused_holds_ticks");
    let listed = query(session, "screens", json!({}))?;
    smoke_eq!(listed["stack"], json!(["play", "menu"]), "stack_lists");
    smoke_eq!(listed["top"], json!("menu"), "top_lists");
    control(session, "screen", json!({"action": "pop"}))?;
    session.step(1);
    smoke_eq!(paused(session)?, json!(false), "popping_resumes");
    let unknown = json!({"action": "go", "name": "nowhere"});
    let refused = refuse(session, "control", "screen", unknown)?;
    smoke_eq!(
        refused["code"],
        json!("invalid_value"),
        "unknown_screen_refused"
    );
    control(session, "screen", json!({"action": "pop"}))?;
    session.step(1);
    Ok(())
}

fn timed(session: &mut dyn SmokeSession) -> Checked<()> {
    let fast = control(session, "time", json!({"scale": 100.0}))?;
    smoke_eq!(fast["time_scale"], json!(8.0), "scale_is_kept_in_range");
    control(session, "time", json!({"scale": 1.0, "paused": true}))?;
    let held = tick(session);
    session.step(3);
    smoke_eq!(tick(session), held, "time_pause_holds_ticks");
    control(session, "time", json!({"paused": false}))?;
    session.step(3);
    smoke_eq!(tick(session) != held, true, "time_resume_ticks");
    Ok(())
}

fn entities() -> Value {
    json!([
        {
            "id": "play",
            "screen": {"name": "play"},
            "transform": {"translation": [0.0, 0.0, 0.0]},
        },
        {
            "id": "menu",
            "screen": {"name": "menu", "pauses": true},
            "transform": {"translation": [0.0, 0.0, 0.0]},
        },
    ])
}

fn scene(id: &str, entities: &Value) -> Value {
    json!({
        "id": id,
        "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": entities,
    })
}
