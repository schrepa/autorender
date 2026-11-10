use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{control, render};
use serde_json::{Value, json};

use super::requests::seen;

const SETTLING_FRAMES: u32 = 60;

fn empty() -> Value {
    json!({"id": "empty", "revision": 1, "materials": [], "meshes": [], "entities": []})
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&empty().to_string());
    control(session, "pause", json!({}))?;
    let mut counted = 0;
    let readiness = loop {
        let mut rendering = render(session)?;
        let readiness = rendering["readiness"].take();
        smoke_eq!(
            readiness["failed"],
            json!([]),
            "nothing_fails_while_settling"
        );
        if readiness["compiling"] != 0 || readiness["loading"]["total"] != 0 {
            continue;
        }
        counted += 1;
        if readiness["settling"] == 0 || counted >= SETTLING_FRAMES {
            break readiness;
        }
    };
    let rows = readiness["unsettled"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let working: Vec<&Value> = rows.iter().filter(|row| row["awaits"] != "view").collect();
    let named = rows
        .iter()
        .all(|row| row["brick"].as_str().is_some_and(|brick| !brick.is_empty()));
    let wanted = json!({"settling": 0, "working": [], "named": true});
    smoke_eq!(
        seen(
            readiness["settling"] == 0 && working.is_empty() && named,
            json!({"settling": readiness["settling"], "unsettled": rows, "named": named}),
            &wanted,
        ),
        wanted,
        "empty_scene_settles"
    );
    Ok(())
}
