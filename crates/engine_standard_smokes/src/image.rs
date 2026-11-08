use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, ask, control, environment, settled};
use serde_json::{Value, json};

use super::requests::{grid, seen};

const SIDE: usize = 24;

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&grid(SIDE).to_string());
    control(session, "pause", json!({}))?;
    let before = image(session)?;
    environment(session, json!({"debug_view": {"view": "normals"}}))?;
    settled(session)?;
    environment(session, json!({"debug_view": {"view": "off"}}))?;
    let after = image(session)?;
    let wanted = json!({"sha256": before});
    smoke_eq!(
        seen(after == before, json!({"sha256": after}), &wanted),
        wanted,
        "off_restores_the_image"
    );
    Ok(())
}

fn image(session: &mut dyn SmokeSession) -> Checked<Value> {
    settled(session)?;
    let first = ask(session, "capture", json!({"alpha": 0}))?["artifact"]["sha256"].take();
    let again = ask(session, "capture", json!({"alpha": 0}))?["artifact"]["sha256"].take();
    let wanted = json!({"sha256": first});
    smoke_eq!(
        seen(
            first.is_string() && again == first,
            json!({"sha256": [first, again]}),
            &wanted,
        ),
        wanted,
        "captures_settle"
    );
    Ok(again)
}
