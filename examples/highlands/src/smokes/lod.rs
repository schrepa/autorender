use engine::smoke::{Checked, SmokeResult, SmokeSession, ask, refuse};
use engine::{smoke_check, smoke_eq};
use serde_json::json;

use super::open;

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    open(session, false)?;
    described(session)?;
    refused(session)
}

fn described(session: &mut dyn SmokeSession) -> Checked<()> {
    let listing = ask(session, "describe", json!({}))?;
    let listed = listing["types"]
        .as_array()
        .is_some_and(|types| types.iter().any(|kind| kind["name"] == "LevelOfDetail"));
    smoke_check!(
        listed,
        "the_level_of_detail_settings_are_discoverable",
        &listed,
        &true
    );
    let look = ask(session, "inspect", json!({"component": "Environment"}))?;
    let detail = &look["value"]["level_of_detail"];
    smoke_check!(
        detail["enabled"] == true
            && detail["pixel_error"] == 1.0
            && detail["impostors"] == true
            && detail["fade_seconds"] == 0.5,
        "the_game_keeps_the_default_detail",
        detail,
        &json!({"enabled": true, "pixel_error": 1.0, "impostors": true, "fade_seconds": 0.5})
    );
    Ok(())
}

fn refused(session: &mut dyn SmokeSession) -> Checked<()> {
    let invalid = refuse(
        session,
        "control",
        "environment",
        json!({"level_of_detail": {"pixel_error": 0.1}}),
    )?;
    smoke_eq!(
        &invalid["code"],
        &json!("invalid_environment"),
        "a_tiny_pixel_error_is_refused"
    );
    smoke_eq!(
        &invalid["path"],
        &json!("$.args.level_of_detail.pixel_error"),
        "a_tiny_pixel_error_is_refused_at"
    );
    Ok(())
}
