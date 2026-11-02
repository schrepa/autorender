use engine_core::smoke::{Checked, ask, inspect, refuse};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::json;

use super::open;

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    open(session)?;
    described(session)?;
    refused(session)
}

fn described(session: &mut dyn SmokeSession) -> Checked<()> {
    let listing = ask(session, "describe", json!({}))?;
    let listed = listing["types"]
        .as_array()
        .is_some_and(|types| types.iter().any(|kind| kind["name"] == "detail"));
    smoke_eq!(
        listed,
        true,
        "the_level_of_detail_settings_are_discoverable"
    );
    let detail = inspect(session, "detail")?;
    let impostors = inspect(session, "impostors")?;
    smoke_eq!(
        json!({
            "enabled": detail["enabled"],
            "pixel_error": detail["pixel_error"],
            "impostors": impostors["enabled"],
            "fade_seconds": detail["fade_seconds"],
        }),
        json!({"enabled": true, "pixel_error": 1.0, "impostors": true, "fade_seconds": 0.5}),
        "the_game_keeps_the_default_detail"
    );
    Ok(())
}

fn refused(session: &mut dyn SmokeSession) -> Checked<()> {
    let invalid = refuse(
        session,
        "control",
        "environment",
        json!({"detail": {"pixel_error": 0.1}}),
    )?;
    smoke_eq!(
        &invalid["code"],
        &json!("invalid_environment"),
        "a_tiny_pixel_error_is_refused"
    );
    smoke_eq!(
        &invalid["path"],
        &json!("$.args.detail.pixel_error"),
        "a_tiny_pixel_error_is_refused_at"
    );
    Ok(())
}
