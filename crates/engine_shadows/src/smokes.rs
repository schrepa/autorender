use engine_smoke::settled;
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::json;

pub(crate) fn tiles_settle(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    smoke_eq!(session.frame(), true, "shadow_frame");
    settled(session)?;
    let answer = session.request("inspect_shadow_tile", json!({ "at": [0.0, 0.0, 0.0] }));
    smoke_eq!(answer.is_ok(), true, "inspect_answers");
    Ok(())
}
