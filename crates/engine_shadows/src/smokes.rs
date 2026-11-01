use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::json;

const SETTLE_FRAMES: u32 = 8;

pub(crate) fn tiles_settle(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    for _ in 0..SETTLE_FRAMES {
        smoke_eq!(session.frame(), true, "shadow_frame");
    }
    let answer = session.request("inspect_shadow_tile", json!({ "at": [0.0, 0.0, 0.0] }));
    smoke_eq!(answer.is_ok(), true, "inspect_answers");
    Ok(())
}
