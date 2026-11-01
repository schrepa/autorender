use serde_json::{Value, json};

use engine_core::smoke::{Checked, control, environment, inspect};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};

const FRAME_MS: f64 = 1000.0 / 60.0;

fn frames(session: &mut dyn SmokeSession) -> Checked<Value> {
    control(
        session,
        "frames",
        json!({ "count": 60, "interval_ms": FRAME_MS }),
    )
}

pub(crate) fn dither(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    smoke_eq!(
        inspect(session, "Dither")?,
        json!({ "enabled": true }),
        "dither_on_by_default"
    );
    let reply = environment(session, json!({ "dither": { "enabled": false } }))?;
    smoke_eq!(
        reply["environment_changed"],
        json!(["dither"]),
        "dither_controlled"
    );
    smoke_eq!(
        inspect(session, "Dither")?,
        json!({ "enabled": false }),
        "dither_off_inspected"
    );
    frames(session)?;
    environment(session, json!({ "dither": { "enabled": true } }))?;
    frames(session)?;
    Ok(())
}
