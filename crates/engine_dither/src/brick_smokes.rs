use serde_json::{Value, json};

use engine_core::{SmokeFailure, SmokeSession, smoke_eq};

const FRAME_MS: f64 = 1000.0 / 60.0;

type Asked = Result<Value, SmokeFailure>;

fn accepted(session: &mut dyn SmokeSession, verb: &str, body: Value, check: &'static str) -> Asked {
    let reply = session.request(verb, body);
    smoke_eq!(
        reply.as_ref().err().map(|refused| &refused["error"]),
        None,
        check
    );
    Ok(reply.unwrap_or_default())
}

fn control(session: &mut dyn SmokeSession, name: &str, args: Value, check: &'static str) -> Asked {
    accepted(
        session,
        "control",
        json!({ "name": name, "args": args }),
        check,
    )
}

fn inspected(session: &mut dyn SmokeSession, key: &str, check: &'static str) -> Asked {
    let reply = accepted(session, "inspect", json!({ "component": key }), check)?;
    Ok(reply["result"]["value"].clone())
}

fn frames(session: &mut dyn SmokeSession, count: u32, check: &'static str) -> Asked {
    control(
        session,
        "frames",
        json!({ "count": count, "interval_ms": FRAME_MS }),
        check,
    )
}

fn changed(reply: &Value) -> Value {
    reply["result"]["environment_changed"].clone()
}

pub(crate) fn dither(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let inspect = "inspects_dither";
    smoke_eq!(
        inspected(session, "dither", inspect)?,
        json!({ "enabled": true }),
        "dither_on_by_default"
    );
    let off = json!({ "dither": { "enabled": false } });
    let reply = control(session, "environment", off, "turns_dither_off")?;
    smoke_eq!(changed(&reply), json!(["dither"]), "dither_controlled");
    smoke_eq!(
        inspected(session, "dither", inspect)?,
        json!({ "enabled": false }),
        "dither_off_inspected"
    );
    frames(session, 60, "runs_undithered_frames")?;
    let on = json!({ "dither": { "enabled": true } });
    control(session, "environment", on, "turns_dither_on")?;
    frames(session, 60, "runs_dithered_frames")?;
    Ok(())
}
