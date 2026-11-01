use serde_json::{Value, json};

use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};

const FRAME_MS: f64 = 1000.0 / 60.0;

type Asked = Result<Value, SmokeFailure>;

fn reply(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Value {
    match session.request(verb, body) {
        Ok(reply) | Err(reply) => reply,
    }
}

fn accepted(session: &mut dyn SmokeSession, verb: &str, body: Value, check: &'static str) -> Asked {
    let reply = reply(session, verb, body);
    engine_core::smoke_check!(
        reply["ok"] == true,
        check,
        reply["error"].clone(),
        json!({ "ok": true })
    );
    Ok(reply)
}

fn control(session: &mut dyn SmokeSession, name: &str, args: Value, check: &'static str) -> Asked {
    accepted(
        session,
        "control",
        json!({ "name": name, "args": args }),
        check,
    )
}

fn inspected(session: &mut dyn SmokeSession, component: &str, check: &'static str) -> Asked {
    let reply = accepted(session, "inspect", json!({ "component": component }), check)?;
    Ok(reply["result"]["value"].clone())
}

fn frames(session: &mut dyn SmokeSession, count: u32, check: &'static str) -> Asked {
    let reply = control(
        session,
        "frames",
        json!({ "count": count, "interval_ms": FRAME_MS }),
        check,
    )?;
    Ok(reply["result"]["rendering"].clone())
}

fn changed(reply: &Value) -> Value {
    reply["result"]["environment_changed"].clone()
}

fn dither(session: &mut dyn SmokeSession) -> SmokeResult {
    let inspect = "inspects_dither";
    engine_core::smoke_eq!(
        inspected(session, "Dither", inspect)?,
        json!({ "enabled": true }),
        "dither_on_by_default"
    );
    let off = json!({ "dither": { "enabled": false } });
    let reply = control(session, "environment", off, "turns_dither_off")?;
    engine_core::smoke_eq!(changed(&reply), json!(["dither"]), "dither_controlled");
    engine_core::smoke_eq!(
        inspected(session, "Dither", inspect)?,
        json!({ "enabled": false }),
        "dither_off_inspected"
    );
    frames(session, 60, "runs_undithered_frames")?;
    let on = json!({ "dither": { "enabled": true } });
    control(session, "environment", on, "turns_dither_on")?;
    frames(session, 60, "runs_dithered_frames")?;
    Ok(())
}
