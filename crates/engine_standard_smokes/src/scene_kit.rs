use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::requests::seen;

pub(super) fn load(
    session: &mut dyn SmokeSession,
    document: &Value,
) -> Result<Value, SmokeFailure> {
    session.source(&document.to_string());
    let loaded = session
        .request("control", json!({"name": "render"}))
        .map(|reply| reply["result"].clone())
        .map_err(|reply| reply["error"].clone());
    let wanted = json!(null);
    smoke_eq!(
        seen(loaded.is_ok(), json!(loaded.as_ref().err()), &wanted),
        wanted,
        "fixture_loads"
    );
    Ok(loaded.unwrap_or_default())
}

pub(super) fn reload(session: &mut dyn SmokeSession, document: &Value) -> Result<Value, Value> {
    session.source(&document.to_string());
    let revision = document["revision"].as_u64().unwrap_or_default().to_string();
    let args = json!({"expected_revision": revision, "revision": revision});
    session
        .request("control", json!({"name": "reload", "args": args}))
        .map_err(|reply| reply["error"].clone())?;
    session
        .request("control", json!({"name": "render"}))
        .map(|reply| reply["result"].clone())
        .map_err(|reply| reply["error"].clone())
}

pub(super) fn ask(
    session: &mut dyn SmokeSession,
    verb: &str,
    body: Value,
) -> Result<Value, SmokeFailure> {
    let sent = json!({"verb": verb, "body": body});
    let reply = session.request(verb, body);
    smoke_eq!(
        seen(reply.is_ok(), json!(reply.as_ref().err()), &sent),
        sent,
        "request_accepted"
    );
    Ok(reply
        .map(|reply| reply["result"].clone())
        .unwrap_or_default())
}

pub(super) fn refuse(
    session: &mut dyn SmokeSession,
    verb: &str,
    body: Value,
) -> Result<Value, SmokeFailure> {
    let sent = json!({"verb": verb, "body": body});
    let reply = session.request(verb, body);
    smoke_eq!(
        seen(reply.is_err(), json!(reply.as_ref().ok()), &sent),
        sent,
        "request_refused"
    );
    Ok(reply
        .err()
        .map(|reply| reply["error"].clone())
        .unwrap_or_default())
}

pub(super) fn control(
    session: &mut dyn SmokeSession,
    name: &str,
    args: Value,
) -> Result<Value, SmokeFailure> {
    ask(session, "control", json!({"name": name, "args": args}))
}

pub(super) fn query(
    session: &mut dyn SmokeSession,
    name: &str,
    args: Value,
) -> Result<Value, SmokeFailure> {
    ask(session, "query", json!({"name": name, "args": args}))
}

pub(super) fn component(
    session: &mut dyn SmokeSession,
    id: &str,
    component: &str,
) -> Result<Value, SmokeFailure> {
    let inspected = ask(
        session,
        "inspect",
        json!({"entity": id, "component": component}),
    )?;
    Ok(inspected["entity"]["components"][component].clone())
}

pub(super) fn render(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    Ok(control(session, "render", json!({}))?["rendering"].clone())
}

pub(super) fn capture(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let captured = ask(session, "capture", json!({}))?;
    let image = captured["artifact"]["sha256"].clone();
    let wanted = json!("sha256");
    smoke_eq!(
        seen(image.is_string(), json!(captured["artifact"]), &wanted),
        wanted,
        "capture_keeps_image"
    );
    Ok(image)
}

pub(super) fn residency(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let assets = query(session, "assets", json!({}))?;
    let rendering = &assets["rendering"];
    Ok(json!({
        "meshes": assets["meshes"], "materials": assets["materials"], "textures": assets["textures"],
        "cpu": assets["resident_cpu_bytes"], "texture_cpu": assets["texture_cpu_bytes"],
        "gpu_meshes": rendering["resident_meshes"], "mesh_gpu": rendering["resident_mesh_bytes"],
        "gpu_textures": rendering["resident_textures"], "texture_gpu": rendering["resident_texture_bytes"],
    }))
}

pub(super) fn near(value: &Value, expected: f64, epsilon: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|value| value.is_finite() && (value - expected).abs() <= epsilon)
}

pub(super) fn near_all(values: &Value, expected: &[f64], epsilon: f64) -> bool {
    values.as_array().is_some_and(|values| {
        values.len() == expected.len()
            && values
                .iter()
                .zip(expected)
                .all(|(value, &expected)| near(value, expected, epsilon))
    })
}

pub(super) fn numbers(values: &Value) -> Vec<f64> {
    values
        .as_array()
        .into_iter()
        .flatten()
        .map(|value| value.as_f64().unwrap_or(f64::NAN))
        .collect()
}
