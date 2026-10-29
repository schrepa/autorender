use engine_core::smoke::{SmokeFailure, SmokeSession};
use engine_core::smoke_check;
use engine_scene::{SceneDocument, SceneInstance};
use serde_json::{Value, json};

pub(super) struct SceneFixture {
    instance: Option<SceneInstance>,
}

impl SceneFixture {
    pub(super) fn load(
        session: &mut dyn SmokeSession,
        document: &Value,
    ) -> Result<(Self, Value), SmokeFailure> {
        let mut fixture = Self { instance: None };
        let loaded = fixture.reload(session, document);
        smoke_check!(
            loaded.is_ok(),
            "fixture_loads",
            json!(loaded.as_ref().err()),
            json!(null)
        );
        Ok((fixture, loaded.unwrap_or_default()))
    }

    pub(super) fn reload(
        &mut self,
        session: &mut dyn SmokeSession,
        document: &Value,
    ) -> Result<Value, Value> {
        let parsed = SceneDocument::from_json(&document.to_string())
            .map_err(|error| json!({"code": error.code, "path": error.path}))?;
        let world = session.app().world_mut();
        let instance = parsed
            .instantiate(world)
            .map_err(|error| json!({"code": error.code, "path": error.path}))?;
        if let Some(previous) = self.instance.replace(instance) {
            previous.despawn(world);
        }
        session
            .request("control", json!({"name": "render"}))
            .map(|reply| reply["result"].clone())
            .map_err(|reply| reply["error"].clone())
    }
}

pub(super) fn ask(
    session: &mut dyn SmokeSession,
    verb: &str,
    body: Value,
) -> Result<Value, SmokeFailure> {
    let sent = json!({"verb": verb, "body": body});
    let reply = session.request(verb, body);
    smoke_check!(
        reply.is_ok(),
        "request_accepted",
        json!(reply.as_ref().err()),
        sent
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
    smoke_check!(
        reply.is_err(),
        "request_refused",
        json!(reply.as_ref().ok()),
        sent
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
    smoke_check!(
        image.is_string(),
        "capture_keeps_image",
        json!(captured["artifact"]),
        json!("sha256")
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
