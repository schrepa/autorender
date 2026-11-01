use std::any::{Any, TypeId};

use serde::Serialize;
use serde_json::{Value, json};

pub use engine_place::ask;
use engine_place::smoke_check;

use crate::{List, SmokeFailure, SmokeSession};

pub type Checked<T> = Result<T, SmokeFailure>;

pub const COMPILE_FRAMES: u32 = 20_000;

pub fn control(session: &mut dyn SmokeSession, name: &str, args: Value) -> Checked<Value> {
    ask(session, "control", json!({"name": name, "args": args}))
}

pub fn query(session: &mut dyn SmokeSession, name: &str, args: Value) -> Checked<Value> {
    ask(session, "query", json!({"name": name, "args": args}))
}

pub fn refuse(
    session: &mut dyn SmokeSession,
    verb: &str,
    name: &str,
    args: Value,
) -> Checked<Value> {
    let reply = session.request(verb, json!({"name": name, "args": args}));
    smoke_check!(
        reply.is_err(),
        "request_refused",
        &reply.as_ref().ok().map(|answered| &answered["result"]),
        &name
    );
    Ok(reply
        .err()
        .map(|mut refused| refused["error"].take())
        .unwrap_or_default())
}

pub fn environment(session: &mut dyn SmokeSession, sections: Value) -> Checked<Value> {
    control(session, "environment", sections)
}

pub fn camera(session: &mut dyn SmokeSession, view: Value) -> Checked<Value> {
    control(session, "camera", view)
}

pub fn inspect(session: &mut dyn SmokeSession, component: &str) -> Checked<Value> {
    let mut inspected = ask(session, "inspect", json!({"component": component}))?;
    Ok(inspected["value"].take())
}

pub fn component(session: &mut dyn SmokeSession, id: &str, component: &str) -> Checked<Value> {
    let mut inspected = ask(
        session,
        "inspect",
        json!({"entity": id, "component": component}),
    )?;
    Ok(inspected["entity"]["components"][component].take())
}

pub fn render(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut rendered = control(session, "render", json!({}))?;
    Ok(rendered["rendering"].take())
}

pub fn present(session: &mut dyn SmokeSession, alpha: f64) -> Checked<Value> {
    let mut presented = control(session, "present", json!({"alpha": alpha}))?;
    Ok(presented["rendering"].take())
}

pub fn renders_until(
    session: &mut dyn SmokeSession,
    frames: u32,
    mut done: impl FnMut(&Value) -> bool,
) -> Checked<(bool, Value)> {
    let mut rendering = Value::Null;
    for _ in 0..frames {
        rendering = render(session)?;
        if done(&rendering) {
            return Ok((true, rendering));
        }
    }
    Ok((false, rendering))
}

pub fn compiled(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut rendering = render(session)?;
    for _ in 0..COMPILE_FRAMES {
        if !compiling(&rendering) {
            return Ok(rendering);
        }
        rendering = render(session)?;
    }
    smoke_check!(
        !compiling(&rendering),
        "pipelines_compile",
        &rendering["pipelines_compiling"],
        &false
    );
    Ok(rendering)
}

pub fn compiling(rendering: &Value) -> bool {
    rendering["pipelines_compiling"] == true
}

pub fn shot(session: &mut dyn SmokeSession, body: Value) -> Checked<Value> {
    let captured = ask(session, "capture", body)?;
    let digest = captured["artifact"]["sha256"].clone();
    smoke_check!(
        digest.is_string(),
        "capture_keeps_its_frame",
        &captured["artifact"],
        &json!({"sha256": "string"})
    );
    Ok(digest)
}

pub fn capture(session: &mut dyn SmokeSession) -> Checked<Value> {
    compiled(session)?;
    shot(session, json!({}))
}

pub fn number(value: &Value) -> f64 {
    value.as_f64().unwrap_or(f64::NAN)
}

pub fn with(base: &Value, fields: Value) -> Value {
    let mut merged = base.clone();
    if let (Value::Object(merged), Value::Object(fields)) = (&mut merged, fields) {
        merged.extend(fields);
    }
    merged
}

pub fn above(value: impl Serialize) -> Value {
    json!({ "above": value })
}

pub fn below(value: impl Serialize) -> Value {
    json!({ "below": value })
}

pub fn at_most(value: impl Serialize) -> Value {
    json!({ "at_most": value })
}

pub fn near(value: impl Serialize, within: f64) -> Value {
    json!({"near": value, "within": within})
}

pub fn not(value: impl Serialize) -> Value {
    json!({ "not": value })
}

pub fn members<L: List>(session: &mut dyn SmokeSession, mut each: impl FnMut(&L::Member)) {
    session.joined(TypeId::of::<L>(), &mut |member: &dyn Any| {
        if let Some(member) = member.downcast_ref::<L::Member>() {
            each(member);
        }
    });
}
