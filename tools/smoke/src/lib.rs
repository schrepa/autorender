use std::any::{Any, TypeId};

use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize)]
pub struct SmokeFailure {
    pub smoke: &'static str,
    pub check: &'static str,
    pub got: Value,
    pub expected: Value,
    pub at: &'static str,
}

impl SmokeFailure {
    #[doc(hidden)]
    pub fn new(
        check: &'static str,
        got: &impl Serialize,
        expected: &impl Serialize,
        at: &'static str,
    ) -> Self {
        Self {
            smoke: "",
            check,
            got: serde_json::to_value(got).unwrap_or(Value::Null),
            expected: serde_json::to_value(expected).unwrap_or(Value::Null),
            at,
        }
    }
}

pub trait SmokeSession {
    fn request(&mut self, verb: &str, body: Value) -> Result<Value, Value>;
    fn step(&mut self, ticks: u32);
    fn frame(&mut self) -> bool;
    fn source(&mut self, document: &str);
    fn file(&mut self, name: &str, bytes: &[u8]);
    fn idle(&mut self, milliseconds: f64);
    fn joined(&mut self, list: TypeId, each: &mut dyn FnMut(&dyn Any));
    fn members(&mut self, _place: &str) -> Vec<&'static str> {
        Vec::new()
    }
    fn slot(&mut self, _slot: TypeId, _each: &mut dyn FnMut(&dyn Any)) {}
    fn without(&mut self, _bricks: &[&'static str]) {}
    fn artifact(&mut self, _path: &str) -> Option<Vec<u8>> {
        None
    }
    fn decode_image(&mut self, _path: &str, _bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), Value> {
        Err(Value::Null)
    }
}

#[macro_export]
macro_rules! smoke_check {
    ($condition:expr, $check:expr, $got:expr, $expected:expr $(,)?) => {
        if !$condition {
            return ::core::result::Result::Err($crate::SmokeFailure::new(
                $check,
                &$got,
                &$expected,
                ::core::concat!(::core::file!(), ":", ::core::line!()),
            ));
        }
    };
}

#[macro_export]
macro_rules! smoke_eq {
    ($got:expr, $expected:expr, $check:expr $(,)?) => {{
        let (got, expected) = (&$got, &$expected);
        if *got != *expected {
            return ::core::result::Result::Err($crate::SmokeFailure::new(
                $check,
                got,
                expected,
                ::core::concat!(::core::file!(), ":", ::core::line!()),
            ));
        }
    }};
}

pub type Checked<T> = Result<T, SmokeFailure>;

pub fn ask(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Checked<Value> {
    let reply = session.request(verb, body);
    smoke_check!(
        reply.is_ok(),
        "request_answered",
        &reply.as_ref().err().map(|refused| &refused["error"]),
        &verb
    );
    Ok(reply
        .map(|mut answered| answered["result"].take())
        .unwrap_or_default())
}

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

pub fn rendered_when(
    session: &mut dyn SmokeSession,
    mut frame: impl FnMut(&mut dyn SmokeSession) -> Checked<Value>,
    mut done: impl FnMut(&Value) -> bool,
) -> Checked<Value> {
    loop {
        let rendering = frame(&mut *session)?;
        if done(&rendering) {
            return Ok(rendering);
        }
    }
}

pub fn compiled(session: &mut dyn SmokeSession) -> Checked<Value> {
    rendered_when(session, render, |rendering| !compiling(rendering))
}

pub fn settled(session: &mut dyn SmokeSession) -> Checked<Value> {
    settled_by(session, render)
}

pub fn settled_by(
    session: &mut dyn SmokeSession,
    mut frame: impl FnMut(&mut dyn SmokeSession) -> Checked<Value>,
) -> Checked<Value> {
    loop {
        let rendering = frame(&mut *session)?;
        let readiness = &rendering["readiness"];
        smoke_eq!(
            readiness["failed"],
            json!([]),
            "nothing_fails_while_settling"
        );
        if readiness["loading"]["total"] == 0
            && readiness["compiling"] == 0
            && readiness["settling"] == 0
        {
            return Ok(rendering);
        }
    }
}

pub fn compiling(rendering: &Value) -> bool {
    rendering["readiness"]["compiling"]
        .as_u64()
        .is_some_and(|compiling| compiling > 0)
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
