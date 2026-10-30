use bevy_app::App;
use serde::Serialize;
use serde_json::{Value, json};

use crate::bricks::BrickList;

pub type SmokeResult = Result<(), SmokeFailure>;

#[derive(Debug, Clone, serde::Serialize)]
pub struct SmokeFailure {
    pub smoke: &'static str,
    pub check: &'static str,
    pub got: Value,
    pub expected: Value,
    pub at: &'static str,
}

impl SmokeFailure {
    pub fn new(
        check: &'static str,
        got: &impl serde::Serialize,
        expected: &impl serde::Serialize,
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

    #[must_use]
    pub fn of(mut self, smoke: &'static str) -> Self {
        self.smoke = smoke;
        self
    }
}

pub trait SmokeSession {
    fn app(&mut self) -> &mut App;
    fn request(&mut self, verb: &str, body: Value) -> Result<Value, Value>;
    fn step(&mut self, ticks: u32);
    fn frame(&mut self) -> bool;
    fn gpu(&self) -> bool;
    fn bricks(&self) -> BrickList;
    fn restart(&mut self, bricks: BrickList);
    fn source(&mut self, document: &str);
}

#[derive(Clone, Copy, Debug)]
pub struct SmokeEntry {
    pub name: &'static str,
    pub brick: &'static str,
    pub gpu: bool,
    pub run: fn(&mut dyn SmokeSession) -> SmokeResult,
}

impl SmokeEntry {
    pub fn run_in(&self, session: &mut dyn SmokeSession) -> SmokeResult {
        (self.run)(session).map_err(|failure| failure.of(self.name))
    }
}

#[macro_export]
macro_rules! smoke_check {
    ($condition:expr, $check:expr, $got:expr, $expected:expr $(,)?) => {
        if !$condition {
            return ::core::result::Result::Err($crate::smoke::SmokeFailure::new(
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
            return ::core::result::Result::Err($crate::smoke::SmokeFailure::new(
                $check,
                got,
                expected,
                ::core::concat!(::core::file!(), ":", ::core::line!()),
            ));
        }
    }};
}

pub type Checked<T> = Result<T, SmokeFailure>;

pub const COMPILE_FRAMES: u32 = 20_000;

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
