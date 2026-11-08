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
    fn source(&mut self, document: &str);
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

pub fn environment(session: &mut dyn SmokeSession, sections: Value) -> Checked<Value> {
    control(session, "environment", sections)
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

pub fn settled(session: &mut dyn SmokeSession) -> Checked<Value> {
    settled_when(session, render, |_| true)
}

pub fn settled_when(
    session: &mut dyn SmokeSession,
    mut frame: impl FnMut(&mut dyn SmokeSession) -> Checked<Value>,
    mut done: impl FnMut(&Value) -> bool,
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
            && done(&rendering)
        {
            return Ok(rendering);
        }
    }
}

pub fn number(value: &Value) -> f64 {
    value.as_f64().unwrap_or(f64::NAN)
}

pub fn above(value: impl Serialize) -> Value {
    json!({ "above": value })
}

pub fn below(value: impl Serialize) -> Value {
    json!({ "below": value })
}

pub fn near(value: impl Serialize, within: f64) -> Value {
    json!({"near": value, "within": within})
}
