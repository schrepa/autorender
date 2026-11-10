use serde::Serialize;
use serde_json::{Map, Value, json};

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

pub const KIND_PASSES: &str = "KindPasses";

pub fn joined_passes(session: &mut dyn SmokeSession, kind: &str) -> Checked<Vec<String>> {
    let table = ask(session, "describe", json!({ "name": KIND_PASSES }))?;
    Ok(table["cells"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|cell| cell["kind"] == kind && cell["join"] == "joins")
        .filter_map(|cell| cell["pass"].as_str().map(str::to_owned))
        .collect())
}

pub fn counted_in_joined_passes(
    session: &mut dyn SmokeSession,
    kind: &str,
    frames: u32,
) -> Checked<Map<String, Value>> {
    let joined = joined_passes(session, kind)?;
    smoke_check!(
        !joined.is_empty(),
        "kind_joins_a_pass",
        json!({ "kind": kind, "joins": [] }),
        json!({ "kind": kind, "joins": above(0) })
    );
    let mut ran: Vec<String> = Vec::new();
    let mut counted = Map::new();
    for _ in 0..frames {
        let rendering = render(&mut *session)?;
        smoke_eq!(
            rendering["kind_problems"],
            json!([]),
            "no_kind_pass_is_silent"
        );
        let dropped: Vec<&Value> = rendering["dropped_draws"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|row| row["kind"] == kind && row["faults"] == true)
            .collect();
        smoke_eq!(json!(dropped), json!([]), "no_draw_of_the_kind_is_dropped");
        for cell in rendering["kind_passes"].as_array().into_iter().flatten() {
            let Some(pass) = cell["pass"].as_str() else {
                continue;
            };
            if !ran.iter().any(|known| known == pass) {
                ran.push(pass.to_owned());
            }
            if cell["kind"] != kind {
                continue;
            }
            let replaces = counted
                .get(pass)
                .is_none_or(|held| drew(cell) && !drew(held));
            if replaces {
                counted.insert(pass.to_owned(), cell.clone());
            }
        }
    }
    let undrawn: Vec<Value> = joined
        .iter()
        .filter(|pass| ran.contains(pass))
        .filter_map(|pass| {
            let cell = counted.get(pass.as_str());
            (!cell.is_some_and(drew)).then(|| json!({ "pass": pass, "cell": cell }))
        })
        .collect();
    smoke_eq!(
        json!(undrawn),
        json!([]),
        "kind_drew_in_every_joined_pass"
    );
    Ok(counted)
}

fn drew(cell: &Value) -> bool {
    cell["state"]["state"] == "drew"
}
