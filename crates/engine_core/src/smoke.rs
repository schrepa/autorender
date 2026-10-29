use bevy_app::App;
use serde_json::Value;

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
