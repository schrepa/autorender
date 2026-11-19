mod cook;
mod plan;
mod step;
mod store;

use std::fmt::Display;
use std::path::Path;
use std::process::ExitCode;

pub use plan::{Declared, Plan};
pub use step::Step;

pub struct Kind {
    pub name: &'static str,
    pub version: u32,
    pub run: fn(&mut Step<'_>) -> Result<(), Refusal>,
}

pub struct Refusal {
    pub code: &'static str,
    pub file: String,
    pub detail: String,
}

impl Refusal {
    pub fn new(code: &'static str, file: &str, detail: impl Display) -> Self {
        let (file, detail) = (file.to_owned(), detail.to_string());
        Self { code, file, detail }
    }
}

pub struct Input<'a> {
    pub name: &'a str,
    pub bytes: &'a [u8],
    pub path: &'a Path,
}

#[must_use]
pub fn main(plan: Plan) -> ExitCode {
    cook::main(plan)
}
