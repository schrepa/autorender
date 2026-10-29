pub mod ambiguities;
pub mod app;
pub mod bench;
pub mod bricks;
pub mod comments;
pub mod coverage;
pub mod dependencies;
pub mod exports;
pub mod faces;
pub mod files;
pub mod formats;
pub mod generated;
pub mod gpu;
pub mod leaves;
pub mod policy;
pub mod stages;
pub mod structure;
pub mod tools;
mod workspace;

pub use workspace::{Dependency, DependencyKind, Package, Repository};

use engine_core::Error;

pub fn at(file: &str, line: usize) -> String {
    format!("{file}:{line}")
}

pub fn report(failures: &[Error]) -> Result<(), String> {
    if failures.is_empty() {
        return Ok(());
    }
    let lines: Vec<String> = failures
        .iter()
        .map(|failure| serde_json::to_string(failure).unwrap_or_default())
        .collect();
    Err(lines.join("\n"))
}

#[derive(Clone, Copy)]
pub struct Rules {
    pub exceptions: Exceptions,
    pub facade: Option<Facade>,
    pub forbidden: &'static [Forbidden],
    pub errors: Option<Errors>,
}

impl Rules {
    pub const NONE: Self = Self {
        exceptions: Exceptions::NONE,
        facade: None,
        forbidden: &[],
        errors: None,
    };
}

#[derive(Clone, Copy)]
pub struct Exceptions {
    pub long_files: &'static [(&'static str, usize)],
    pub wide_structs: &'static [(&'static str, &'static str, usize)],
    pub unsafe_files: &'static [&'static str],
    pub core_uses_brick: &'static [(&'static str, &'static str)],
}

impl Exceptions {
    pub const NONE: Self = Self {
        long_files: &[],
        wide_structs: &[],
        unsafe_files: &[],
        core_uses_brick: &[],
    };
}

#[derive(Clone, Copy)]
pub struct Facade {
    pub package: &'static str,
    pub inner: &'static str,
    pub internal: &'static [&'static str],
}

#[derive(Clone, Copy)]
pub struct Forbidden {
    pub within: &'static str,
    pub except: &'static str,
    pub code: &'static str,
    pub instead: &'static str,
}

#[derive(Clone, Copy)]
pub struct Errors {
    pub within: &'static str,
    pub file: &'static str,
}
