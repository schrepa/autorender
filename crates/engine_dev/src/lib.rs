#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod digest;
mod entities;
mod request;
mod table;

#[cfg(not(target_arch = "wasm32"))]
pub mod authoring;
pub mod schemas;

pub use request::{Applied, Refused, Reply, Request, Verb, apply, reply};
pub use table::{Instruments, install};
