#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod digest;
mod entities;
pub mod record;
mod request;
mod table;

#[cfg(not(target_arch = "wasm32"))]
pub mod authoring;
pub mod schemas;

pub use request::{Applied, Refused, Reply, Request, Verb, apply, reply};

pub fn install(world: &mut bevy_ecs::world::World) {
    table::install(world);
    record::install(world);
}
