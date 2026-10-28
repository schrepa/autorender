#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
mod player;

include!(concat!(env!("OUT_DIR"), "/source.rs"));

pub use player::{PlayerWalk, ScenePlayer, features};
