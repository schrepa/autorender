#[cfg(target_arch = "wasm32")]
mod browser;
mod field;
mod game;
mod land;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
mod options;
mod random;
mod records;
mod scene;
mod views;

include!(concat!(env!("OUT_DIR"), "/source.rs"));

pub use game::Highlands;
pub use land::RockMaps;
pub use options::{Options, controls, parameters};
