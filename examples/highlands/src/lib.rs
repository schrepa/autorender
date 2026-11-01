#[cfg(target_arch = "wasm32")]
mod browser;
mod game;
mod land;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
mod options;
mod records;
mod scene;

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

pub use game::Highlands;
pub use land::RockMaps;
pub use options::{Options, controls, parameters};

engine::__if_dev! {
    mod smokes;

    pub use smokes::smokes;
}
