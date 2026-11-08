#[cfg(target_arch = "wasm32")]
mod browser;
mod game;
mod land;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
mod options;
mod records;
mod scene;

pub mod bricks {
    pub mod game;
    pub mod pond;
}

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

use engine_core::Listing;
use engine_fly_camera::FlyCameraBrick;

pub use game::Highlands;
pub use land::RockMaps;
pub use options::{Options, controls, parameters};

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard::everything(list)
        .with::<FlyCameraBrick>()
        .with::<bricks::pond::PondBrick>()
        .with::<bricks::game::HighlandsBrick>()
}

engine_worker::export_worker!(bricks);

engine::__if_dev! {
    mod smokes;

    pub use smokes::smokes;
}
