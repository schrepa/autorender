#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

extern crate engine_ecs as bevy_ecs;

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
mod player;

pub mod bricks {
    pub mod walk;
}

include!(concat!(env!("OUT_DIR"), "/sources.rs"));

use engine_core::Listing;
use engine_fly_camera::FlyCameraBrick;
use engine_standard_smokes::StandardSmokesBrick;

pub use player::{ScenePlayer, listed};

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard::everything(list)
        .with::<FlyCameraBrick>()
        .with::<bricks::walk::PlayerWalkBrick>()
        .with::<StandardSmokesBrick>()
}

engine_worker::export_worker!(bricks);

engine::__if_dev! {
    mod smokes;

    pub use smokes::smokes;
}
