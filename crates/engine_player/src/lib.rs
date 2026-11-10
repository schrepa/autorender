#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod player;
#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
mod pure;

pub mod bricks {
    pub mod corpora;
    pub mod walk;
}

use engine_core::Listing;
use engine_fly_camera::FlyCameraBrick;
use engine_standard_smokes::StandardSmokesBrick;

pub use player::ScenePlayer;

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard::everything(list)
        .with::<FlyCameraBrick>()
        .with::<bricks::walk::PlayerWalkBrick>()
        .with::<StandardSmokesBrick>()
        .with::<bricks::corpora::PlayerCorporaBrick>()
}

#[cfg(feature = "dev")]
mod smokes;

engine::game!(
    pub struct PlayerGame(ScenePlayer),
    "Engine Player",
    bricks,
    build = ScenePlayer::build
);
