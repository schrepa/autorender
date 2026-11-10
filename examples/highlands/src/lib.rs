mod game;
mod jobs;
mod land;
mod records;
mod scene;

pub mod bricks {
    pub mod game;
    pub mod pond;
}

use engine_core::Listing;
use engine_fly_camera::FlyCameraBrick;

pub use game::Highlands;

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard::everything(list)
        .with::<FlyCameraBrick>()
        .with::<bricks::pond::PondBrick>()
        .with::<bricks::game::HighlandsBrick>()
}

#[cfg(feature = "dev")]
mod smokes;

engine::game!(
    pub struct HighlandsGame(Highlands),
    "Highlands",
    bricks,
    build = Highlands::build,
    scene = "highlands.scene.json"
);
