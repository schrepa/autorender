use engine_core::Listing;
use engine_fly_camera::FlyCameraBrick;
use engine_lighting::LightingModelBrick;
use highlands_game::HighlandsBrick;
use highlands_pond::PondBrick;
use highlands_toon_lighting::ToonLightingBrick;

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard::standard(list)
        .with::<FlyCameraBrick>()
        .with::<PondBrick>()
        .with::<HighlandsBrick>()
}

pub fn toon<L: Listing>(list: L) -> L {
    bricks(list)
        .without::<LightingModelBrick>()
        .with::<ToonLightingBrick>()
}
