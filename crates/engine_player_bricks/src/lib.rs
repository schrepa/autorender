use engine_core::Listing;
use engine_fly_camera::FlyCameraBrick;
use engine_player_walk::PlayerWalkBrick;
use engine_standard_smokes::StandardSmokesBrick;

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard::standard(list)
        .with::<FlyCameraBrick>()
        .with::<PlayerWalkBrick>()
        .with::<StandardSmokesBrick>()
}
