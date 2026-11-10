#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine::prelude::*;
use engine_collision::CollisionBrick;
use engine_grounding::GroundingBrick;
use engine_navigation::NavigationBrick;

mod smokes;

#[derive(Resource, Default)]
pub struct Walker;

impl Game for Walker {
    const SMOKES: &'static [Smoke] = &smokes::SMOKES;
}

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard_rendering::minimal(list)
        .with::<CollisionBrick>()
        .with::<NavigationBrick>()
        .with::<GroundingBrick>()
        .places(engine_places_simulation::PLACES)
        .places(engine_solids::PLACES)
}

engine::game!(pub struct WalkerGame(Walker), "Walker", bricks);
