#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine::prelude::*;

mod smokes;

#[derive(Resource, Default)]
pub struct Flat;

impl Game for Flat {
    const SMOKES: &'static [Smoke] = &smokes::SMOKES;
}

pub fn bricks<L: Listing>(list: L) -> L {
    engine_standard_rendering::two_d(list)
}

engine::game!(pub struct FlatGame(Flat), "Flat", bricks);
