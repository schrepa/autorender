#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine::prelude::*;

pub mod bricks {
    pub mod plates;
}

mod smokes;

use crate::bricks::plates::PlatesBrick;

#[derive(Resource, Default)]
pub struct Bare;

impl Game for Bare {
    const SMOKES: &'static [Smoke] = &smokes::SMOKES;
}

pub fn bricks<L: Listing>(list: L) -> L {
    list.with::<PlatesBrick>()
}

engine::game!(pub struct BareGame(Bare), "Bare", bricks);
