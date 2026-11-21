#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine::Start;
use engine::prelude::*;

pub mod controls;
pub mod font;
pub mod state;
pub mod ui_layout;

pub mod bricks {
    pub mod fluid;
}

use crate::bricks::fluid::FluidBrick;

#[derive(Resource, Default)]
pub struct Quicksilver;

impl Game for Quicksilver {
    fn start() -> Start {
        let start = Start::default();
        Start {
            catch_up_seconds: 2.0 / f64::from(start.ticks_per_second),
            ..start
        }
    }
}

pub fn bricks<L: Listing>(list: L) -> L {
    list.with::<FluidBrick>()
}

engine::game!(
    pub struct QuicksilverGame(Quicksilver),
    "Quicksilver",
    bricks,
    scene = "quicksilver.scene.json"
);
