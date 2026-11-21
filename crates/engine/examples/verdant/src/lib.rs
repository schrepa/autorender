#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use brick_present::PresentBrick;
use brick_touch::TouchBrick;
use engine::Start;
use engine::prelude::*;

pub mod mesh;
pub mod names;
pub mod state;
pub mod terrain;

pub mod bricks {
    pub mod control;
    pub mod dock;
    pub mod frame;
    pub mod world;
}

use crate::bricks::control::ControlBrick;
use crate::bricks::dock::DockBrick;
use crate::bricks::frame::FrameBrick;
use crate::bricks::world::WorldBrick;
use crate::state::VerdantState;

#[derive(Resource, Default)]
pub struct Verdant;

impl Game for Verdant {
    fn start() -> Start {
        let start = Start::default();
        Start {
            catch_up_seconds: 2.0 / f64::from(start.ticks_per_second),
            ..start
        }
    }
}

pub fn bricks<L: Listing>(list: L) -> L {
    list.with::<TouchBrick>()
        .with::<ControlBrick>()
        .with::<WorldBrick>()
        .with::<FrameBrick>()
        .with::<PresentBrick<VerdantState>>()
        .with::<DockBrick>()
}

engine::game!(
    pub struct VerdantGame(Verdant),
    "Verdant",
    bricks,
    scene = "verdant.scene.json"
);
