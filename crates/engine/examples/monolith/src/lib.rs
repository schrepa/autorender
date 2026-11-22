#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use brick_hiz::HizBrick;
use brick_nanite::NaniteBrick;
use brick_present::PresentBrick;
use brick_touch::TouchBrick;
use engine::Start;
use engine::prelude::*;

pub mod layout;
pub mod names;
pub mod pbr;
pub mod state;

pub mod bricks {
    pub mod control;
    pub mod dock;
    pub mod frame;
    pub mod sanctuary;
}

use crate::bricks::control::ControlBrick;
use crate::bricks::dock::DockBrick;
use crate::bricks::frame::FrameBrick;
use crate::bricks::sanctuary::SanctuaryBrick;
use crate::state::MonolithState;

#[derive(Resource, Default)]
pub struct Monolith;

impl Game for Monolith {
    fn start() -> Start {
        let start = Start::default();
        Start {
            catch_up_seconds: 2.0 / f64::from(start.ticks_per_second),
            #[cfg(target_arch = "wasm32")]
            max_pixel_ratio: Some(1.25),
            ..start
        }
    }
}

pub fn bricks<L: Listing>(list: L) -> L {
    list.with::<TouchBrick>()
        .with::<HizBrick>()
        .with::<NaniteBrick>()
        .with::<ControlBrick>()
        .with::<SanctuaryBrick>()
        .with::<FrameBrick>()
        .with::<PresentBrick<MonolithState>>()
        .with::<DockBrick>()
}

engine::game!(
    pub struct MonolithGame(Monolith),
    "Monolith",
    bricks,
    scene = "monolith.scene.json"
);
