#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod ground;
mod image;
mod partners;
mod poses;
pub mod requests;

use engine_core::{Brick, Bricks, Smoke, Source};

pub struct StandardSmokesBrick;

impl Brick for StandardSmokesBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[
        Smoke {
            name: "drawn_poses_agree",
            run: poses::run,
            without: &[],
        },
        Smoke {
            name: "collision_matches_drawn_ground",
            run: ground::run,
            without: &[],
        },
        Smoke {
            name: "settled_image_repeats",
            run: image::run,
            without: &[],
        },
        Smoke {
            name: "missing_partners_are_reported",
            run: partners::run,
            without: &[partners::LEFT_OUT],
        },
    ];

    fn register(_: &mut Bricks<'_>) {}
}
