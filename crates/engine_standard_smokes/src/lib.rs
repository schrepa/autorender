#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod calls;
mod empty;
mod ground;
mod image;
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
        },
        Smoke {
            name: "collision_matches_drawn_ground",
            run: ground::run,
        },
        Smoke {
            name: "settled_image_repeats",
            run: image::run,
        },
        Smoke {
            name: "calls_by_name_in_lists",
            run: calls::run,
        },
        Smoke {
            name: "empty_scene_settles",
            run: empty::run,
        },
    ];

    fn register(_: &mut Bricks<'_>) {}
}
