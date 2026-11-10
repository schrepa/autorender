#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod calls;
mod colliders;
mod empty;
mod far_ground;
mod ground;
mod image;
mod movers;
mod partners;
mod poses;
pub mod requests;
mod unpaused;

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
            name: "calls_by_name_in_lists",
            run: calls::run,
            without: &[],
        },
        Smoke {
            name: "empty_scene_settles",
            run: empty::run,
            without: &[],
        },
        Smoke {
            name: "missing_partners_are_reported",
            run: partners::run,
            without: &[partners::LEFT_OUT],
        },
        Smoke {
            name: "standard_list_starts_clean",
            run: engine_smoke::starts_clean,
            without: &[],
        },
        Smoke {
            name: "removed_mover_releases_its_claim",
            run: movers::run,
            without: &[],
        },
        Smoke {
            name: "colliders_over_the_limit_are_refused_alone",
            run: colliders::run,
            without: &[],
        },
        Smoke {
            name: "far_ground_stands_on_terrain_covers",
            run: far_ground::run,
            without: &[],
        },
        Smoke {
            name: "calls_by_name_apply_without_a_pause",
            run: unpaused::run,
            without: &[],
        },
    ];

    fn register(_: &mut Bricks<'_>) {}
}
