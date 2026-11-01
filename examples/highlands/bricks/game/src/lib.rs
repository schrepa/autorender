#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod bodies;
mod cave;
mod caves;
mod hiker;
mod random;
mod views;

use engine_core::{Brick, Bricks, ComponentKind, SceneComponents, Schedule, Source, settings};

const HIKER: &str = "hiker";
const BANK_BOULDER: &str = "highlands_bank_boulder";
const LOG: &str = "highlands_log";
const BARK: &str = "highlands_bark";
const PALE: &str = "highlands_pale";
const STONE: &str = "highlands_stone";
const WALL_ROCK: &str = "highlands_wall_rock";
const CAVE_WALL: &str = "highlands_cave_wall";

settings! {
    #[section("highlands")]
    #[serde(deny_unknown_fields)]
    pub struct HighlandsSettings {
        pub cave: Option<[f32; 3]> = None => any,
        pub bodies: u32 = 0 => any,
        pub stand: Option<[f32; 4]> = None => any,
        pub watch_bodies: bool = false => any,
    }
}

pub struct HighlandsBrick;

impl Brick for HighlandsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<HighlandsSettings>()
            .resource::<views::BodiesView>()
            .join::<SceneComponents>(ComponentKind::of::<hiker::Hiker>(HIKER), None)
            .systems(Schedule::FixedSimulation, hiker::hike)
            .systems(Schedule::FixedPostSimulation, (views::watch, cave::open))
            .systems(Schedule::WorldSync, (views::look, bodies::place));
    }
}

fn report_failure(event: &str, error: &engine_core::Error) {
    let record = serde_json::json!({
        "event": event,
        "error": error,
    });
    eprintln!("{record}");
}
