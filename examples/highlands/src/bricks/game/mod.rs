#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/bricks/game.rs"));

mod bodies;
mod cave;
mod caves;
mod content;
mod control;
mod random;
mod rocks;
mod trees;
mod views;

use engine_core::{Brick, Bricks, JobKind, Schedule, Source, stats};

stats! {
    #[section("highlands_stats")]
    pub struct HighlandsStats {
        #[pending] pub content: u32,
        pub failed_jobs: u64,
        pub latest_job_failure: Option<String>,
    }
}

pub struct HighlandsBrick;

impl Brick for HighlandsBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[
        &content::BOULDER,
        &content::CONIFER,
        &content::CONIFER_SHADOW,
        &content::CAVE,
    ];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .stats::<HighlandsStats>()
            .resource::<content::Content>()
            .resource::<cave::Cave>()
            .resource::<bodies::Bodies>()
            .resource::<views::Views>()
            .resource::<views::BodiesView>()
            .operation::<control::HighlandsControl>()
            .systems(
                Schedule::Presentation,
                (
                    content::arrive,
                    views::watch,
                    cave::open,
                    bodies::place,
                    content::publish,
                ),
            )
            .systems(Schedule::FixedPreSimulation, bodies::wait_for_ground)
            .systems(Schedule::WorldSync, views::look);
    }
}

fn report_failure(event: &str, error: &engine_core::Error) {
    let record = serde_json::json!({
        "event": event,
        "error": error,
    });
    eprintln!("{record}");
}
