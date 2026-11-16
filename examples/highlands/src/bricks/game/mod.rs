#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/bricks/game.rs"));

mod cave;
mod content;
mod control;
mod views;

pub(crate) use control::HighlandsControl;

use engine_core::{Brick, Bricks, JobKind, Schedule, Source, stats};

stats! {
    #[section("highlands_stats")]
    pub struct HighlandsStats {
        pub failed_jobs: u64,
        pub latest_job_failure: Option<String>,
    }
}

pub struct HighlandsBrick;

impl Brick for HighlandsBrick {
    const SOURCE: &'static Source = &SOURCE;
    const JOBS: &'static [&'static JobKind] = &[
        &crate::jobs::BOULDER,
        &crate::jobs::PLANT,
        &crate::jobs::CAVE,
    ];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .stats::<HighlandsStats>()
            .resource::<content::Content>()
            .resource::<cave::Cave>()
            .resource::<views::Views>()
            .resource::<views::BodiesView>()
            .operation::<control::HighlandsControl>()
            .systems(
                Schedule::Presentation,
                (
                    crate::sources::read,
                    content::arrive,
                    views::watch,
                    cave::open,
                    content::publish,
                ),
            )
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
