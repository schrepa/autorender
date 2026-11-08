mod counts;
mod digest;

use engine::BrickList;
use engine::smokes::{Smoke, SmokeEntry};

const GAME: &str = "player";

const SMOKES: [Smoke; 2] = [
    Smoke {
        name: "drawn_and_culled_counts_agree",
        run: counts::run,
    },
    Smoke {
        name: "same_ticks_same_digest",
        run: digest::run,
    },
];

pub fn smokes(bricks: &BrickList) -> engine::Result<Vec<SmokeEntry>> {
    let mut smokes = engine::smokes::registered(bricks)?;
    smokes.extend(SMOKES.iter().map(|smoke| SmokeEntry::of(GAME, smoke)));
    Ok(smokes)
}
