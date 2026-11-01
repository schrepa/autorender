mod agent;
mod gpu_presentation;
mod native;
mod pixels;

use engine::BrickList;
use engine::smokes::{Runs, Smoke, SmokeEntry};

const GAME: &str = "player";

const SMOKES: [Smoke; 3] = [
    Smoke {
        name: "agent",
        runs: Runs::Gpu,
        run: agent::agent,
    },
    Smoke {
        name: "native",
        runs: Runs::Gpu,
        run: native::native,
    },
    Smoke {
        name: "gpu_presentation",
        runs: Runs::Gpu,
        run: gpu_presentation::gpu_presentation,
    },
];

pub fn smokes(bricks: &BrickList) -> engine::Result<Vec<SmokeEntry>> {
    let mut smokes = engine::smokes::registered(bricks)?;
    smokes.extend(SMOKES.iter().map(|smoke| SmokeEntry::of(GAME, smoke)));
    Ok(smokes)
}
