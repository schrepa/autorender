mod jobs;
mod lod;
mod toon_lighting;
mod torrent;

use engine::smokes::{Runs, Smoke, SmokeEntry};
use engine_core::smoke::{Checked, control, inspect, number, query, render};
use engine_core::{SmokeSession, smoke_eq};
use serde_json::{Value, json};

const GAME: &str = "highlands";

const SETTLE_FRAMES: u32 = 20_000;

const SMOKES: [Smoke; 3] = [
    Smoke {
        name: "highlands_jobs",
        runs: Runs::Gpu,
        run: jobs::run,
    },
    Smoke {
        name: "highlands_lod",
        runs: Runs::Gpu,
        run: lod::run,
    },
    Smoke {
        name: "highlands_torrent",
        runs: Runs::Gpu,
        run: torrent::run,
    },
];

const TOON: Smoke = Smoke {
    name: "highlands_toon_lighting",
    runs: Runs::Gpu,
    run: toon_lighting::run,
};

pub fn smokes() -> Vec<SmokeEntry> {
    let mut smokes: Vec<SmokeEntry> = SMOKES
        .iter()
        .map(|smoke| SmokeEntry::of(GAME, smoke))
        .collect();
    smokes.push(SmokeEntry {
        list: Some(highlands_bricks::toon),
        ..SmokeEntry::of(GAME, &TOON)
    });
    smokes
}

fn open(session: &mut dyn SmokeSession, in_place: bool) -> Checked<()> {
    if in_place {
        control(session, "jobs", json!({ "workers": 0 }))?;
    }
    session.source(crate::scene::SCENE);
    control(session, "pause", json!({}))?;
    Ok(())
}

fn aim(session: &mut dyn SmokeSession, view: &str) -> Checked<()> {
    control(session, "scene_view", json!({ "name": view })).map(drop)
}

fn heights(session: &mut dyn SmokeSession, points: &[[f64; 2]]) -> Checked<Vec<Value>> {
    let mut samples = Vec::with_capacity(points.len());
    for chunk in points.chunks(100) {
        let mut found = query(session, "terrain_height", json!({ "points": chunk }))?;
        if let Value::Array(found) = found["samples"].take() {
            samples.extend(found);
        }
    }
    smoke_eq!(samples.len(), points.len(), "heights_answer_every_point");
    Ok(samples)
}

fn height_at(session: &mut dyn SmokeSession, point: [f64; 2]) -> Checked<Value> {
    let mut samples = heights(session, &[point])?;
    Ok(samples.pop().unwrap_or_default())
}

fn settle(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut resident = 0;
    let mut terrain = Value::Null;
    for _ in 0..SETTLE_FRAMES {
        render(session)?;
        terrain = inspect(session, "terrain_stats")?;
        let ready = number(&terrain["resident_tiles"]) > 0.0
            && terrain["loading_tiles"] == 0
            && terrain["fallback_tiles"] == 0;
        resident = if ready { resident + 1 } else { 0 };
        if resident > 1 {
            break;
        }
    }
    smoke_eq!(resident > 1, true, "terrain_settles");
    Ok(terrain)
}
