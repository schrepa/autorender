mod jobs;
mod lod;
mod toon_lighting;
mod torrent;

use engine::BrickList;
use engine::smokes::{Restart, Runs, Smoke, SmokeEntry};
use engine_core::smoke::{Checked, control, inspect, number, query, render};
use engine_core::{SmokeSession, smoke_eq};
use serde_json::{Value, json};

const GAME: &str = "highlands";

const LIGHTING: &str = "lighting";

const FLY_CAMERA_INPUT: (&str, &str) = ("engine_fly_camera", "input_camera");

const SETTLE_FRAMES: u32 = 20_000;

const SMOKES: [Smoke; 2] = [
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

const JOBS: Smoke = Smoke {
    name: "highlands_jobs",
    runs: Runs::Gpu,
    run: |session| jobs::workers(session).map(drop),
};

const TOON: Smoke = Smoke {
    name: "highlands_toon_lighting",
    runs: Runs::Gpu,
    run: |session| toon_lighting::standard(session).map(drop),
};

pub fn smokes(bricks: &BrickList) -> engine::Result<Vec<SmokeEntry>> {
    let mut smokes = engine::smokes::registered(bricks)?;
    smokes.retain(|entry| (entry.owner, entry.name) != FLY_CAMERA_INPUT);
    smokes.push(SmokeEntry {
        restart: Some(Restart {
            first: jobs::workers,
            then: jobs::page,
        }),
        ..SmokeEntry::of(GAME, &JOBS)
    });
    smokes.extend(SMOKES.iter().map(|smoke| SmokeEntry::of(GAME, smoke)));
    smokes.push(SmokeEntry {
        list: Some(highlands_bricks::toon),
        restart: Some(Restart {
            first: toon_lighting::standard,
            then: toon_lighting::toon,
        }),
        ..SmokeEntry::of(GAME, &TOON)
    });
    let toon = highlands_bricks::toon(BrickList::default());
    for entry in engine::smokes::registered(&toon)? {
        if entry.owner != LIGHTING || entry.without.is_some() {
            continue;
        }
        smokes.push(SmokeEntry {
            name: Box::leak(format!("{}.{}", TOON.name, entry.name).into_boxed_str()),
            list: Some(highlands_bricks::toon),
            ..entry
        });
    }
    Ok(smokes)
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

fn settle(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut resident = 0;
    for _ in 0..SETTLE_FRAMES {
        render(session)?;
        let shading = inspect(session, "terrain_shading_stats")?;
        let terrain = inspect(session, "terrain_stats")?;
        let ready = number(&shading["tiles"]) > 0.0
            && terrain["loading_tiles"] == 0
            && terrain["fallback_tiles"] == 0;
        resident = if ready { resident + 1 } else { 0 };
        if resident > 1 {
            return Ok(());
        }
    }
    smoke_eq!(resident > 1, true, "terrain_settles");
    Ok(())
}
