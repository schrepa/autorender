mod lod;
mod toon_lighting;
mod torrent;

use engine::BrickList;
use engine::smokes::{Restart, Runs, Smoke, SmokeEntry};
use engine_core::smoke::{Checked, control, inspect, number, render};
use engine_core::{SmokeSession, smoke_eq};
use serde_json::json;

const GAME: &str = "highlands";

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

const TOON: Smoke = Smoke {
    name: "highlands_toon_lighting",
    runs: Runs::Gpu,
    run: |session| toon_lighting::standard(session).map(drop),
};

pub fn smokes(bricks: &BrickList) -> engine::Result<Vec<SmokeEntry>> {
    let mut smokes = engine::smokes::registered(bricks)?;
    smokes.retain(|entry| (entry.owner, entry.name) != FLY_CAMERA_INPUT);
    smokes.extend(SMOKES.iter().map(|smoke| SmokeEntry::of(GAME, smoke)));
    smokes.push(SmokeEntry {
        list: Some(crate::toon),
        restart: Some(Restart {
            first: toon_lighting::standard,
            then: toon_lighting::toon,
        }),
        ..SmokeEntry::of(GAME, &TOON)
    });
    Ok(smokes)
}

fn open(session: &mut dyn SmokeSession) -> Checked<()> {
    session.source(crate::scene::SCENE);
    control(session, "pause", json!({}))?;
    Ok(())
}

fn aim(session: &mut dyn SmokeSession, view: &str) -> Checked<()> {
    control(session, "scene_view", json!({ "name": view })).map(drop)
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
