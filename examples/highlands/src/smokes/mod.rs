mod toon_lighting;
mod torrent;

use engine::BrickList;
use engine::smokes::{Restart, Runs, Smoke, SmokeEntry};
use engine_smoke::{Checked, control, inspect, number};
use engine_core::{SmokeSession, smoke_eq};
use serde_json::json;

const GAME: &str = "highlands";

const FLY_CAMERA_INPUT: (&str, &str) = ("engine_fly_camera", "input_camera");

const SMOKES: [Smoke; 1] = [Smoke {
    name: "highlands_torrent",
    runs: Runs::Gpu,
    run: torrent::run,
}];

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

pub(super) type Aim = ([f32; 3], [f32; 3], f32);

fn aim(session: &mut dyn SmokeSession, (position, target, fov_degrees): Aim) -> Checked<()> {
    control(
        session,
        "camera",
        json!({
            "position": position,
            "target": target,
            "vertical_fov_radians": fov_degrees.to_radians(),
            "near": 0.2,
            "far": 20_000.0,
        }),
    )
    .map(drop)
}

fn settle(session: &mut dyn SmokeSession) -> Checked<()> {
    engine_smoke::settled(session)?;
    let shading = inspect(session, "terrain_shading_stats")?;
    let terrain = inspect(session, "terrain_stats")?;
    smoke_eq!(
        json!({
            "shaded": number(&shading["tiles"]) > 0.0,
            "loading": terrain["loading_tiles"],
            "fallback": terrain["fallback_tiles"],
        }),
        json!({"shaded": true, "loading": 0, "fallback": 0}),
        "terrain_settles"
    );
    Ok(())
}
