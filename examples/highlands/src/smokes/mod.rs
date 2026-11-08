mod torrent;

use engine::BrickList;
use engine::smokes::{Smoke, SmokeEntry};
use engine_core::SmokeSession;
use engine_smoke::{Checked, control};
use serde_json::json;

const GAME: &str = "highlands";

const SMOKES: [Smoke; 1] = [Smoke {
    name: "highlands_torrent",
    run: torrent::run,
}];

pub fn smokes(bricks: &BrickList) -> engine::Result<Vec<SmokeEntry>> {
    let mut smokes = engine::smokes::registered(bricks)?;
    smokes.extend(SMOKES.iter().map(|smoke| SmokeEntry::of(GAME, smoke)));
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
