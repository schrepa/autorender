use engine_core::smoke::{Checked, ask, control, inspect, number};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

const DESCRIBED: [&str; 3] = ["follow_camera", "camera_target", "camera_shake"];
const SETTLE_STEPS: u32 = 16;
const SETTLE_TICKS: u32 = 15;
const OPEN_BEHIND: f64 = 4.0;
const WALL_AT: f64 = 3.0;

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene(false).to_string());
    control(session, "pause", json!({}))?;
    for name in DESCRIBED {
        ask(session, "describe", json!({"name": name}))?;
    }
    settle(session);
    let open = behind(session)?;
    smoke_eq!(open > OPEN_BEHIND, true, "follows_at_its_distance");
    shakes(session, open)?;
    session.source(&scene(true).to_string());
    settle(session);
    let walled = behind(session)?;
    smoke_eq!(walled < WALL_AT, true, "wall_pulls_the_camera_in");
    Ok(())
}

fn settle(session: &mut dyn SmokeSession) {
    for _ in 0..SETTLE_STEPS {
        session.step(SETTLE_TICKS);
    }
}

fn behind(session: &mut dyn SmokeSession) -> Checked<f64> {
    let camera = inspect(session, "Camera")?;
    Ok(number(&camera["position"][2]))
}

fn shakes(session: &mut dyn SmokeSession, steady: f64) -> Checked<()> {
    control(
        session,
        "camera_shake",
        json!({"strength": 1.0, "seconds": 1.0}),
    )?;
    session.step(SETTLE_TICKS);
    let shaken = behind(session)?;
    smoke_eq!(
        (shaken - steady).abs() > 1e-4,
        true,
        "shake_moves_the_camera"
    );
    settle(session);
    let calm = behind(session)?;
    smoke_eq!((calm - steady).abs() < 1e-2, true, "shake_decays");
    Ok(())
}

fn scene(walled: bool) -> Value {
    let mut entities = vec![json!({
        "id": "hero",
        "camera_target": {},
        "transform": {"translation": [0.0, 0.0, 0.0]},
    })];
    if walled {
        entities.push(json!({
            "id": "wall",
            "collider": {"shape": {"kind": "cuboid", "half_extents": [8.0, 8.0, 0.5]}},
            "transform": {"translation": [0.0, 0.0, WALL_AT]},
        }));
    }
    json!({
        "id": if walled { "follow-walled" } else { "follow-open" },
        "revision": 1,
        "camera": {
            "position": [0.0, 2.0, 10.0],
            "target": [0.0, 1.5, 0.0],
            "fov_degrees": 60,
            "near": 0.1,
            "far": 1000,
        },
        "materials": [],
        "meshes": [],
        "entities": entities,
    })
}
