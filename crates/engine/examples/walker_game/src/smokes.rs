use engine_core::{Smoke, SmokeFailure, SmokeSession, smoke_check};
use engine_smoke::{control, query, settled};
use serde_json::{Value, json};

const WALKER: &str = "walker";

const DESTINATION: [f64; 2] = [6.0, 0.0];

const TICKS: u32 = 30;

const ROUNDS: u32 = 60;

pub(crate) const SMOKES: [Smoke; 1] = [Smoke {
    name: "walker_crosses_a_collider_floor",
    run,
}];

fn scene() -> Value {
    json!({
        "id": "walker", "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {"id": "floor", "transform": {"translation": [0.0, -0.5, 0.0]},
             "collider": {"shape": {"kind": "cuboid", "half_extents": [16.0, 0.5, 16.0]}}},
            {"id": WALKER, "transform": {"translation": [-6.0, 0.0, 0.0]},
             "nav_agent": {"destination": DESTINATION}, "grounded": {"offset": 0.0},
             "node": {"clock": "real", "mobility": "moving"}},
            {"id": "camera", "transform": {"translation": [0.0, 12.0, 12.0], "rotation_degrees": [-45.0, 0.0, 0.0]},
             "node": {"clock": "real", "mobility": "moving"},
             "camera": {"near": 0.1, "far": 200.0, "projection": {"Perspective": {"vertical_fov_radians": 0.9}}},
             "view": {}},
        ],
    })
}

fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene().to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    let mut seen = Value::Null;
    for _ in 0..ROUNDS {
        session.step(TICKS);
        seen = query(session, "nav_agent", json!({ "entity": WALKER }))?;
        let status = seen["status"].as_str().unwrap_or_default();
        smoke_check!(
            !matches!(status, "unreachable" | "refused"),
            "walker_finds_a_way",
            seen,
            json!({ "status": "walking" })
        );
        if status == "arrived" {
            return Ok(());
        }
    }
    Err(SmokeFailure::new(
        "walker_arrives",
        &seen,
        &json!({ "status": "arrived", "destination": DESTINATION }),
        concat!(file!(), ":", line!()),
    ))
}
