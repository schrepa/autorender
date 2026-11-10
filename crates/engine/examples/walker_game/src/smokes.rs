use engine_core::{Smoke, SmokeFailure, SmokeSession, smoke_check, smoke_eq};
use engine_smoke::{control, query, settled};
use serde_json::{Value, json};

const WALKER: &str = "walker";

const DESTINATION: [f64; 2] = [6.0, 0.0];

const TICKS: u32 = 30;

const ROUNDS: u32 = 60;

pub(crate) const SMOKES: [Smoke; 2] = [
    Smoke {
        name: "walker_crosses_a_collider_floor",
        run,
    },
    Smoke {
        name: "walker_without_grounding_is_refused",
        run: run_ungrounded,
    },
];

fn scene(walker: Value) -> Value {
    json!({
        "id": "walker", "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {"id": "floor", "transform": {"translation": [0.0, -0.5, 0.0]},
             "collider": {"shape": {"kind": "cuboid", "half_extents": [16.0, 0.5, 16.0]}}},
            walker,
            {"id": "camera", "transform": {"translation": [0.0, 12.0, 12.0], "rotation_degrees": [-45.0, 0.0, 0.0]},
             "node": {"clock": "real", "mobility": "moving"},
             "camera": {"near": 0.1, "far": 200.0, "projection": {"Perspective": {"vertical_fov_radians": 0.9}}},
             "view": {}},
        ],
    })
}

fn grounded_walker() -> Value {
    json!({"id": WALKER, "transform": {"translation": [-6.0, 0.0, 0.0]},
           "nav_agent": {"destination": DESTINATION}, "grounded": {"offset": 0.0},
           "node": {"clock": "real", "mobility": "moving"}})
}

fn ungrounded_walker() -> Value {
    json!({"id": WALKER, "transform": {"translation": [-6.0, 0.0, 0.0]},
           "nav_agent": {"destination": DESTINATION},
           "node": {"clock": "real", "mobility": "moving"}})
}

fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene(grounded_walker()).to_string());
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

fn run_ungrounded(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene(ungrounded_walker()).to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    let wanted = json!({ "status": "refused", "refused": "steps_not_taken" });
    let mut seen = Value::Null;
    for _ in 0..ROUNDS {
        session.step(TICKS);
        seen = query(session, "nav_agent", json!({ "entity": WALKER }))?;
        let status = seen["status"].as_str().unwrap_or_default();
        smoke_check!(
            !matches!(status, "unreachable" | "arrived"),
            "ungrounded_walker_never_arrives",
            seen,
            wanted
        );
        if status == "refused" {
            break;
        }
    }
    let got = json!({ "status": seen["status"], "refused": seen["refused"] });
    smoke_eq!(got, wanted, "ungrounded_walker_is_refused");
    let why = query(session, "why", json!({ "entity": WALKER }))?;
    let reasons: Vec<&Value> = why["refused"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|refused| refused["part"] == "navigation")
        .map(|refused| &refused["reason"]["reason"])
        .collect();
    smoke_eq!(
        json!(reasons),
        json!(["nav_steps_not_taken"]),
        "why_names_the_untaken_steps"
    );
    Ok(())
}
