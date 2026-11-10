use engine_core::{PassName, Smoke, SmokeFailure, SmokeSession, smoke_check};
use engine_smoke::{above, control, counted_in_joined_passes, number, settled};
use serde_json::{Value, json};

use crate::bricks::glimmer::{GLIMMER, GlimmerPass};

const FRAMES: u32 = 8;

pub(crate) const SMOKES: [Smoke; 1] = [Smoke {
    name: "glimmer_counted_in_every_pass",
    run,
}];

fn scene() -> Value {
    json!({
        "id": "glimmer-passes", "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {"id": "spark", "drawn": {"kind": GLIMMER.name()},
             "transform": {"translation": [0.0, 1.0, 0.0]}},
            {"id": "camera", "transform": {"translation": [0.0, 3.0, 8.0], "rotation_degrees": [-20.0, 0.0, 0.0]},
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
    let counted = counted_in_joined_passes(session, GLIMMER.name(), FRAMES)?;
    let own = counted
        .get(GlimmerPass::NAME)
        .cloned()
        .unwrap_or(Value::Null);
    smoke_check!(
        own["state"] == "drew" && number(&own["draws"]) > 0.0,
        "glimmer_draws_in_its_own_pass",
        own,
        json!({ "state": "drew", "draws": above(0) })
    );
    Ok(())
}
