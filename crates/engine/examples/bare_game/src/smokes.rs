use engine_core::{PassName, Smoke, SmokeFailure, SmokeSession, smoke_check, smoke_eq};
use engine_smoke::{above, control, counted_in_joined_passes, joined_passes, number, settled};
use serde_json::{Value, json};

use crate::bricks::plates::{PLATE, PLATE_MODULE, PlatesPass};

const FRAMES: u32 = 8;

pub(crate) const SMOKES: [Smoke; 2] = [
    Smoke {
        name: "bare_game_draws_its_own_kind",
        run,
        without: &[],
    },
    Smoke {
        name: "bare_game_starts_clean",
        run: engine_smoke::starts_clean,
        without: &[],
    },
];

fn scene() -> Value {
    json!({
        "id": "bare-plates", "revision": 1,
        "materials": [{
            "id": "plates", "color": [1.0, 1.0, 1.0, 1.0],
            "slots": {"base_color": PLATE_MODULE}, "lighting": "unlit", "cull": "none",
        }],
        "meshes": [],
        "entities": [
            {"id": "floor", "drawn": {"kind": PLATE.name(), "material": "plates"},
             "transform": {"scale": [8.0, 1.0, 8.0]}},
            {"id": "camera", "transform": {"translation": [0.0, 8.0, 8.0], "rotation_degrees": [-45.0, 0.0, 0.0]},
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
    let joined = joined_passes(session, PLATE.name())?;
    smoke_eq!(
        json!(joined),
        json!([PlatesPass::NAME]),
        "plates_join_only_their_own_pass"
    );
    let counted = counted_in_joined_passes(session, PLATE.name(), FRAMES)?;
    let own = counted
        .get(PlatesPass::NAME)
        .cloned()
        .unwrap_or(Value::Null);
    smoke_check!(
        own["state"]["state"] == "drew" && number(&own["state"]["draws"]) > 0.0,
        "plates_draw_in_their_own_pass",
        own,
        json!({ "state": "drew", "draws": above(0) })
    );
    Ok(())
}
