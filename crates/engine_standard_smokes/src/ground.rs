use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, above, below, control, near, number, query, settled};
use serde_json::{Value, json};

use super::requests::seen;

const GROUND: &str = "ground";

fn fixture() -> Value {
    json!({
        "id": "collision-ground", "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {"id": GROUND, "terrain": {
                "origin": [-128.0, 0.0, -128.0], "tiles": [1, 1], "spacing": 1.0,
                "layers": [{"type": "noise", "seed": 3, "height": 140.0, "wavelength": 600.0}],
            }},
            {"id": "camera", "transform": {"translation": [0.0, 80.0, 140.0], "rotation_degrees": [-19.6538, 0.0, 0.0]},
             "node": {"clock": "real", "mobility": "moving"},
             "camera": {"near": 0.5, "far": 4000.0, "projection": {"Perspective": {"vertical_fov_radians": std::f64::consts::FRAC_PI_3}}}, "view": {}},
        ],
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&fixture().to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    ray(session)?;
    swept(session)
}

fn ray(session: &mut dyn SmokeSession) -> Checked<()> {
    let ray = query(
        session,
        "raycast",
        json!({"origin": [10.0, 200.0, -7.0], "direction": [0.0, -1.0, 0.0], "max_distance": 400.0}),
    )?;
    let under = query(session, "terrain_height", json!({"points": [[10.0, -7.0]]}))?;
    let height = &under["samples"][0]["height"];
    let wanted = json!({"entity": GROUND, "height": near(height, 2e-3)});
    smoke_eq!(
        seen(
            ray["hit"]["entity"]["id"] == GROUND
                && (number(&ray["hit"]["point"][1]) - number(height)).abs() < 2e-3,
            ray["hit"].clone(),
            &wanted,
        ),
        wanted,
        "ray_and_query_agree"
    );
    Ok(())
}

fn swept(session: &mut dyn SmokeSession) -> Checked<()> {
    let sweep = query(
        session,
        "terrain_sweep",
        json!({"center": [0.0, 0.0], "half_extent": 120.0, "samples": 24}),
    )?;
    let collided = &sweep["collision"];
    let wanted = json!({"compared": above(100), "worst": {"max_difference": below(5e-3)}});
    smoke_eq!(
        seen(
            number(&collided["compared"]) > 100.0
                && number(&collided["worst"]["max_difference"]) < 5e-3,
            collided.clone(),
            &wanted,
        ),
        wanted,
        "collision_matches_the_drawn_ground"
    );
    Ok(())
}
