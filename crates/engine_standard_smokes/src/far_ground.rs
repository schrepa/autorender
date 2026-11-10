use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, component, control, near, number, query, render, settled};
use serde_json::{Value, json};

use super::requests::seen;

const LANTERN: &str = "lantern";
const FAR: [f64; 2] = [400.0, 400.0];
const DROPPED_FROM: f64 = 300.0;
const WITHIN: f64 = 1.0;
const TICKS: u32 = 4;
const FIELD: &str = "field";

fn fixture() -> Value {
    json!({
        "id": "far-ground", "revision": 1,
        "materials": [],
        "meshes": [],
        "entities": [
            {"id": "ground", "terrain": {
                "origin": [-512.0, 0.0, -512.0], "tiles": [4, 4], "spacing": 1.0,
                "layers": [{"type": "noise", "seed": 5, "height": 24.0, "wavelength": 480.0,
                            "ridges": 0.0, "warp": 0.0, "roughness": 0.0}],
            }},
            {"id": LANTERN, "transform": {"translation": [FAR[0], DROPPED_FROM, FAR[1]]},
             "node": {"mobility": "moving"}, "grounded": {"offset": 0.0}},
            {"id": "camera", "transform": {"translation": [-400.0, 60.0, -400.0], "rotation_degrees": [0.0, 225.0, 0.0]},
             "node": {"clock": "real", "mobility": "moving"},
             "camera": {"near": 0.5, "far": 4000.0, "projection": {"Perspective": {"vertical_fov_radians": 1.0}}},
             "view": {}},
        ],
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&fixture().to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    session.step(TICKS);
    render(session)?;
    beyond_the_ring(session)?;
    stood(session)
}

fn beyond_the_ring(session: &mut dyn SmokeSession) -> Checked<()> {
    let ray = query(
        session,
        "raycast",
        json!({"origin": [FAR[0], DROPPED_FROM, FAR[1]], "direction": [0.0, -1.0, 0.0], "max_distance": 600.0}),
    )?;
    smoke_eq!(ray["hit"], Value::Null, "no_collision_tile_holds_the_far_point");
    Ok(())
}

fn stood(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut under = query(session, "terrain_height", json!({ "points": [FAR] }))?;
    let ground = under["samples"][0]["height"].take();
    let mut placed = component(session, LANTERN, "GlobalTransform")?;
    let height = placed["translation"][1].take();
    let mut grounded = component(session, LANTERN, "grounded")?;
    let surface = grounded["surface"].take();
    let got = json!({"height": height, "ground": ground, "surface": surface});
    let wanted = json!({
        "height": near(&ground, WITHIN),
        "ground": "a terrain height",
        "surface": FIELD,
    });
    let held = ground.is_number()
        && (number(&height) - number(&ground)).abs() < WITHIN
        && surface == FIELD;
    smoke_eq!(
        seen(held, got, &wanted),
        wanted,
        "far_ground_is_answered_from_terrain_covers"
    );
    Ok(())
}
