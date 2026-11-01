use serde_json::{Value, json};

use engine_core::smoke::{Checked, ask, control, environment};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};

const SIDE: usize = 12;
const FRAME_MS: f64 = 1000.0 / 60.0;
const PAINTS: [[f32; 4]; 4] = [
    [0.85, 0.35, 0.2, 1.0],
    [0.25, 0.6, 0.85, 1.0],
    [0.4, 0.75, 0.3, 1.0],
    [0.9, 0.75, 0.25, 1.0],
];

fn grid() -> Value {
    let mut entities = vec![json!({
        "id": "ground", "mesh": "box", "material": "ground",
        "transform": {
            "translation": [0.0, -0.5, 0.0],
            "scale": [SIDE as f64 * 6.0, 1.0, SIDE as f64 * 6.0],
        },
    })];
    let middle = (SIDE as f64 - 1.0) / 2.0;
    for index in 0..SIDE * SIDE {
        let (row, column) = ((index / SIDE) as f64, (index % SIDE) as f64);
        let scale = 0.8 + (index % 5) as f64 * 0.3;
        entities.push(json!({
            "id": format!("piece-{index}"), "mesh": "box",
            "material": format!("paint-{}", index % PAINTS.len()),
            "transform": {
                "translation": [(column - middle) * 4.0, scale / 2.0, (row - middle) * 4.0],
                "rotation_degrees": [0, (index * 37) % 360, 0],
                "scale": [scale, scale, scale],
            },
        }));
    }
    let mut materials = vec![json!({ "id": "ground", "color": [0.55, 0.58, 0.52, 1.0] })];
    materials.extend(
        PAINTS
            .iter()
            .enumerate()
            .map(|(index, color)| json!({ "id": format!("paint-{index}"), "color": color })),
    );
    json!({
        "id": "foliage_grid", "revision": 1,
        "camera": {
            "position": [0.0, 30.0, SIDE as f64 * 2.6], "target": [0.0, 0.0, 0.0],
            "fov_degrees": 60, "near": 0.1, "far": 1000,
        },
        "materials": materials,
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": entities,
    })
}

fn inspected(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut inspected = ask(session, "inspect", json!({ "key": "foliage" }))?;
    Ok(inspected["value"].take())
}

pub(crate) fn foliage_settings(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&grid().to_string());
    control(session, "pause", json!({}))?;
    control(session, "render", json!({}))?;
    session.step(3);
    let defaults = json!({
        "transmission": 1.0,
        "variation": 1.0,
        "hashed_cutoff": false,
        "ground_blend": 0.4_f32,
        "crown_depth": 2.0,
    });
    smoke_eq!(inspected(session)?, defaults, "foliage_defaults");
    let hashed = json!({ "foliage": { "hashed_cutoff": true } });
    let reply = environment(session, hashed)?;
    smoke_eq!(
        reply["environment_changed"],
        json!(["foliage"]),
        "foliage_controlled"
    );
    smoke_eq!(
        inspected(session)?["hashed_cutoff"],
        json!(true),
        "hashed_cutoff_inspected"
    );
    control(
        session,
        "frames",
        json!({ "count": 60, "interval_ms": FRAME_MS }),
    )?;
    smoke_eq!(session.frame(), true, "foliage_frames_drawn");
    environment(session, json!({ "foliage": {} }))?;
    smoke_eq!(inspected(session)?, defaults, "foliage_reset");
    Ok(())
}
