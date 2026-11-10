use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use engine_smoke::{Checked, component, control, query, render, settled, settled_when};
use serde_json::{Value, json};

use super::requests::seen;

const EPSILON: f64 = 0.0002;

fn fixture() -> Value {
    json!({
        "id": "drawn-poses", "revision": 1,
        "materials": [{"id": "plain", "color": [0.6, 0.35, 0.25, 1]}],
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "entities": [
            {"id": "player", "drawn": {"kind": "mesh", "mesh": "box", "material": "plain"}, "transform": {"translation": [0, 0.5, 4]}},
            {"id": "player-accessory", "parent": "player", "drawn": {"kind": "mesh", "mesh": "box", "material": "plain"},
             "transform": {"translation": [0, 0.8, 0], "scale": [0.3, 0.3, 0.3]}},
            {"id": "camera", "transform": {"translation": [0, 9, 16], "rotation_degrees": [-26.5651, 0, 0]}, "node": {"clock": "real", "mobility": "moving"}, "camera": {"near": 0.1, "far": 500, "projection": {"Perspective": {"vertical_fov_radians": 0.9599311}}}, "view": {}},
        ],
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&fixture().to_string());
    control(session, "pause", json!({}))?;
    settled(session)?;
    motion(session)?;
    drawn(session)
}

fn numbers(values: &Value) -> Vec<f64> {
    values
        .as_array()
        .into_iter()
        .flatten()
        .map(|value| value.as_f64().unwrap_or(f64::NAN))
        .collect()
}

fn near_all(values: &Value, expected: &[f64]) -> bool {
    values.as_array().is_some_and(|values| {
        values.len() == expected.len()
            && values.iter().zip(expected).all(|(value, &expected)| {
                value
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && (value - expected).abs() <= EPSILON)
            })
    })
}

fn translation(session: &mut dyn SmokeSession, id: &str) -> Checked<Vec<f64>> {
    Ok(numbers(
        &component(session, id, "GlobalTransform")?["translation"],
    ))
}

fn motion(session: &mut dyn SmokeSession) -> Checked<()> {
    let player_before = translation(session, "player")?;
    let accessory_before = translation(session, "player-accessory")?;
    let mut moved = player_before.clone();
    if let Some(z) = moved.get_mut(2) {
        *z -= 0.65;
    }
    control(
        session,
        "teleport",
        json!({"moves": [{"entity": "player", "translation": moved}]}),
    )?;
    session.step(6);
    let delta = |after: &[f64], before: &[f64]| -> Vec<f64> {
        after
            .iter()
            .zip(before)
            .map(|(after, before)| after - before)
            .collect()
    };
    let player_delta = delta(&translation(session, "player")?, &player_before);
    let accessory_delta = delta(
        &translation(session, "player-accessory")?,
        &accessory_before,
    );
    let length = player_delta
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let wanted = json!({"length": 0.1});
    smoke_eq!(
        seen(length > 0.1, json!(player_delta), &wanted),
        wanted,
        "player_displaced"
    );
    let wanted = json!(player_delta);
    smoke_eq!(
        seen(
            near_all(&json!(accessory_delta), &player_delta),
            json!(accessory_delta),
            &wanted,
        ),
        wanted,
        "exact_step_parent_motion"
    );
    Ok(())
}

fn flat_rows(rows: &Value) -> Vec<f64> {
    rows.as_array()
        .into_iter()
        .flatten()
        .flat_map(numbers)
        .collect()
}

fn drawn(session: &mut dyn SmokeSession) -> Checked<()> {
    let mut cpu = serde_json::Map::new();
    let mut gpu = serde_json::Map::new();
    let mut agree = true;
    for id in ["player", "player-accessory"] {
        let drawn = settled_when(
            session,
            |session| {
                let mut rendering = render(session)?;
                rendering["drawn"] = query(session, "mesh_drawn", json!({"entity": id}))?;
                Ok(rendering)
            },
            |rendering| rendering["drawn"]["read_back"]["pose"].is_array(),
        )?["drawn"]
            .take();
        let (computed, read) = (
            flat_rows(&drawn["drawn_pose"]),
            flat_rows(&drawn["read_back"]["pose"]),
        );
        agree &= computed.len() == 12 && near_all(&json!(read), &computed);
        cpu.insert(id.to_owned(), json!(computed));
        gpu.insert(id.to_owned(), json!(read));
    }
    let wanted = Value::Object(cpu);
    smoke_eq!(
        seen(agree, Value::Object(gpu), &wanted),
        wanted,
        "cpu_and_gpu_drawn_poses_agree"
    );
    Ok(())
}
