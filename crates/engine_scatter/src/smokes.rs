use serde_json::{Value, json};

use engine_core::{SmokeFailure, SmokeSession, smoke_eq};

const SIDE: usize = 12;
const FRAME_MS: f64 = 1000.0 / 60.0;
const PAINTS: [[f32; 4]; 4] = [
    [0.85, 0.35, 0.2, 1.0],
    [0.25, 0.6, 0.85, 1.0],
    [0.4, 0.75, 0.3, 1.0],
    [0.9, 0.75, 0.25, 1.0],
];

type Asked = Result<Value, SmokeFailure>;

fn accepted(session: &mut dyn SmokeSession, verb: &str, body: Value, check: &'static str) -> Asked {
    let reply = session.request(verb, body);
    smoke_eq!(
        reply.as_ref().err().map(|refused| &refused["error"]),
        None,
        check
    );
    Ok(reply.unwrap_or_default())
}

fn control(session: &mut dyn SmokeSession, name: &str, args: Value, check: &'static str) -> Asked {
    accepted(
        session,
        "control",
        json!({ "name": name, "args": args }),
        check,
    )
}

fn inspected(session: &mut dyn SmokeSession, component: &str, check: &'static str) -> Asked {
    let reply = accepted(session, "inspect", json!({ "component": component }), check)?;
    Ok(reply["result"]["value"].clone())
}

fn frames(session: &mut dyn SmokeSession, count: u32, check: &'static str) -> Asked {
    let reply = control(
        session,
        "frames",
        json!({ "count": count, "interval_ms": FRAME_MS }),
        check,
    )?;
    Ok(reply["result"]["rendering"].clone())
}

fn changed(reply: &Value) -> Value {
    reply["result"]["environment_changed"].clone()
}

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

fn placed(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&grid().to_string());
    let eye = SIDE as f64 * 2.6;
    control(
        session,
        "camera",
        json!({
            "position": [0.0, 30.0, eye],
            "target": [0.0, 0.0, 0.0],
            "vertical_fov_radians": 60.0_f64.to_radians(),
            "near": 0.1,
            "far": 1000.0,
        }),
        "camera_placed",
    )?;
    control(session, "pause", json!({}), "pauses")?;
    control(session, "render", json!({}), "renders")?;
    control(session, "step", json!({ "ticks": 3 }), "steps")?;
    Ok(())
}

pub(crate) fn foliage_settings(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    placed(session)?;
    let defaults = json!({
        "transmission": 1.0,
        "variation": 1.0,
        "hashed_cutoff": false,
        "ground_blend": 0.4_f32,
        "crown_depth": 2.0,
    });
    let inspect = "inspects_foliage";
    smoke_eq!(
        inspected(session, "foliage", inspect)?,
        defaults.clone(),
        "foliage_defaults"
    );
    let hashed = json!({ "foliage": { "hashed_cutoff": true } });
    let reply = control(session, "environment", hashed, "sets_hashed_cutoff")?;
    smoke_eq!(changed(&reply), json!(["foliage"]), "foliage_controlled");
    smoke_eq!(
        inspected(session, "foliage", inspect)?["hashed_cutoff"].clone(),
        json!(true),
        "hashed_cutoff_inspected"
    );
    frames(session, 60, "runs_foliage_frames")?;
    let reset = json!({ "foliage": {} });
    control(session, "environment", reset, "resets_foliage")?;
    smoke_eq!(
        inspected(session, "foliage", inspect)?,
        defaults,
        "foliage_reset"
    );
    Ok(())
}

pub(crate) fn scatter_covers(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    smoke_eq!(session.frame(), true, "frame_drawn");
    Ok(())
}
