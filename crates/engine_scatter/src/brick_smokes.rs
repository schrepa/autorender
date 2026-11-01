use glam::{Quat, Vec3};
use serde_json::{Value, json};

use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{Cube, Transform};

const SIDE: usize = 12;
const FRAME_MS: f64 = 1000.0 / 60.0;
const PAINTS: [[f32; 4]; 4] = [
    [0.85, 0.35, 0.2, 1.0],
    [0.25, 0.6, 0.85, 1.0],
    [0.4, 0.75, 0.3, 1.0],
    [0.9, 0.75, 0.25, 1.0],
];

type Asked = Result<Value, SmokeFailure>;

fn reply(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Value {
    match session.request(verb, body) {
        Ok(reply) | Err(reply) => reply,
    }
}

fn accepted(session: &mut dyn SmokeSession, verb: &str, body: Value, check: &'static str) -> Asked {
    let reply = reply(session, verb, body);
    engine_core::smoke_check!(
        reply["ok"] == true,
        check,
        reply["error"].clone(),
        json!({ "ok": true })
    );
    Ok(reply)
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

fn grid(session: &mut dyn SmokeSession) -> SmokeResult {
    let world = session.app().world_mut();
    world.spawn((
        Transform {
            translation: Vec3::new(0.0, -0.5, 0.0),
            rotation: Quat::IDENTITY,
            scale: Vec3::new(SIDE as f32 * 6.0, 1.0, SIDE as f32 * 6.0),
        },
        Cube {
            color: [0.55, 0.58, 0.52, 1.0],
        },
    ));
    let middle = (SIDE as f32 - 1.0) / 2.0;
    for index in 0..SIDE * SIDE {
        let (row, column) = ((index / SIDE) as f32, (index % SIDE) as f32);
        let scale = 0.8 + (index % 5) as f32 * 0.3;
        let turn = ((index * 37) % 360) as f32;
        world.spawn((
            Transform {
                translation: Vec3::new((column - middle) * 4.0, scale / 2.0, (row - middle) * 4.0),
                rotation: Quat::from_rotation_y(turn.to_radians()),
                scale: Vec3::splat(scale),
            },
            Cube {
                color: PAINTS[index % PAINTS.len()],
            },
        ));
    }
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

pub fn foliage_settings(session: &mut dyn SmokeSession) -> SmokeResult {
    grid(session)?;
    let defaults = json!({
        "transmission": 1.0,
        "variation": 1.0,
        "hashed_cutoff": false,
        "ground_blend": 0.4_f32,
        "crown_depth": 2.0,
    });
    let inspect = "inspects_foliage";
    engine_core::smoke_eq!(
        inspected(session, "Foliage", inspect)?,
        defaults.clone(),
        "foliage_defaults"
    );
    let hashed = json!({ "foliage": { "hashed_cutoff": true } });
    let reply = control(session, "environment", hashed, "sets_hashed_cutoff")?;
    engine_core::smoke_eq!(changed(&reply), json!(["foliage"]), "foliage_controlled");
    engine_core::smoke_eq!(
        inspected(session, "Foliage", inspect)?["hashed_cutoff"].clone(),
        json!(true),
        "hashed_cutoff_inspected"
    );
    frames(session, 60, "runs_foliage_frames")?;
    let reset = json!({ "foliage": {} });
    control(session, "environment", reset, "resets_foliage")?;
    engine_core::smoke_eq!(
        inspected(session, "Foliage", inspect)?,
        defaults,
        "foliage_reset"
    );
    Ok(())
}
