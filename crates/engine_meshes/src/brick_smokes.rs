use glam::{Quat, Vec3};
use serde_json::{Value, json};

use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{Cube, Transform};

const SIDE: usize = 12;
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

fn refused(session: &mut dyn SmokeSession, body: Value, code: &str, check: &'static str) -> Asked {
    let reply = reply(session, "control", body);
    engine_core::smoke_check!(
        reply["error"]["code"] == code,
        check,
        reply["error"].clone(),
        json!({ "code": code })
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

fn previewed(session: &mut dyn SmokeSession, check: &'static str) -> Asked {
    let reply = accepted(session, "capture", json!({ "alpha": 0.5 }), check)?;
    Ok(reply["result"]["rendering"].clone())
}

fn changed(reply: &Value) -> Value {
    reply["result"]["environment_changed"].clone()
}

fn passes(rendering: &Value, pass: &str) -> bool {
    rendering["render_passes"]
        .as_array()
        .is_some_and(|passes| passes.iter().any(|name| name == pass))
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

pub(crate) fn culling_settings(session: &mut dyn SmokeSession) -> SmokeResult {
    grid(session)?;
    let on = json!({ "enabled": true });
    let off = json!({ "enabled": false });
    engine_core::smoke_eq!(
        inspected(session, "OcclusionCulling", "inspects_occlusion")?,
        on.clone(),
        "occlusion_on_by_default"
    );
    engine_core::smoke_eq!(
        inspected(session, "SubpixelCulling", "inspects_subpixel")?,
        on.clone(),
        "subpixel_on_by_default"
    );
    let rendering = previewed(session, "previews_with_culling")?;
    engine_core::smoke_eq!(
        rendering["occlusion_culling"]["active"].clone(),
        json!(true),
        "gpu_frames_cull_occluded"
    );
    engine_core::smoke_eq!(
        rendering["subpixel_culling"]["active"].clone(),
        json!(true),
        "gpu_frames_cull_subpixel"
    );
    engine_core::smoke_check!(
        passes(&rendering, "occlusion_test"),
        "occlusion_test_runs",
        rendering["render_passes"].clone(),
        json!("occlusion_test")
    );
    let both = json!({ "occlusion_culling": off.clone(), "subpixel_culling": off.clone() });
    let reply = control(session, "environment", both, "turns_culling_off")?;
    engine_core::smoke_eq!(
        changed(&reply),
        json!(["occlusion_culling", "subpixel_culling"]),
        "culling_settings_controlled"
    );
    engine_core::smoke_eq!(
        inspected(session, "OcclusionCulling", "inspects_occlusion_off")?,
        off,
        "culling_settings_inspected"
    );
    let rendering = previewed(session, "previews_without_culling")?;
    engine_core::smoke_eq!(
        json!([
            rendering["occlusion_culling"]["active"],
            rendering["subpixel_culling"]["active"],
            rendering["occlusion_culling"]["instances"],
            rendering["subpixel_culling"]["instances"],
        ]),
        json!([false, false, null, null]),
        "culling_settings_observed"
    );
    engine_core::smoke_check!(
        !passes(&rendering, "occlusion_test"),
        "no_occlusion_test_while_off",
        rendering["render_passes"].clone(),
        json!("no occlusion_test")
    );
    let phases = json!({ "name": "environment", "args": { "occlusion_culling": { "enabled": true, "phases": 2 } } });
    refused(
        session,
        phases,
        "unknown_field",
        "culling_settings_validated",
    )?;
    let both = json!({ "occlusion_culling": on.clone(), "subpixel_culling": on });
    control(session, "environment", both, "turns_culling_on")?;
    let rendering = previewed(session, "previews_with_culling_again")?;
    engine_core::smoke_eq!(
        rendering["occlusion_culling"]["active"].clone(),
        json!(true),
        "culling_runs_again"
    );
    Ok(())
}
