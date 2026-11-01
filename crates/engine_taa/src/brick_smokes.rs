use glam::{Quat, Vec3};
use serde_json::{Value, json};

use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{Cube, Transform};

const SIDE: usize = 12;
const FRAME_MS: f64 = 1000.0 / 60.0;
const QUIET_FRAMES: u32 = 13;
const SETTLE_LIMIT: u32 = 600;
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

fn taa_defaults(enabled: bool) -> Value {
    json!({
        "enabled": enabled,
        "nearest_motion": true,
        "rounded_clamp": false,
        "particle_response": 2.0,
    })
}

fn sha(reply: &Value) -> Value {
    reply["result"]["artifact"]["sha256"].clone()
}

fn settled(session: &mut dyn SmokeSession) -> SmokeResult {
    let (mut calm, mut shadows) = (0, Value::Null);
    for _ in 0..SETTLE_LIMIT {
        let reply = control(session, "render", json!({}), "renders_until_settled")?;
        shadows = reply["result"]["rendering"]["shadows"].clone();
        let idle = |field: &str| shadows[field].as_u64() == Some(0);
        let still = shadows["tile_cut"]["cut"].as_u64() == Some(0)
            && idle("tiles_drawn")
            && idle("sky_drawn");
        calm = if still { calm + 1 } else { 0 };
        if calm >= QUIET_FRAMES {
            return Ok(());
        }
    }
    engine_core::smoke_check!(
        calm >= QUIET_FRAMES,
        "exact_frames_settle",
        shadows,
        json!({ "quiet_frames": QUIET_FRAMES })
    );
    Ok(())
}

fn exact(session: &mut dyn SmokeSession, check: &'static str) -> Asked {
    settled(session)?;
    let reply = accepted(session, "capture", json!({}), check)?;
    engine_core::smoke_check!(
        sha(&reply).is_string(),
        "capture_keeps_artifact",
        reply["result"].clone(),
        json!({ "artifact": { "sha256": "string" } })
    );
    Ok(sha(&reply))
}

pub fn temporal_antialiasing(session: &mut dyn SmokeSession) -> SmokeResult {
    grid(session)?;
    let still = json!({ "haze": { "spread": 0.0 } });
    control(session, "environment", still, "haze_holds_still")?;
    let inspect = "inspects_temporal_antialiasing";
    let value = inspected(session, "TemporalAntialiasing", inspect)?;
    engine_core::smoke_eq!(value, taa_defaults(false), "off_by_default");
    let without = exact(session, "captures_without_taa")?;
    let on = json!({ "temporal_antialiasing": { "enabled": true } });
    let reply = control(session, "environment", on.clone(), "turns_taa_on")?;
    engine_core::smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "settings_controlled"
    );
    let with = exact(session, "captures_with_taa")?;
    engine_core::smoke_eq!(with.clone(), without, "captures_independent_of_setting");
    let rendering = frames(session, 90, "runs_temporal_frames")?;
    engine_core::smoke_eq!(
        rendering["temporal_antialiasing"]["active"].clone(),
        json!(true),
        "ordinary_frames_resolve"
    );
    engine_core::smoke_check!(
        passes(&rendering, "temporal_resolve"),
        "resolve_pass_runs",
        rendering["render_passes"].clone(),
        json!("temporal_resolve")
    );
    engine_core::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        taa_defaults(true),
        "settings_inspected"
    );
    let off = json!({ "temporal_antialiasing": { "enabled": false } });
    let reply = control(session, "environment", off, "turns_taa_off")?;
    engine_core::smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "off_controlled"
    );
    engine_core::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        taa_defaults(false),
        "off_inspected"
    );
    let rendering = frames(session, 30, "runs_plain_frames")?;
    engine_core::smoke_eq!(
        rendering["temporal_antialiasing"]["active"].clone(),
        json!(false),
        "settings_observed"
    );
    engine_core::smoke_check!(
        !passes(&rendering, "temporal_resolve"),
        "no_resolve_pass_while_off",
        rendering["render_passes"].clone(),
        json!("no temporal_resolve")
    );
    let samples = json!({ "name": "environment", "args": { "temporal_antialiasing": { "enabled": true, "samples": 8 } } });
    refused(session, samples, "unknown_field", "settings_validated")?;
    control(session, "environment", on, "turns_taa_on_again")?;
    let rendering = frames(session, 30, "runs_temporal_frames_again")?;
    engine_core::smoke_eq!(
        rendering["temporal_antialiasing"]["active"].clone(),
        json!(true),
        "resolves_again"
    );
    dither(session)?;
    sturdier(session)?;
    let after = exact(session, "captures_after_taa")?;
    engine_core::smoke_eq!(after, with, "captures_exact_after_temporal_frames");
    Ok(())
}

fn sturdier(session: &mut dyn SmokeSession) -> SmokeResult {
    let inspect = "inspects_resolve";
    let bare = json!({ "enabled": true, "nearest_motion": false, "rounded_clamp": false, "particle_response": 0.0 });
    let args = json!({ "temporal_antialiasing": bare.clone() });
    control(session, "environment", args, "sets_bare_resolve")?;
    engine_core::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        bare,
        "bare_resolve_inspected"
    );
    frames(session, 60, "runs_bare_frames")?;
    let sturdy = json!({ "enabled": true, "nearest_motion": true, "rounded_clamp": true, "particle_response": 2.0 });
    let args = json!({ "temporal_antialiasing": sturdy.clone() });
    let reply = control(session, "environment", args, "sets_sturdy_resolve")?;
    engine_core::smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "sturdier_resolve_controlled"
    );
    engine_core::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        sturdy,
        "sturdy_resolve_inspected"
    );
    let rendering = frames(session, 60, "runs_sturdy_frames")?;
    engine_core::smoke_eq!(
        rendering["temporal_antialiasing"]["active"].clone(),
        json!(true),
        "sturdier_resolve_runs"
    );
    let beyond = json!({ "name": "environment", "args": { "temporal_antialiasing": { "enabled": true, "particle_response": 9 } } });
    refused(
        session,
        beyond,
        "invalid_environment",
        "particle_response_bounded",
    )?;
    let on = json!({ "temporal_antialiasing": { "enabled": true } });
    control(session, "environment", on, "restores_resolve")?;
    engine_core::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        taa_defaults(true),
        "resolve_restored"
    );
    Ok(())
}
