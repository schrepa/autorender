use glam::{Quat, Vec3};
use serde_json::{Value, json};

use crate::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use crate::{Cube, Transform};

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
    let (ok, reply) = match session.request(verb, body) {
        Ok(reply) => (true, reply),
        Err(reply) => (false, reply),
    };
    if reply.get("ok").is_some_and(Value::is_boolean) {
        return reply;
    }
    if ok {
        json!({ "ok": true, "result": reply })
    } else {
        json!({ "ok": false, "error": reply })
    }
}

fn accepted(session: &mut dyn SmokeSession, verb: &str, body: Value, check: &'static str) -> Asked {
    let reply = reply(session, verb, body);
    crate::smoke_check!(
        reply["ok"] == true,
        check,
        reply["error"].clone(),
        json!({ "ok": true })
    );
    Ok(reply)
}

fn refused(
    session: &mut dyn SmokeSession,
    body: Value,
    code: &str,
    check: &'static str,
) -> Asked {
    let reply = reply(session, "control", body);
    crate::smoke_check!(
        reply["error"]["code"] == code,
        check,
        reply["error"].clone(),
        json!({ "code": code })
    );
    Ok(reply)
}

fn control(
    session: &mut dyn SmokeSession,
    name: &str,
    args: Value,
    check: &'static str,
) -> Asked {
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

fn exact(session: &mut dyn SmokeSession, check: &'static str) -> Asked {
    let reply = accepted(session, "capture", json!({}), check)?;
    crate::smoke_check!(
        sha(&reply).is_string(),
        "capture_keeps_artifact",
        reply["result"].clone(),
        json!({ "artifact": { "sha256": "string" } })
    );
    Ok(sha(&reply))
}

pub(crate) fn temporal_antialiasing(session: &mut dyn SmokeSession) -> SmokeResult {
    grid(session)?;
    let inspect = "inspects_temporal_antialiasing";
    let value = inspected(session, "TemporalAntialiasing", inspect)?;
    crate::smoke_eq!(value, taa_defaults(false), "off_by_default");
    let without = exact(session, "captures_without_taa")?;
    let on = json!({ "temporal_antialiasing": { "enabled": true } });
    let reply = control(session, "environment", on.clone(), "turns_taa_on")?;
    crate::smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "settings_controlled"
    );
    let with = exact(session, "captures_with_taa")?;
    crate::smoke_eq!(with.clone(), without, "captures_independent_of_setting");
    let rendering = frames(session, 90, "runs_temporal_frames")?;
    crate::smoke_eq!(
        rendering["temporal_antialiasing"]["active"].clone(),
        json!(true),
        "ordinary_frames_resolve"
    );
    crate::smoke_check!(
        passes(&rendering, "temporal_resolve"),
        "resolve_pass_runs",
        rendering["render_passes"].clone(),
        json!("temporal_resolve")
    );
    crate::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        taa_defaults(true),
        "settings_inspected"
    );
    let off = json!({ "temporal_antialiasing": { "enabled": false } });
    let reply = control(session, "environment", off, "turns_taa_off")?;
    crate::smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "off_controlled"
    );
    crate::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        taa_defaults(false),
        "off_inspected"
    );
    let rendering = frames(session, 30, "runs_plain_frames")?;
    crate::smoke_eq!(
        rendering["temporal_antialiasing"]["active"].clone(),
        json!(false),
        "settings_observed"
    );
    crate::smoke_check!(
        !passes(&rendering, "temporal_resolve"),
        "no_resolve_pass_while_off",
        rendering["render_passes"].clone(),
        json!("no temporal_resolve")
    );
    let samples = json!({ "name": "environment", "args": { "temporal_antialiasing": { "enabled": true, "samples": 8 } } });
    refused(session, samples, "unknown_field", "settings_validated")?;
    control(session, "environment", on, "turns_taa_on_again")?;
    let rendering = frames(session, 30, "runs_temporal_frames_again")?;
    crate::smoke_eq!(
        rendering["temporal_antialiasing"]["active"].clone(),
        json!(true),
        "resolves_again"
    );
    dither(session)?;
    sturdier(session)?;
    let after = exact(session, "captures_after_taa")?;
    crate::smoke_eq!(after, with, "captures_exact_after_temporal_frames");
    Ok(())
}

fn dither(session: &mut dyn SmokeSession) -> SmokeResult {
    let inspect = "inspects_dither";
    crate::smoke_eq!(
        inspected(session, "Dither", inspect)?,
        json!({ "enabled": true }),
        "dither_on_by_default"
    );
    let off = json!({ "dither": { "enabled": false } });
    let reply = control(session, "environment", off, "turns_dither_off")?;
    crate::smoke_eq!(changed(&reply), json!(["dither"]), "dither_controlled");
    crate::smoke_eq!(
        inspected(session, "Dither", inspect)?,
        json!({ "enabled": false }),
        "dither_off_inspected"
    );
    frames(session, 60, "runs_undithered_frames")?;
    let on = json!({ "dither": { "enabled": true } });
    control(session, "environment", on, "turns_dither_on")?;
    frames(session, 60, "runs_dithered_frames")?;
    Ok(())
}

fn sturdier(session: &mut dyn SmokeSession) -> SmokeResult {
    let inspect = "inspects_resolve";
    let bare = json!({ "enabled": true, "nearest_motion": false, "rounded_clamp": false, "particle_response": 0.0 });
    let args = json!({ "temporal_antialiasing": bare.clone() });
    control(session, "environment", args, "sets_bare_resolve")?;
    crate::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        bare,
        "bare_resolve_inspected"
    );
    frames(session, 60, "runs_bare_frames")?;
    let sturdy = json!({ "enabled": true, "nearest_motion": true, "rounded_clamp": true, "particle_response": 2.0 });
    let args = json!({ "temporal_antialiasing": sturdy.clone() });
    let reply = control(session, "environment", args, "sets_sturdy_resolve")?;
    crate::smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "sturdier_resolve_controlled"
    );
    crate::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        sturdy,
        "sturdy_resolve_inspected"
    );
    let rendering = frames(session, 60, "runs_sturdy_frames")?;
    crate::smoke_eq!(
        rendering["temporal_antialiasing"]["active"].clone(),
        json!(true),
        "sturdier_resolve_runs"
    );
    let beyond = json!({ "name": "environment", "args": { "temporal_antialiasing": { "enabled": true, "particle_response": 9 } } });
    refused(session, beyond, "invalid_environment", "particle_response_bounded")?;
    let on = json!({ "temporal_antialiasing": { "enabled": true } });
    control(session, "environment", on, "restores_resolve")?;
    crate::smoke_eq!(
        inspected(session, "TemporalAntialiasing", inspect)?,
        taa_defaults(true),
        "resolve_restored"
    );
    Ok(())
}

pub(crate) fn foliage_settings(session: &mut dyn SmokeSession) -> SmokeResult {
    grid(session)?;
    let defaults = json!({
        "transmission": 1.0,
        "variation": 1.0,
        "hashed_cutoff": false,
        "ground_blend": 0.4_f32,
        "crown_depth": 2.0,
    });
    let inspect = "inspects_foliage";
    crate::smoke_eq!(
        inspected(session, "Foliage", inspect)?,
        defaults.clone(),
        "foliage_defaults"
    );
    let hashed = json!({ "foliage": { "hashed_cutoff": true } });
    let reply = control(session, "environment", hashed, "sets_hashed_cutoff")?;
    crate::smoke_eq!(changed(&reply), json!(["foliage"]), "foliage_controlled");
    crate::smoke_eq!(
        inspected(session, "Foliage", inspect)?["hashed_cutoff"].clone(),
        json!(true),
        "hashed_cutoff_inspected"
    );
    frames(session, 60, "runs_foliage_frames")?;
    let reset = json!({ "foliage": {} });
    control(session, "environment", reset, "resets_foliage")?;
    crate::smoke_eq!(
        inspected(session, "Foliage", inspect)?,
        defaults,
        "foliage_reset"
    );
    Ok(())
}

pub(crate) fn culling_settings(session: &mut dyn SmokeSession) -> SmokeResult {
    grid(session)?;
    let on = json!({ "enabled": true });
    let off = json!({ "enabled": false });
    crate::smoke_eq!(
        inspected(session, "OcclusionCulling", "inspects_occlusion")?,
        on.clone(),
        "occlusion_on_by_default"
    );
    crate::smoke_eq!(
        inspected(session, "SubpixelCulling", "inspects_subpixel")?,
        on.clone(),
        "subpixel_on_by_default"
    );
    let rendering = previewed(session, "previews_with_culling")?;
    crate::smoke_eq!(
        rendering["occlusion_culling"]["active"].clone(),
        json!(true),
        "gpu_frames_cull_occluded"
    );
    crate::smoke_eq!(
        rendering["subpixel_culling"]["active"].clone(),
        json!(true),
        "gpu_frames_cull_subpixel"
    );
    crate::smoke_check!(
        passes(&rendering, "occlusion_test"),
        "occlusion_test_runs",
        rendering["render_passes"].clone(),
        json!("occlusion_test")
    );
    let both = json!({ "occlusion_culling": off.clone(), "subpixel_culling": off.clone() });
    let reply = control(session, "environment", both, "turns_culling_off")?;
    crate::smoke_eq!(
        changed(&reply),
        json!(["occlusion_culling", "subpixel_culling"]),
        "culling_settings_controlled"
    );
    crate::smoke_eq!(
        inspected(session, "OcclusionCulling", "inspects_occlusion_off")?,
        off,
        "culling_settings_inspected"
    );
    let rendering = previewed(session, "previews_without_culling")?;
    crate::smoke_eq!(
        json!([
            rendering["occlusion_culling"]["active"],
            rendering["subpixel_culling"]["active"],
            rendering["occlusion_culling"]["instances"],
            rendering["subpixel_culling"]["instances"],
        ]),
        json!([false, false, null, null]),
        "culling_settings_observed"
    );
    crate::smoke_check!(
        !passes(&rendering, "occlusion_test"),
        "no_occlusion_test_while_off",
        rendering["render_passes"].clone(),
        json!("no occlusion_test")
    );
    let phases = json!({ "name": "environment", "args": { "occlusion_culling": { "enabled": true, "phases": 2 } } });
    refused(session, phases, "unknown_field", "culling_settings_validated")?;
    let both = json!({ "occlusion_culling": on.clone(), "subpixel_culling": on });
    control(session, "environment", both, "turns_culling_on")?;
    let rendering = previewed(session, "previews_with_culling_again")?;
    crate::smoke_eq!(
        rendering["occlusion_culling"]["active"].clone(),
        json!(true),
        "culling_runs_again"
    );
    Ok(())
}
