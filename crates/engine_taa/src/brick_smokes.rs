use serde_json::{Value, json};

use engine_core::{SmokeFailure, SmokeSession, smoke_eq};

use crate::temporal::PASS;

const SIDE: usize = 12;
const FRAME_MS: f64 = 1000.0 / 60.0;
const SECTION: &str = "temporal_antialiasing";
const STATS: &str = "temporal_antialiasing_stats";
const DITHER: &str = "dither";
const PAINTS: [[f32; 4]; 4] = [
    [0.85, 0.35, 0.2, 1.0],
    [0.25, 0.6, 0.85, 1.0],
    [0.4, 0.75, 0.3, 1.0],
    [0.9, 0.75, 0.25, 1.0],
];

type Asked = Result<Value, SmokeFailure>;

fn seen(held: bool, got: Value, expected: &Value) -> Value {
    if held { expected.clone() } else { got }
}

fn reply(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Value {
    match session.request(verb, body) {
        Ok(reply) | Err(reply) => reply,
    }
}

fn accepted(session: &mut dyn SmokeSession, verb: &str, body: Value, check: &'static str) -> Asked {
    let reply = reply(session, verb, body);
    let wanted = json!({ "ok": true });
    smoke_eq!(
        seen(reply["ok"] == true, reply["error"].clone(), &wanted),
        wanted,
        check
    );
    Ok(reply)
}

fn refused(session: &mut dyn SmokeSession, body: Value, code: &str, check: &'static str) -> Asked {
    let reply = reply(session, "control", body);
    let wanted = json!({ "code": code });
    smoke_eq!(
        seen(
            reply["error"]["code"] == code,
            reply["error"].clone(),
            &wanted
        ),
        wanted,
        check
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

fn active(rendering: &Value) -> Value {
    rendering[STATS]["active"].clone()
}

fn scene() -> Value {
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
        "id": "temporal_grid", "revision": 1,
        "materials": materials,
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": entities,
    })
}

fn grid(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene().to_string());
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

pub(crate) fn temporal_antialiasing(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    grid(session)?;
    let still = json!({ "haze": { "spread": 0.0 } });
    control(session, "environment", still, "haze_holds_still")?;
    let inspect = "inspects_temporal_antialiasing";
    let value = inspected(session, SECTION, inspect)?;
    smoke_eq!(value, taa_defaults(false), "off_by_default");
    let on = json!({ "temporal_antialiasing": { "enabled": true } });
    let reply = control(session, "environment", on.clone(), "turns_taa_on")?;
    smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "settings_controlled"
    );
    let rendering = frames(session, 90, "runs_temporal_frames")?;
    smoke_eq!(active(&rendering), json!(true), "ordinary_frames_resolve");
    let resolving = json!(PASS.pass);
    smoke_eq!(
        seen(
            passes(&rendering, PASS.pass),
            rendering["render_passes"].clone(),
            &resolving
        ),
        resolving,
        "resolve_pass_runs"
    );
    smoke_eq!(
        inspected(session, SECTION, inspect)?,
        taa_defaults(true),
        "settings_inspected"
    );
    let off = json!({ "temporal_antialiasing": { "enabled": false } });
    let reply = control(session, "environment", off, "turns_taa_off")?;
    smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "off_controlled"
    );
    smoke_eq!(
        inspected(session, SECTION, inspect)?,
        taa_defaults(false),
        "off_inspected"
    );
    let rendering = frames(session, 30, "runs_plain_frames")?;
    smoke_eq!(active(&rendering), json!(false), "settings_observed");
    let plain = json!(format!("no {}", PASS.pass));
    smoke_eq!(
        seen(
            !passes(&rendering, PASS.pass),
            rendering["render_passes"].clone(),
            &plain
        ),
        plain,
        "no_resolve_pass_while_off"
    );
    let samples = json!({ "name": "environment", "args": { "temporal_antialiasing": { "enabled": true, "samples": 8 } } });
    refused(session, samples, "unknown_field", "settings_validated")?;
    control(session, "environment", on, "turns_taa_on_again")?;
    let rendering = frames(session, 30, "runs_temporal_frames_again")?;
    smoke_eq!(active(&rendering), json!(true), "resolves_again");
    dither(session)?;
    sturdier(session)
}

fn dither(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let inspect = "inspects_dither";
    smoke_eq!(
        inspected(session, DITHER, inspect)?,
        json!({ "enabled": true }),
        "dither_on_by_default"
    );
    let off = json!({ "dither": { "enabled": false } });
    let reply = control(session, "environment", off, "turns_dither_off")?;
    smoke_eq!(changed(&reply), json!([DITHER]), "dither_controlled");
    smoke_eq!(
        inspected(session, DITHER, inspect)?,
        json!({ "enabled": false }),
        "dither_off_inspected"
    );
    frames(session, 60, "runs_undithered_frames")?;
    let on = json!({ "dither": { "enabled": true } });
    control(session, "environment", on, "turns_dither_on")?;
    frames(session, 60, "runs_dithered_frames")?;
    Ok(())
}

fn sturdier(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let inspect = "inspects_resolve";
    let bare = json!({ "enabled": true, "nearest_motion": false, "rounded_clamp": false, "particle_response": 0.0 });
    let args = json!({ "temporal_antialiasing": bare.clone() });
    control(session, "environment", args, "sets_bare_resolve")?;
    smoke_eq!(
        inspected(session, SECTION, inspect)?,
        bare,
        "bare_resolve_inspected"
    );
    frames(session, 60, "runs_bare_frames")?;
    let sturdy = json!({ "enabled": true, "nearest_motion": true, "rounded_clamp": true, "particle_response": 2.0 });
    let args = json!({ "temporal_antialiasing": sturdy.clone() });
    let reply = control(session, "environment", args, "sets_sturdy_resolve")?;
    smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "sturdier_resolve_controlled"
    );
    smoke_eq!(
        inspected(session, SECTION, inspect)?,
        sturdy,
        "sturdy_resolve_inspected"
    );
    let rendering = frames(session, 60, "runs_sturdy_frames")?;
    smoke_eq!(active(&rendering), json!(true), "sturdier_resolve_runs");
    let beyond = json!({ "name": "environment", "args": { "temporal_antialiasing": { "enabled": true, "particle_response": 9 } } });
    refused(
        session,
        beyond,
        "invalid_environment",
        "particle_response_bounded",
    )?;
    let on = json!({ "temporal_antialiasing": { "enabled": true } });
    control(session, "environment", on, "restores_resolve")?;
    smoke_eq!(
        inspected(session, SECTION, inspect)?,
        taa_defaults(true),
        "resolve_restored"
    );
    Ok(())
}
