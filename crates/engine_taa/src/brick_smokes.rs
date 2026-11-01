use serde_json::{Value, json};

use engine_core::smoke::{Checked, ask, control, environment, refuse, render, shot};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};

const SIDE: usize = 12;
const FRAME_MS: f64 = 1000.0 / 60.0;
const QUIET_FRAMES: u32 = 13;
const SETTLE_LIMIT: u32 = 600;
const SECTION: &str = "temporal_antialiasing";
const STATS: &str = "temporal_antialiasing_stats";
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
        "id": "temporal_grid", "revision": 1,
        "camera": {
            "position": [0.0, 30.0, SIDE as f64 * 2.6], "target": [0.0, 0.0, 0.0],
            "fov_degrees": 60, "near": 0.1, "far": 1000,
        },
        "materials": materials,
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": entities,
    })
}

fn inspected(session: &mut dyn SmokeSession, key: &str) -> Checked<Value> {
    let mut inspected = ask(session, "inspect", json!({ "key": key }))?;
    Ok(inspected["value"].take())
}

fn active(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut stats = inspected(session, STATS)?;
    Ok(stats["active"].take())
}

fn frames(session: &mut dyn SmokeSession, count: u32) -> Checked<Value> {
    control(
        session,
        "frames",
        json!({ "count": count, "interval_ms": FRAME_MS }),
    )
}

fn changed(reply: &Value) -> Value {
    reply["environment_changed"].clone()
}

fn taa_defaults(enabled: bool) -> Value {
    json!({
        "enabled": enabled,
        "nearest_motion": true,
        "rounded_clamp": false,
        "particle_response": 2.0,
    })
}

fn shadows_still(session: &mut dyn SmokeSession) -> bool {
    let Ok(mut answered) = session.request("inspect", json!({ "key": "shadow_stats" })) else {
        return true;
    };
    let shadows = answered["result"]["value"].take();
    let idle = |field: &str| shadows[field].as_u64() == Some(0);
    shadows["tile_cut"]["cut"].as_u64() == Some(0) && idle("tiles_drawn") && idle("sky_drawn")
}

fn settled(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let mut calm = 0;
    for _ in 0..SETTLE_LIMIT {
        render(session)?;
        calm = if shadows_still(session) { calm + 1 } else { 0 };
        if calm >= QUIET_FRAMES {
            return Ok(());
        }
    }
    smoke_eq!(calm, QUIET_FRAMES, "exact_frames_settle");
    Ok(())
}

fn exact(session: &mut dyn SmokeSession) -> Checked<Value> {
    settled(session)?;
    shot(session, json!({}))
}

pub(crate) fn temporal_antialiasing(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&grid().to_string());
    control(session, "pause", json!({}))?;
    control(session, "render", json!({}))?;
    session.step(3);
    environment(session, json!({ "haze": { "spread": 0.0 } }))?;
    smoke_eq!(
        inspected(session, SECTION)?,
        taa_defaults(false),
        "off_by_default"
    );
    let without = exact(session)?;
    let on = json!({ "temporal_antialiasing": { "enabled": true } });
    let reply = environment(session, on.clone())?;
    smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "settings_controlled"
    );
    let with = exact(session)?;
    smoke_eq!(with, without, "captures_independent_of_setting");
    frames(session, 90)?;
    smoke_eq!(active(session)?, json!(true), "ordinary_frames_resolve");
    smoke_eq!(
        inspected(session, SECTION)?,
        taa_defaults(true),
        "settings_inspected"
    );
    let off = json!({ "temporal_antialiasing": { "enabled": false } });
    let reply = environment(session, off)?;
    smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "off_controlled"
    );
    smoke_eq!(
        inspected(session, SECTION)?,
        taa_defaults(false),
        "off_inspected"
    );
    frames(session, 30)?;
    smoke_eq!(active(session)?, json!(false), "settings_observed");
    let samples = json!({ "temporal_antialiasing": { "enabled": true, "samples": 8 } });
    let refused = refuse(session, "control", "environment", samples)?;
    smoke_eq!(refused["code"], json!("unknown_field"), "settings_validated");
    environment(session, on)?;
    frames(session, 30)?;
    smoke_eq!(active(session)?, json!(true), "resolves_again");
    dither(session)?;
    sturdier(session)?;
    let after = exact(session)?;
    smoke_eq!(after, with, "captures_exact_after_temporal_frames");
    Ok(())
}

fn dither(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    smoke_eq!(
        inspected(session, "dither")?,
        json!({ "enabled": true }),
        "dither_on_by_default"
    );
    let reply = environment(session, json!({ "dither": { "enabled": false } }))?;
    smoke_eq!(changed(&reply), json!(["dither"]), "dither_controlled");
    smoke_eq!(
        inspected(session, "dither")?,
        json!({ "enabled": false }),
        "dither_off_inspected"
    );
    frames(session, 60)?;
    environment(session, json!({ "dither": { "enabled": true } }))?;
    frames(session, 60)?;
    Ok(())
}

fn sturdier(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let bare = json!({ "enabled": true, "nearest_motion": false, "rounded_clamp": false, "particle_response": 0.0 });
    environment(session, json!({ "temporal_antialiasing": bare.clone() }))?;
    smoke_eq!(
        inspected(session, SECTION)?,
        bare,
        "bare_resolve_inspected"
    );
    frames(session, 60)?;
    let sturdy = json!({ "enabled": true, "nearest_motion": true, "rounded_clamp": true, "particle_response": 2.0 });
    let reply = environment(session, json!({ "temporal_antialiasing": sturdy.clone() }))?;
    smoke_eq!(
        changed(&reply),
        json!(["temporal_antialiasing"]),
        "sturdier_resolve_controlled"
    );
    smoke_eq!(
        inspected(session, SECTION)?,
        sturdy,
        "sturdy_resolve_inspected"
    );
    frames(session, 60)?;
    smoke_eq!(active(session)?, json!(true), "sturdier_resolve_runs");
    let beyond = json!({ "temporal_antialiasing": { "enabled": true, "particle_response": 9 } });
    let refused = refuse(session, "control", "environment", beyond)?;
    smoke_eq!(
        refused["code"],
        json!("out_of_range"),
        "particle_response_bounded"
    );
    let on = json!({ "temporal_antialiasing": { "enabled": true } });
    environment(session, on)?;
    smoke_eq!(
        inspected(session, SECTION)?,
        taa_defaults(true),
        "resolve_restored"
    );
    Ok(())
}
