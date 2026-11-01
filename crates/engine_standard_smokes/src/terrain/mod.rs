mod draws;
mod fixture;
mod shapes;
mod surface;
mod water_sky;

use std::f64::consts::FRAC_PI_3;

use engine_core::smoke::{
    Checked, above, camera, control, environment, number, query, render, renders_until,
};
use engine_core::{SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::requests::seen;
use super::stats::terrain;

pub(crate) use draws::run as draws;
pub(crate) use fixture::{GROUND, fixture};
pub(crate) use shapes::run as shapes;
pub(crate) use surface::run as surface;
pub(crate) use water_sky::run as water_sky;

pub(crate) const FAST: u32 = 50_000_000;
pub(crate) const WAIT_FRAMES: u32 = 600;
pub(crate) const SETTLE_FRAMES: u32 = 2400;

pub(crate) fn heights(session: &mut dyn SmokeSession, points: &[[f64; 2]]) -> Checked<Vec<Value>> {
    let mut samples = Vec::with_capacity(points.len());
    for chunk in points.chunks(100) {
        let mut found = query(session, "terrain_height", json!({"points": chunk}))?;
        if let Value::Array(found) = found["samples"].take() {
            samples.extend(found);
        }
    }
    smoke_eq!(samples.len(), points.len(), "heights_answer_every_point");
    Ok(samples)
}

pub(crate) fn height_at(session: &mut dyn SmokeSession, point: [f64; 2]) -> Checked<Value> {
    let mut samples = heights(session, &[point])?;
    Ok(samples.pop().unwrap_or_default())
}

pub(crate) fn raycast(
    session: &mut dyn SmokeSession,
    origin: [f64; 3],
    max_distance: f64,
) -> Checked<Value> {
    query(
        session,
        "raycast",
        json!({"origin": origin, "direction": [0.0, -1.0, 0.0], "max_distance": max_distance}),
    )
}

pub(crate) fn settle(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut resident = 0;
    let mut rendering = Value::Null;
    for _ in 0..SETTLE_FRAMES {
        rendering = render(session)?;
        let drawn = terrain(&rendering);
        let ready = number(&drawn["tiles"]) > 0.0
            && drawn["loading_tiles"] == 0
            && drawn["fallback_tiles"] == 0;
        resident = if ready { resident + 1 } else { 0 };
        if resident > 1 {
            return Ok(rendering);
        }
    }
    let wanted = json!({"tiles": above(0), "loading_tiles": 0, "fallback_tiles": 0});
    smoke_eq!(
        seen(false, terrain(&rendering), &wanted),
        wanted,
        "terrain_settles"
    );
    Ok(rendering)
}

pub(crate) fn subdivided(session: &mut dyn SmokeSession) -> Checked<Value> {
    let (mut last, mut held) = (Value::Null, 0);
    let (steady, rendering) = renders_until(session, WAIT_FRAMES, |rendering| {
        let leaves = terrain(rendering)["leaf_cut"]["kept"].clone();
        held = if leaves == last { held + 1 } else { 0 };
        last = leaves;
        held >= 16
    })?;
    let wanted = json!({ "steady_frames": 16 });
    smoke_eq!(
        seen(
            steady,
            terrain(&rendering)["leaf_cut"]["kept"].clone(),
            &wanted
        ),
        wanted,
        "the_subdivision_settles"
    );
    Ok(rendering)
}

pub(crate) fn fast() -> Value {
    json!({ "samples_per_frame": FAST })
}

pub(crate) fn view(position: [f64; 3], target: [f64; 3]) -> Value {
    json!({
        "position": position,
        "target": target,
        "vertical_fov_radians": FRAC_PI_3,
        "near": 0.5,
        "far": 4000.0,
    })
}

pub(crate) fn overview() -> Value {
    view([0.0, 80.0, 140.0], [0.0, 30.0, 0.0])
}

pub(crate) fn open(session: &mut dyn SmokeSession, hiker: bool) -> Checked<Value> {
    fixture(session, hiker);
    opened(session)
}

pub(crate) fn opened(session: &mut dyn SmokeSession) -> Checked<Value> {
    control(session, "pause", json!({}))?;
    let changed = environment(session, json!({ "terrain": fast() }))?;
    smoke_eq!(
        &changed["environment_changed"],
        &json!(["terrain"]),
        "terrain_section_changes"
    );
    camera(session, overview())?;
    settle(session)
}

pub(crate) fn viewed(session: &mut dyn SmokeSession) -> Checked<Value> {
    query(session, "terrain_view", json!({ "terrain": GROUND }))
}

pub(crate) fn layer_sweep(
    session: &mut dyn SmokeSession,
    center: [f64; 2],
    half_extent: f64,
    samples: u32,
) -> Checked<Value> {
    query(
        session,
        "terrain_layer_sweep",
        json!({"center": center, "half_extent": half_extent, "samples": samples}),
    )
}
