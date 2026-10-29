mod draws;
mod fixture;
mod shapes;
mod surface;
mod water_sky;

use std::f64::consts::FRAC_PI_3;

use bevy_ecs::prelude::Entity;
use engine_core::smoke::{SmokeFailure, SmokeSession};
use engine_core::{Bricks, smoke_check, smoke_eq};
use serde::Serialize;
use serde_json::{Value, json};

use crate::{Layer, PathStats, SurroundingsStatus, Terrain, TerrainBakes, Terrains};

pub use fixture::{FIXTURE, Fixture, fixture, name};

pub type Checked<T> = Result<T, SmokeFailure>;

pub const FAST: u32 = 50_000_000;
pub const WAIT_FRAMES: u32 = 600;
pub const SETTLE_FRAMES: u32 = 2400;
pub const COMPILE_FRAMES: u32 = 20_000;

pub(crate) fn register(bricks: &mut Bricks<'_>) {
    bricks
        .smoke("terrain_draws", true, draws::run)
        .smoke("terrain_shapes", true, shapes::run)
        .smoke("terrain_surface", true, surface::run)
        .smoke("terrain_water_sky", true, water_sky::run);
}

pub fn ask(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Checked<Value> {
    let reply = session.request(verb, body);
    smoke_check!(
        reply.is_ok(),
        "request_answered",
        &reply.as_ref().err().map(|refused| &refused["error"]),
        &verb
    );
    Ok(reply
        .map(|mut answered| answered["result"].take())
        .unwrap_or_default())
}

pub fn control(session: &mut dyn SmokeSession, name: &str, args: Value) -> Checked<Value> {
    ask(session, "control", json!({"name": name, "args": args}))
}

pub fn query(session: &mut dyn SmokeSession, name: &str, args: Value) -> Checked<Value> {
    ask(session, "query", json!({"name": name, "args": args}))
}

pub fn refuse(
    session: &mut dyn SmokeSession,
    verb: &str,
    name: &str,
    args: Value,
) -> Checked<Value> {
    let reply = session.request(verb, json!({"name": name, "args": args}));
    smoke_check!(
        reply.is_err(),
        "request_refused",
        &reply.as_ref().ok().map(|answered| &answered["result"]),
        &name
    );
    Ok(reply
        .err()
        .map(|mut refused| refused["error"].take())
        .unwrap_or_default())
}

pub fn environment(session: &mut dyn SmokeSession, sections: Value) -> Checked<Value> {
    control(session, "environment", sections)
}

pub fn camera(session: &mut dyn SmokeSession, view: Value) -> Checked<Value> {
    control(session, "camera", view)
}

pub fn inspect(session: &mut dyn SmokeSession, component: &str) -> Checked<Value> {
    let mut inspected = ask(session, "inspect", json!({"component": component}))?;
    Ok(inspected["value"].take())
}

pub fn component(session: &mut dyn SmokeSession, id: &str, component: &str) -> Checked<Value> {
    let mut inspected = ask(
        session,
        "inspect",
        json!({"entity": id, "component": component}),
    )?;
    Ok(inspected["entity"]["components"][component].take())
}

pub fn render(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut rendered = control(session, "render", json!({}))?;
    Ok(rendered["rendering"].take())
}

pub fn present(session: &mut dyn SmokeSession, alpha: f64) -> Checked<Value> {
    let mut presented = control(session, "present", json!({"alpha": alpha}))?;
    Ok(presented["rendering"].take())
}

pub fn heights(session: &mut dyn SmokeSession, points: &[[f64; 2]]) -> Checked<Vec<Value>> {
    let mut samples = Vec::with_capacity(points.len());
    for chunk in points.chunks(100) {
        let mut found = query(session, "terrain_height", json!({"points": chunk}))?;
        if let Value::Array(found) = found["samples"].take() {
            samples.extend(found);
        }
    }
    smoke_check!(
        samples.len() == points.len(),
        "heights_answer_every_point",
        &samples.len(),
        &points.len()
    );
    Ok(samples)
}

pub fn height_at(session: &mut dyn SmokeSession, point: [f64; 2]) -> Checked<Value> {
    let mut samples = heights(session, &[point])?;
    Ok(samples.pop().unwrap_or_default())
}

pub fn raycast(
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

pub fn renders_until(
    session: &mut dyn SmokeSession,
    frames: u32,
    mut done: impl FnMut(&Value) -> bool,
) -> Checked<(bool, Value)> {
    let mut rendering = Value::Null;
    for _ in 0..frames {
        rendering = render(session)?;
        if done(&rendering) {
            return Ok((true, rendering));
        }
    }
    Ok((false, rendering))
}

pub fn settle(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut resident = 0;
    let mut rendering = Value::Null;
    for _ in 0..SETTLE_FRAMES {
        rendering = render(session)?;
        let terrain = &rendering["terrain"];
        let ready = number(&terrain["tiles"]) > 0.0
            && terrain["loading_tiles"] == 0
            && terrain["fallback_tiles"] == 0;
        resident = if ready { resident + 1 } else { 0 };
        if resident > 1 {
            return Ok(rendering);
        }
    }
    smoke_check!(
        false,
        "terrain_settles",
        &rendering["terrain"],
        &json!({"tiles": above(0), "loading_tiles": 0, "fallback_tiles": 0})
    );
    Ok(rendering)
}

pub fn subdivided(session: &mut dyn SmokeSession) -> Checked<Value> {
    let (mut last, mut held) = (Value::Null, 0);
    let (steady, rendering) = renders_until(session, WAIT_FRAMES, |rendering| {
        let leaves = &rendering["terrain"]["leaves"];
        held = if *leaves == last { held + 1 } else { 0 };
        last = leaves.clone();
        held >= 16
    })?;
    smoke_check!(
        steady,
        "the_subdivision_settles",
        &rendering["terrain"]["leaves"],
        &json!({ "steady_frames": 16 })
    );
    Ok(rendering)
}

pub fn compiled(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut rendering = render(session)?;
    for _ in 0..COMPILE_FRAMES {
        if !compiling(&rendering) {
            return Ok(rendering);
        }
        rendering = render(session)?;
    }
    smoke_check!(
        !compiling(&rendering),
        "pipelines_compile",
        &rendering["pipelines_compiling"],
        &false
    );
    Ok(rendering)
}

pub fn compiling(rendering: &Value) -> bool {
    rendering["pipelines_compiling"] == true
}

pub fn shot(session: &mut dyn SmokeSession, body: Value) -> Checked<Value> {
    let captured = ask(session, "capture", body)?;
    let digest = captured["artifact"]["sha256"].clone();
    smoke_check!(
        digest.is_string(),
        "capture_keeps_its_frame",
        &captured["artifact"],
        &json!({"sha256": "string"})
    );
    Ok(digest)
}

pub fn capture(session: &mut dyn SmokeSession) -> Checked<Value> {
    compiled(session)?;
    shot(session, json!({}))
}

pub fn number(value: &Value) -> f64 {
    value.as_f64().unwrap_or(f64::NAN)
}

pub fn with(base: &Value, fields: Value) -> Value {
    let mut merged = base.clone();
    if let (Value::Object(merged), Value::Object(fields)) = (&mut merged, fields) {
        merged.extend(fields);
    }
    merged
}

pub fn above(value: impl Serialize) -> Value {
    json!({ "above": value })
}

pub fn below(value: impl Serialize) -> Value {
    json!({ "below": value })
}

pub fn at_most(value: impl Serialize) -> Value {
    json!({ "at_most": value })
}

pub fn near(value: impl Serialize, within: f64) -> Value {
    json!({"near": value, "within": within})
}

pub fn not(value: impl Serialize) -> Value {
    json!({ "not": value })
}

pub fn fast() -> Value {
    json!({ "samples_per_frame": FAST })
}

pub fn view(position: [f64; 3], target: [f64; 3]) -> Value {
    json!({
        "position": position,
        "target": target,
        "vertical_fov_radians": FRAC_PI_3,
        "near": 0.5,
        "far": 4000.0,
    })
}

pub fn overview() -> Value {
    view([0.0, 80.0, 140.0], [0.0, 30.0, 0.0])
}

pub fn open(session: &mut dyn SmokeSession) -> Checked<(Fixture, Value)> {
    let fixture = fixture(session.app().world_mut());
    control(session, "pause", json!({}))?;
    let opened = environment(session, json!({ "terrain": fast() }))?;
    smoke_eq!(
        &opened["environment_changed"],
        &json!(["terrain"]),
        "terrain_section_changes"
    );
    camera(session, overview())?;
    let settled = settle(session)?;
    Ok((fixture, settled))
}

#[derive(Default)]
pub(crate) struct Seen {
    pub terrain: Option<Terrain>,
    pub resident: usize,
    pub errors: Vec<f32>,
    pub ranges: Vec<f32>,
    pub bakes: TerrainBakes,
    pub paths: Vec<(usize, Option<PathStats>)>,
    pub surroundings: Option<SurroundingsStatus>,
}

pub(crate) fn seen(session: &mut dyn SmokeSession, entity: Entity) -> Seen {
    let world = session.app().world();
    let terrain = world.get::<Terrain>(entity).cloned();
    let Some(terrains) = world.get_resource::<Terrains>() else {
        return Seen {
            terrain,
            ..Seen::default()
        };
    };
    let (errors, ranges) = terrains
        .level_detail(entity)
        .map_or_else(Default::default, |(errors, ranges)| {
            (errors.to_vec(), ranges.to_vec())
        });
    let paths = terrains
        .layers(entity)
        .map(|layers| {
            layers
                .layers()
                .iter()
                .enumerate()
                .filter_map(|(index, layer)| {
                    if let Layer::Path(stage) = layer {
                        Some((index, stage.baked.as_ref().map(|profile| profile.stats)))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    Seen {
        terrain,
        resident: terrains.tiles(entity).count(),
        errors,
        ranges,
        bakes: terrains.bakes(entity),
        paths,
        surroundings: terrains.surroundings(entity),
    }
}
