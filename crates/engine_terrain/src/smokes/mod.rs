mod draws;
mod fixture;
mod layers;
mod shapes;
mod surface;
mod water_sky;

use std::f64::consts::FRAC_PI_3;

use bevy_ecs::prelude::Entity;
use engine_core::smoke::{
    COMPILE_FRAMES, Checked, SmokeSession, above, ask, at_most, below, camera, capture, compiling,
    component, control, environment, inspect, near, not, number, present, query, refuse, render,
    renders_until, shot, with,
};
use engine_core::{Bricks, smoke_check, smoke_eq};
use glam::Vec2;
use serde_json::{Value, json};

use crate::layer::maps;
use crate::{SurroundingsStatus, Terrain, TerrainBakes, Terrains};

pub use fixture::{FIXTURE, Fixture, fixture, name};
pub(crate) use layers::layers;

pub const FAST: u32 = 50_000_000;
pub const WAIT_FRAMES: u32 = 600;
pub const SETTLE_FRAMES: u32 = 2400;

pub(crate) fn register(bricks: &mut Bricks<'_>) {
    bricks
        .smoke("terrain_draws", true, draws::run)
        .smoke("terrain_shapes", true, shapes::run)
        .smoke("terrain_surface", true, surface::run)
        .smoke("terrain_water_sky", true, water_sky::run);
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
        let leaves = &rendering["terrain"]["leaf_cut"]["kept"];
        held = if *leaves == last { held + 1 } else { 0 };
        last = leaves.clone();
        held >= 16
    })?;
    smoke_check!(
        steady,
        "the_subdivision_settles",
        &rendering["terrain"]["leaf_cut"]["kept"],
        &json!({ "steady_frames": 16 })
    );
    Ok(rendering)
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
    pub paths: Vec<(usize, Option<Value>)>,
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
                .filter(|(_, layer)| layer.outputs().iter().any(|name| name == maps::PATH))
                .map(|(index, layer)| (index, layer.stats()))
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

pub(crate) fn layer_sweep(
    session: &mut dyn SmokeSession,
    center: Vec2,
    half_extent: f32,
    samples: u32,
) -> Value {
    let world = session.app().world();
    let Some(terrains) = world.get_resource::<Terrains>() else {
        return Value::Null;
    };
    let (mut compared, mut worst, mut at) = (0_usize, 0.0_f32, None);
    for row in 0..samples {
        for column in 0..samples {
            let unit = Vec2::new(column as f32, row as f32) / (samples - 1) as f32 * 2.0 - 1.0;
            let point = center + unit * half_extent;
            let Some(sample) = terrains.sample(point.x, point.y) else {
                continue;
            };
            if sample.hole || sample.level != 0 {
                continue;
            }
            let (Some(terrain), Some(stack)) = (
                world.get::<Terrain>(sample.terrain),
                terrains.layers(sample.terrain),
            ) else {
                continue;
            };
            let local = point - Vec2::new(terrain.origin.x, terrain.origin.z);
            let height = terrain.origin.y + stack.height_at(&terrain.pyramid, local);
            let difference = (height - sample.height).abs();
            compared += 1;
            if difference > worst || at.is_none() {
                worst = engine_math::max(worst, difference);
                at = Some(point.to_array());
            }
        }
    }
    json!({"compared": compared, "worst": {"max_difference": worst, "at": at}})
}
