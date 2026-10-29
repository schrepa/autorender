use engine::prelude::*;
use engine::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine::terrain::data::smokes::{Checked, compiled, control, render, settle};
use engine::smoke_check;
use highlands_pond::{Pond, PondBrick};
use serde_json::{Value, json};

const SURFACE: f32 = 2.0;
const POND_SIZE: f32 = 16.0;
const DROP: f32 = 6.0;
const FLOATS_AT: f32 = SURFACE - 0.1;
const FLOAT_WITHIN: f32 = 0.3;
const DEPTH_WITHIN: f32 = 1.0e-4;
const FLOAT_TICKS: u32 = 480;
const TICKS_AT_ONCE: u32 = 120;

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    session.restart(session.bricks().with::<PondBrick>());
    let (pond, body) = build(session.app().world_mut()).map_err(|error| {
        SmokeFailure::new(
            "the_pond_scene_builds",
            &error,
            &Value::Null,
            concat!(file!(), ":", line!()),
        )
    })?;
    control(session, "pause", json!({}))?;
    settle(session)?;
    holds_water(session)?;
    floats(session, body)?;
    draws(session, pond)
}

fn scene() -> Value {
    json!({
        "id": "pond-fixture",
        "revision": 1,
        "camera": {
            "position": [0, 14, 24],
            "target": [0, 1, 0],
            "fov_degrees": 55,
            "near": 0.1,
            "far": 1000,
        },
        "entities": [{
            "id": "ground",
            "terrain": {
                "origin": [-128, 0, -128],
                "tiles": [1, 1],
                "spacing": 1,
                "layers": [{ "type": "noise", "seed": 3, "height": 0, "wavelength": 120 }],
            },
        }],
    })
}

fn build(world: &mut World) -> engine::Result<(Entity, Entity)> {
    let document: SceneDocument = engine::scene::parse_json(&scene().to_string())?;
    let _scene = document.instantiate(world)?;
    let pond = world
        .spawn(Pond {
            center: Vec2::ZERO,
            size: Vec2::splat(POND_SIZE),
            surface: SURFACE,
        })
        .id();
    let body = world
        .spawn((
            Transform::from_xyz(0.0, DROP, 0.0),
            Collider::cuboid(Vec3::splat(0.5))?,
            RigidBody::default(),
        ))
        .id();
    Ok((pond, body))
}

fn holds_water(session: &mut dyn SmokeSession) -> Checked<()> {
    let world = session.app().world();
    let ground = Ground::sample_in(world, 0.0, 0.0).map(|point| point.height);
    let water = Ground::water_in(world, Vec3::new(0.0, SURFACE, 0.0));
    let expected = ground.map(|height| SURFACE - height);
    let held = water.is_some_and(|water| {
        water.kind == WaterKind::Lake
            && (water.surface - SURFACE).abs() <= DEPTH_WITHIN
            && expected.is_some_and(|depth| {
                depth > 0.0 && (water.depth - depth).abs() <= DEPTH_WITHIN
            })
    });
    smoke_check!(
        held,
        "the_pond_holds_water_down_to_the_ground",
        &water,
        &json!({"kind": "lake", "surface": SURFACE, "depth": expected})
    );
    Ok(())
}

fn floats(session: &mut dyn SmokeSession, body: Entity) -> Checked<()> {
    for _ in 0..FLOAT_TICKS / TICKS_AT_ONCE {
        session.step(TICKS_AT_ONCE);
    }
    let height = session
        .app()
        .world()
        .get::<Transform>(body)
        .map(|placed| placed.translation.y);
    smoke_check!(
        height.is_some_and(|height| (height - FLOATS_AT).abs() <= FLOAT_WITHIN),
        "the_body_floats_on_the_pond",
        &height,
        &json!({"near": FLOATS_AT, "within": FLOAT_WITHIN})
    );
    Ok(())
}

fn draws(session: &mut dyn SmokeSession, pond: Entity) -> SmokeResult {
    let shown = compiled(session)?;
    smoke_check!(
        drawn(&shown),
        "the_pond_pass_draws",
        &shown["render_passes"],
        &json!({"includes": "pond"})
    );
    session.app().world_mut().despawn(pond);
    render(session)?;
    let dry = render(session)?;
    smoke_check!(
        !drawn(&dry),
        "a_scene_without_ponds_skips_the_pass",
        &dry["render_passes"],
        &json!({"excludes": "pond"})
    );
    Ok(())
}

fn drawn(rendering: &Value) -> bool {
    rendering["render_passes"]
        .as_array()
        .is_some_and(|passes| passes.iter().any(|pass| pass == "pond"))
}
