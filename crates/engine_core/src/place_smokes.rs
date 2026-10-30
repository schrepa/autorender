use bevy_ecs::prelude::World;
use glam::Vec3;
use serde_json::{Value, json};

use crate::smoke::{SmokeResult, SmokeSession, control, render};
use crate::{
    BrickList, BrickRegistry, Ground, GroundPoint, GroundSlot, List, Slot, TILE_QUADS, WaterBodies,
    WaterBodiesList, WaterFace,
};

const TILES: u32 = 2;

const SPACING: f32 = 1.0;

const HALF: f32 = (TILES * TILE_QUADS) as f32 * SPACING * 0.5;

const OUTSIDE: f32 = 3.0 * HALF;

const STEPS: u32 = 3;

const WAIT_FRAMES: u32 = 2400;

const HELD_FRAMES: u32 = 30;

const UNIT_WITHIN: f32 = 1.0e-3;

const GRID: u16 = 16;

const GRID_STEP: f32 = 2.0;

const ABOVE_GROUND: f32 = 0.05;

fn members_of(session: &mut dyn SmokeSession, place: &str) -> Vec<&'static str> {
    session
        .app()
        .world()
        .get_resource::<BrickRegistry>()
        .map(|registry| registry.members(place))
        .unwrap_or_default()
}

fn scene(terrain: bool) -> String {
    let mut entities = vec![json!({
        "id": "floor", "mesh": "box", "material": "plain",
        "transform": { "translation": [0, -0.5, 0], "scale": [8, 1, 8] },
    })];
    if terrain {
        entities.push(json!({
            "id": "ground",
            "terrain": {
                "origin": [-HALF, 0, -HALF], "tiles": [TILES, TILES], "spacing": SPACING,
                "layers": [{ "type": "noise", "seed": 3, "height": 8, "wavelength": 120 }],
            },
        }));
    }
    json!({
        "id": "place-ground", "revision": 1,
        "camera": { "position": [0, 60, 80], "target": [0, 0, 0], "fov_degrees": 60, "near": 0.1, "far": 2000 },
        "materials": [{ "id": "plain", "color": [0.6, 0.6, 0.6, 1] }],
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": entities,
    })
    .to_string()
}

fn opened(session: &mut dyn SmokeSession, document: &str, bricks: BrickList) -> SmokeResult {
    session.source(document);
    session.restart(bricks);
    control(session, "pause", json!({}))?;
    session.step(STEPS);
    render(session)?;
    Ok(())
}

fn seen(point: GroundPoint) -> Value {
    json!({
        "height": point.height,
        "normal": point.normal.to_array(),
        "level": point.level,
        "hole": point.hole,
    })
}

fn resident(session: &mut dyn SmokeSession) -> SmokeResult {
    let (mut held, mut last, mut centre) = (0, None, None);
    for _ in 0..WAIT_FRAMES {
        if held >= HELD_FRAMES {
            break;
        }
        render(session)?;
        let world = session.app().world();
        let revision = Ground::revision_in(world);
        centre = Ground::point_in(world, 0.0, 0.0);
        held = if centre.is_some() && last == Some(revision) {
            held + 1
        } else {
            0
        };
        last = Some(revision);
    }
    crate::smoke_check!(
        held >= HELD_FRAMES,
        "the_ground_becomes_resident",
        &json!({ "held_frames": held, "answered": centre.is_some() }),
        &json!({ "held_frames": HELD_FRAMES, "answered": true })
    );
    let answered = centre.is_some_and(|point| {
        point.height.is_finite() && (point.normal.length() - 1.0).abs() <= UNIT_WITHIN
    });
    crate::smoke_check!(
        answered,
        "the_ground_answers_its_centre",
        &centre.map(seen),
        &json!({ "height": "finite", "normal": "unit" })
    );
    Ok(())
}

fn steady(session: &mut dyn SmokeSession) -> SmokeResult {
    let beyond = Ground::point_in(session.app().world(), OUTSIDE, OUTSIDE);
    crate::smoke_check!(
        beyond.is_none(),
        "the_ground_answers_nothing_outside",
        &beyond.map(seen),
        &Value::Null
    );
    let before = Ground::revision_in(session.app().world());
    render(session)?;
    render(session)?;
    let after = Ground::revision_in(session.app().world());
    crate::smoke_eq!(after, before, "the_ground_revision_holds_without_edits");
    Ok(())
}

pub(crate) fn ground(session: &mut dyn SmokeSession) -> SmokeResult {
    let fillers = members_of(session, GroundSlot::NAME);
    if fillers.is_empty() {
        return Ok(());
    }
    let original = session.bricks();
    opened(session, &scene(true), original.clone())?;
    let answering = session
        .app()
        .world()
        .get_resource::<Ground>()
        .map(Ground::brick);
    crate::smoke_check!(
        answering.is_some_and(|brick| fillers.contains(&brick)),
        "the_filler_answers_the_ground",
        &answering,
        &fillers
    );
    if session.gpu() {
        resident(session)?;
    }
    steady(session)?;
    let left_out = original.clone().without(fillers.iter().copied());
    opened(session, &scene(false), left_out)?;
    let absent = session.app().world().get_resource::<Ground>().is_none();
    crate::smoke_check!(absent, "the_ground_leaves_with_its_filler", &absent, &true);
    session.restart(original);
    control(session, "pause", json!({}))?;
    Ok(())
}

fn grid() -> impl Iterator<Item = (f32, f32)> {
    let at = |index: u16| (f32::from(index) - (f32::from(GRID) - 1.0) * 0.5) * GRID_STEP;
    (0..GRID).flat_map(move |row| (0..GRID).map(move |column| (at(column), at(row))))
}

fn sampled(world: &World, body: &'static str, face: &dyn WaterFace) -> SmokeResult {
    for (x, z) in grid() {
        let ground = Ground::point_in(world, x, z).map(|point| point.height);
        let at = Vec3::new(x, ground.unwrap_or(0.0), z);
        let Some(water) = face.water(world, at) else {
            continue;
        };
        let found = json!({ "body": body, "at": [x, z], "water": water, "ground": ground });
        let dry_or_deep = water.depth >= 0.0;
        crate::smoke_check!(
            dry_or_deep,
            "water_depth_is_not_negative",
            &found,
            &json!({ "depth": { "at_least": 0.0 } })
        );
        crate::smoke_check!(
            water.surface.is_finite(),
            "water_surface_is_finite",
            &found,
            &json!({ "surface": "finite" })
        );
        crate::smoke_check!(
            ground.is_none_or(|height| water.surface >= height - ABOVE_GROUND),
            "water_lies_on_the_ground",
            &found,
            &json!({ "surface": { "at_least": ground.map(|height| height - ABOVE_GROUND) } })
        );
    }
    Ok(())
}

fn reached(world: &World, body: &'static str, face: &dyn WaterFace) -> SmokeResult {
    for (low, high) in face.reaches(world) {
        crate::smoke_check!(
            low.is_finite() && high.is_finite() && low.cmple(high).all(),
            "water_reaches_are_ordered_boxes",
            &json!({ "body": body, "low": low.to_array(), "high": high.to_array() }),
            &json!({ "low": "finite, at most high", "high": "finite" })
        );
    }
    Ok(())
}

pub(crate) fn water_bodies(session: &mut dyn SmokeSession) -> SmokeResult {
    let members = members_of(session, WaterBodiesList::NAME);
    if members.is_empty() {
        return Ok(());
    }
    control(session, "pause", json!({}))?;
    session.step(STEPS);
    render(session)?;
    let world = session.app().world();
    let answering: Vec<&'static str> = world
        .get_resource::<WaterBodies>()
        .map(|bodies| bodies.bodies().map(|(brick, _)| brick).collect())
        .unwrap_or_default();
    crate::smoke_check!(
        members.iter().all(|brick| answering.contains(brick)),
        "every_member_is_a_water_body",
        &answering,
        &members
    );
    let Some(bodies) = world.get_resource::<WaterBodies>() else {
        return Ok(());
    };
    for (body, face) in bodies.bodies() {
        sampled(world, body, face)?;
        reached(world, body, face)?;
    }
    Ok(())
}
