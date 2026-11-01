use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession, compiled};
use engine_core::{BrickRegistry, CpuSlot, Slot, brick_name, smoke_check};
use engine_grounding::GroundingBrick;
use engine_places::GroundSlot;
use engine_render::graph::SCENE_CLEAR;
use serde_json::{Map, Value, json};

use super::engine_frames::{core_only, rendered};
use super::requests::{FRAME_MS, Pixels, accepted, pixels};

const FRAMES: u32 = 8;

const UNLINKED: [&str; 4] = [
    "shader_link",
    "face_missing",
    "slot_unfilled",
    "render_failed",
];

const TOLERANCE: u8 = 8;

pub(crate) fn flat_sprites(session: &mut dyn SmokeSession) -> SmokeResult {
    let base = session.bricks();
    let found = rendered(session);
    smoke_check!(
        !found.bricks.is_empty(),
        "render_bricks_listed",
        json!(found.bricks),
        json!("the render bricks of the list")
    );
    let left_out = grounds(session);
    session.restart(
        core_only(&base, &found)
            .without(left_out)
            .with::<SpriteBrick>(),
    );
    let rendering = presented(session)?;
    groundless(session)?;
    linked(&rendering)?;
    compiled(session)?;
    let reply = accepted(session, "capture", json!({}), "captures_sprite")?;
    if let Some(image) = pixels(&reply)? {
        shown(&image)?;
    }
    Ok(())
}

fn presented(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let pause = json!({ "name": "pause", "args": {} });
    let played = json!({ "name": "frames", "args": { "count": FRAMES, "interval_ms": FRAME_MS } });
    let reply = session
        .request("control", pause)
        .and_then(|_| session.request("control", played));
    let refused = reply
        .as_ref()
        .err()
        .map(|refused| refused["error"].clone())
        .unwrap_or_default();
    smoke_check!(
        !UNLINKED.iter().any(|code| refused["code"] == *code),
        "sprite_links_alone",
        refused,
        json!({ "none_of": UNLINKED })
    );
    smoke_check!(
        reply.is_ok(),
        "frames_present",
        refused,
        json!({ "frames": FRAMES })
    );
    Ok(reply
        .map(|answered| answered["result"]["rendering"].clone())
        .unwrap_or_default())
}

fn grounds(session: &mut dyn SmokeSession) -> Vec<&'static str> {
    let filler = session
        .app()
        .world()
        .get_resource::<BrickRegistry>()
        .and_then(|registry| registry.filled(GroundSlot::NAME))
        .map(|(brick, _)| brick);
    filler
        .into_iter()
        .chain([brick_name::<GroundingBrick>()])
        .collect()
}

fn groundless(session: &mut dyn SmokeSession) -> SmokeResult {
    let world = session.app().world();
    let registry = world.get_resource::<BrickRegistry>();
    let ground = registry
        .and_then(|registry| registry.filled(GroundSlot::NAME))
        .map(|(brick, _)| brick);
    let grounding = brick_name::<GroundingBrick>();
    let grounded = registry.is_some_and(|registry| registry.names().any(|name| name == grounding));
    let held = world.contains_resource::<<GroundSlot as CpuSlot>::Cpu>();
    smoke_check!(
        ground.is_none() && !grounded && !held,
        "installs_no_ground",
        json!({ "ground": ground, "grounding": grounded, "resource": held }),
        json!({ "ground": null, "grounding": false, "resource": false })
    );
    Ok(())
}

fn members(lists: &Value) -> usize {
    lists
        .as_object()
        .into_iter()
        .flat_map(Map::values)
        .filter_map(Value::as_array)
        .map(Vec::len)
        .sum()
}

fn linked(rendering: &Value) -> SmokeResult {
    let linked = &rendering["linked"];
    let lists = &linked["lists"];
    smoke_check!(
        linked["faces"].as_object().is_some_and(Map::is_empty)
            && linked["nobody"].as_array().is_some_and(Vec::is_empty)
            && lists.is_object()
            && members(lists) == 0,
        "links_no_face",
        linked,
        json!({ "faces": {}, "nobody": [], "lists": "no members" })
    );
    Ok(())
}

fn shown(image: &Pixels) -> SmokeResult {
    let center = color_at(image, image.width / 2, image.height / 2);
    smoke_check!(
        [LIGHT, DARK]
            .iter()
            .any(|texel| near(center, [texel[0], texel[1], texel[2]])),
        "sprite_drawn_in_the_center",
        json!(center),
        json!({ "one_of": [LIGHT, DARK] })
    );
    let corner = color_at(image, 0, 0);
    let clear = cleared();
    smoke_check!(
        near(corner, clear),
        "clear_color_in_the_corner",
        json!(corner),
        json!(clear)
    );
    Ok(())
}

fn color_at(image: &Pixels, x: u32, y: u32) -> [u8; 3] {
    let at = (y as usize * image.width as usize + x as usize) * 4;
    image
        .rgba
        .get(at..at + 3)
        .map_or([0; 3], |rgb| [rgb[0], rgb[1], rgb[2]])
}

fn near(got: [u8; 3], expected: [u8; 3]) -> bool {
    got.iter()
        .zip(expected)
        .all(|(&got, expected)| got.abs_diff(expected) <= TOLERANCE)
}

fn cleared() -> [u8; 3] {
    let [red, green, blue, _] = SCENE_CLEAR;
    [red, green, blue].map(encoded)
}

fn encoded(linear: f64) -> u8 {
    let srgb = if linear <= 0.003_130_8 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round() as u8
}
