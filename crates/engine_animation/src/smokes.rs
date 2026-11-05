use std::f32::consts::FRAC_1_SQRT_2;

use engine_core::smoke::{
    Checked, above, ask, compiled, component, control, not, number, query, render, with,
};
use engine_core::{SmokeFailure, SmokeSession, smoke_check, smoke_eq};
use serde_json::{Value, json};

const STILL: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const TURNED: [f32; 4] = [0.0, 0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2];
const CORNERS: [[f32; 2]; 4] = [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]];
const RINGS: [(f32, u32); 5] = [(0.0, 0), (0.5, 0), (1.0, 1), (1.5, 1), (2.0, 2)];
const CLOSE: f64 = 0.02;
const WAIT_FRAMES: u32 = 600;
const SETTLE_FRAMES: u32 = 8;
const CHANNEL_TOLERANCE: u8 = 24;
const POSE_PIXELS: usize = 200;

fn limb() -> Value {
    let vertices: Vec<Value> = RINGS
        .iter()
        .flat_map(|&(height, _)| {
            CORNERS.iter().map(move |&[x, z]| {
                json!({
                    "position": [x, height, z],
                    "normal": [x * 10.0, 0.0, z * 10.0],
                    "uv": [0.0, 0.0],
                })
            })
        })
        .collect();
    let side = CORNERS.len() as u32;
    let mut indices = Vec::new();
    for ring in 0..RINGS.len() as u32 - 1 {
        for corner in 0..side {
            let next = (corner + 1) % side;
            let (a, b) = (ring * side + corner, ring * side + next);
            let (c, d) = ((ring + 1) * side + next, (ring + 1) * side + corner);
            indices.extend([a, c, b, a, d, c]);
        }
    }
    json!({ "kind": "inline", "vertices": vertices, "indices": indices })
}

fn bound(joint: u32, down: f32) -> Value {
    json!({
        "joint": joint,
        "inverse_bind": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, down, 0.0, 1.0],
        ],
    })
}

fn held_rotation(name: &str, rotation: [f32; 4]) -> Value {
    json!({
        "name": name,
        "duration": 1.0,
        "tracks": [{ "joint": 1, "rotation": { "times": [0.0], "values": [rotation] } }],
    })
}

fn held_translation(joint: u32, translation: [f32; 3]) -> Value {
    json!({ "joint": joint, "translation": { "times": [0.0], "values": [translation] } })
}

fn rig() -> Value {
    let skin: Vec<Value> = RINGS
        .iter()
        .flat_map(|&(_, joint)| {
            CORNERS.iter().map(
                move |_| json!({ "joints": [joint, 0, 0, 0], "weights": [1.0, 0.0, 0.0, 0.0] }),
            )
        })
        .collect();
    json!({
        "joints": [
            { "name": "base" },
            { "name": "mid", "parent": 0, "translation": [0.0, 1.0, 0.0] },
            { "name": "tip", "parent": 1, "translation": [0.0, 1.0, 0.0] },
        ],
        "palette": [bound(0, 0.0), bound(1, -1.0), bound(2, -2.0)],
        "clips": [
            {
                "name": "bend",
                "duration": 1.0,
                "tracks": [{
                    "joint": 1,
                    "rotation": { "times": [0.0, 1.0], "values": [STILL, TURNED] },
                }],
            },
            held_rotation("bent", TURNED),
            held_rotation("straight", STILL),
            {
                "name": "reach",
                "duration": 1.0,
                "tracks": [
                    held_translation(1, [0.5, 1.0, 0.0]),
                    held_translation(2, [0.0, 1.0, 0.5]),
                ],
            },
        ],
        "parts": [{ "mesh": "limb", "material": "paint", "skin": skin }],
    })
}

fn scene() -> Value {
    json!({
        "id": "animation-plays",
        "revision": 1,
        "camera": {
            "position": [0.0, 1.0, 6.0],
            "target": [0.0, 1.0, 0.0],
            "fov_degrees": 50,
            "near": 0.1,
            "far": 100.0,
        },
        "materials": [{
            "id": "paint",
            "color": [0.8, 0.2, 0.2, 1.0],
            "emissive": [1.0, 0.2, 0.2],
        }],
        "meshes": [{ "id": "limb", "source": limb() }],
        "entities": [
            { "id": "bar", "animator": { "rig": rig(), "play": { "clip": "bend" } } },
            { "id": "gun", "parent": "bar", "attachment": { "joint": "tip" } },
        ],
    })
}

fn state(session: &mut dyn SmokeSession) -> Checked<Value> {
    query(session, "animation_state", json!({ "entity": "bar" }))
}

fn order(session: &mut dyn SmokeSession, name: &str, fields: Value) -> Checked<Value> {
    control(session, name, with(&json!({ "entity": "bar" }), fields))
}

fn vector(value: &Value) -> [f64; 3] {
    [number(&value[0]), number(&value[1]), number(&value[2])]
}

fn close(got: [f64; 3], wanted: [f64; 3]) -> bool {
    got.iter()
        .zip(&wanted)
        .all(|(got, wanted)| (got - wanted).abs() < CLOSE)
}

fn joint(session: &mut dyn SmokeSession, name: &str) -> Checked<[f64; 3]> {
    let held = state(session)?;
    Ok(vector(&held["joints"][name]))
}

fn at(
    session: &mut dyn SmokeSession,
    name: &str,
    wanted: [f64; 3],
    check: &'static str,
) -> Checked<()> {
    let got = joint(session, name)?;
    smoke_check!(close(got, wanted), check, &got, &wanted);
    Ok(())
}

fn moves(session: &mut dyn SmokeSession) -> Checked<()> {
    let start = joint(session, "tip")?;
    session.step(30);
    let moved = joint(session, "tip")?;
    smoke_check!(
        !close(moved, start),
        "clip_moves_joint",
        &moved,
        &not(start)
    );
    Ok(())
}

fn crossfades(session: &mut dyn SmokeSession) -> Checked<()> {
    order(session, "animation_play", json!({ "clip": "bent" }))?;
    session.step(2);
    at(session, "tip", [-1.0, 1.0, 0.0], "cut_poses_joint")?;
    order(
        session,
        "animation_play",
        json!({ "clip": "straight", "seconds": 1.0 }),
    )?;
    session.step(30);
    let middle = joint(session, "tip")?;
    smoke_check!(
        middle[0] > -0.95 && middle[0] < -0.05,
        "crossfade_blends",
        &middle,
        &json!({ "x_between": [-0.95, -0.05] })
    );
    session.step(40);
    at(session, "tip", [0.0, 2.0, 0.0], "crossfade_lands")
}

fn masks(session: &mut dyn SmokeSession) -> Checked<()> {
    let added = order(session, "animation_layer", json!({ "root": "tip" }))?;
    smoke_eq!(added["layer"], json!(1), "layer_added");
    order(
        session,
        "animation_play",
        json!({ "clip": "reach", "layer": 1 }),
    )?;
    session.step(2);
    at(session, "mid", [0.0, 1.0, 0.0], "mask_spares_parent")?;
    at(session, "tip", [0.0, 2.0, 0.5], "mask_moves_subtree")
}

fn aims(session: &mut dyn SmokeSession) -> Checked<()> {
    order(
        session,
        "animation_aim",
        json!({ "joint": "mid", "rotation": TURNED }),
    )?;
    session.step(1);
    at(session, "tip", [-1.0, 1.0, 0.5], "aim_overrides_clip")?;
    let tip = joint(session, "tip")?;
    let gun = component(session, "gun", "Transform")?;
    let held = vector(&gun["translation"]);
    smoke_check!(close(held, tip), "attachment_follows", &held, &tip);
    order(session, "animation_aim", json!({ "joint": "mid" }))?;
    session.step(1);
    at(session, "tip", [0.0, 2.0, 0.5], "aim_clears")
}

fn finishes(session: &mut dyn SmokeSession) -> Checked<()> {
    order(
        session,
        "animation_play",
        json!({ "clip": "bend", "looped": false }),
    )?;
    session.step(10);
    let running = state(session)?;
    smoke_eq!(
        running["layers"][0]["finished"],
        json!(false),
        "one_shot_runs"
    );
    session.step(80);
    let done = state(session)?;
    smoke_eq!(
        done["layers"][0]["finished"],
        json!(true),
        "one_shot_finishes"
    );
    let duration = done["clips"]
        .as_array()
        .and_then(|clips| clips.iter().find(|clip| clip["name"] == "bend"))
        .map_or(f64::NAN, |clip| number(&clip["duration"]));
    smoke_check!(
        (duration - 1.0).abs() < CLOSE,
        "clip_duration",
        &duration,
        &1.0
    );
    Ok(())
}

fn resident(session: &mut dyn SmokeSession) -> Checked<()> {
    for _ in 0..WAIT_FRAMES {
        let held = state(session)?;
        if held["resident"] == held["parts"] {
            return Ok(());
        }
        render(session)?;
    }
    let held = state(session)?;
    smoke_eq!(held["resident"], held["parts"], "parts_resident");
    Ok(())
}

fn shown(session: &mut dyn SmokeSession) -> Checked<Option<Vec<u8>>> {
    compiled(session)?;
    for _ in 0..SETTLE_FRAMES {
        render(session)?;
    }
    let captured = ask(session, "capture", json!({}))?;
    let path = captured["artifact"]["path"].as_str().unwrap_or_default();
    let Some(bytes) = session.artifact(path) else {
        return Ok(None);
    };
    let decoded = session.decode_image(path, &bytes);
    smoke_check!(
        decoded.is_ok(),
        "capture_decodes",
        &decoded.as_ref().err(),
        &"image"
    );
    Ok(decoded.ok().map(|(_, _, pixels)| pixels))
}

fn draws(session: &mut dyn SmokeSession) -> Checked<()> {
    if !session.frame() {
        return Ok(());
    }
    resident(session)?;
    let Some(first) = shown(session)? else {
        return Ok(());
    };
    order(
        session,
        "animation_aim",
        json!({ "joint": "mid", "rotation": STILL }),
    )?;
    session.step(1);
    let Some(second) = shown(session)? else {
        return Ok(());
    };
    let changed = first
        .as_chunks::<4>()
        .0
        .iter()
        .zip(second.as_chunks::<4>().0)
        .filter(|(a, b)| (0..3).any(|at| a[at].abs_diff(b[at]) > CHANNEL_TOLERANCE))
        .count();
    smoke_check!(
        changed > POSE_PIXELS,
        "pose_changes_pixels",
        &changed,
        &above(POSE_PIXELS)
    );
    Ok(())
}

pub(crate) fn plays(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene().to_string());
    control(session, "pause", json!({}))?;
    moves(session)?;
    crossfades(session)?;
    masks(session)?;
    aims(session)?;
    finishes(session)?;
    draws(session)
}
