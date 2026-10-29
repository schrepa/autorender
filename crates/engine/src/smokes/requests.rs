use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_scene::SceneDocument;
use serde_json::{Map, Value, json};

use crate::camera::FlyCameraController;

pub(crate) type Asked = Result<Value, SmokeFailure>;

pub(crate) const FRAME_MS: f64 = 1000.0 / 60.0;

pub(crate) fn reply(session: &mut dyn SmokeSession, verb: &str, body: Value) -> Value {
    match session.request(verb, body) {
        Ok(reply) | Err(reply) => reply,
    }
}

pub(crate) fn accepted(
    session: &mut dyn SmokeSession,
    verb: &str,
    body: Value,
    check: &'static str,
) -> Asked {
    let reply = reply(session, verb, body);
    engine_core::smoke_check!(
        reply["ok"] == true,
        check,
        reply["error"].clone(),
        json!({ "ok": true })
    );
    Ok(reply)
}

pub(crate) fn refused(
    session: &mut dyn SmokeSession,
    verb: &str,
    body: Value,
    check: &'static str,
) -> Asked {
    let reply = reply(session, verb, body);
    engine_core::smoke_check!(
        reply["ok"] == false,
        check,
        reply["result"].clone(),
        json!({ "ok": false })
    );
    Ok(reply)
}

pub(crate) fn refused_with(
    session: &mut dyn SmokeSession,
    verb: &str,
    body: Value,
    code: &str,
    check: &'static str,
) -> Asked {
    let reply = reply(session, verb, body);
    engine_core::smoke_check!(
        reply["error"]["code"] == code,
        check,
        reply["error"].clone(),
        json!({ "code": code })
    );
    Ok(reply)
}

pub(crate) fn named(name: &str, args: Value) -> Value {
    json!({ "name": name, "args": args })
}

pub(crate) fn control(
    session: &mut dyn SmokeSession,
    name: &str,
    args: Value,
    check: &'static str,
) -> Asked {
    accepted(session, "control", named(name, args), check)
}

pub(crate) fn query(
    session: &mut dyn SmokeSession,
    name: &str,
    args: Value,
    check: &'static str,
) -> Asked {
    accepted(session, "query", named(name, args), check)
}

pub(crate) fn status(session: &mut dyn SmokeSession, check: &'static str) -> Asked {
    query(session, "status", json!({}), check)
}

pub(crate) fn inspected(
    session: &mut dyn SmokeSession,
    component: &str,
    check: &'static str,
) -> Asked {
    let reply = accepted(session, "inspect", json!({ "component": component }), check)?;
    Ok(reply["result"]["value"].clone())
}

pub(crate) fn component(
    session: &mut dyn SmokeSession,
    entity: &str,
    component: &str,
    check: &'static str,
) -> Asked {
    let body = json!({ "entity": entity, "component": component });
    let reply = accepted(session, "inspect", body, check)?;
    Ok(reply["result"]["entity"]["components"][component].clone())
}

pub(crate) fn environment(
    session: &mut dyn SmokeSession,
    args: Value,
    check: &'static str,
) -> Asked {
    control(session, "environment", args, check)
}

pub(crate) fn frames(session: &mut dyn SmokeSession, count: u32, check: &'static str) -> Asked {
    control(
        session,
        "frames",
        json!({ "count": count, "interval_ms": FRAME_MS }),
        check,
    )
}

pub(crate) fn counter(reply: &Value, field: &str) -> u64 {
    let observation = &reply["observation"];
    let observation = if observation.is_object() {
        observation
    } else {
        &reply["result"]["observation"]
    };
    observation[field]
        .as_str()
        .and_then(|text| text.parse().ok())
        .unwrap_or(u64::MAX)
}

pub(crate) fn tick(reply: &Value) -> u64 {
    counter(reply, "tick")
}

pub(crate) fn frame(reply: &Value) -> u64 {
    counter(reply, "frame")
}

pub(crate) fn number(value: &Value) -> f64 {
    value.as_f64().unwrap_or(f64::NAN)
}

pub(crate) fn names(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

pub(crate) fn sorted_keys(value: &Value) -> Vec<String> {
    let mut keys: Vec<String> = value
        .as_object()
        .map(|object| object.keys().cloned().collect())
        .unwrap_or_default();
    keys.sort();
    keys
}

pub(crate) fn idle(milliseconds: f64) {
    let until = crate::host::now_ms() + milliseconds;
    while crate::host::now_ms() < until {
        std::hint::spin_loop();
    }
}

pub(crate) fn serve(session: &mut dyn SmokeSession, document: &Value, without: &[&str]) -> SmokeResult {
    session.source(&document.to_string());
    session.restart(without);
    control(session, "pause", json!({}), "pauses")?;
    Ok(())
}

pub(crate) fn spawned(session: &mut dyn SmokeSession, document: &Value) -> Result<(), Value> {
    let text = document.to_string();
    let scene = SceneDocument::from_json(&text)
        .map_err(|error| json!({ "code": error.code, "path": error.path }))?;
    scene
        .instantiate(session.app().world_mut())
        .map(|_| ())
        .map_err(|error| json!({ "code": error.code, "path": error.path }))
}

pub(crate) fn spawn(session: &mut dyn SmokeSession, document: &Value) -> SmokeResult {
    let refusal = spawned(session, document).err();
    engine_core::smoke_check!(
        refusal.is_none(),
        "scene_instantiates",
        refusal.clone().unwrap_or_default(),
        json!(null)
    );
    let world = session.app().world_mut();
    let camera = *world.resource::<engine_core::Camera>();
    if let Some(mut controller) = world.get_resource_mut::<FlyCameraController>() {
        controller.align_to(&camera);
    }
    Ok(())
}

pub(crate) fn grid(side: usize, moving: bool) -> Value {
    let paints = [
        [0.85, 0.35, 0.2, 1.0],
        [0.25, 0.6, 0.85, 1.0],
        [0.4, 0.75, 0.3, 1.0],
        [0.9, 0.75, 0.25, 1.0],
    ];
    let cube = json!({ "kind": "cuboid", "half_extents": [0.5, 0.5, 0.5] });
    let mut ground = json!({
        "id": "ground", "mesh": "box", "material": "ground",
        "transform": { "translation": [0.0, -0.5, 0.0], "scale": [side as f64 * 6.0, 1.0, side as f64 * 6.0] },
    });
    if moving {
        ground["collider"] = json!({ "shape": cube });
    }
    let mut entities = vec![ground];
    let middle = (side as f64 - 1.0) / 2.0;
    for index in 0..side * side {
        let (row, column) = ((index / side) as f64, (index % side) as f64);
        let scale = 0.8 + (index % 5) as f64 * 0.3;
        let mut piece = json!({
            "id": format!("piece-{index}"),
            "mesh": if index % 3 == 0 { "gem" } else { "box" },
            "material": format!("paint-{}", index % paints.len()),
            "transform": {
                "translation": [(column - middle) * 4.0, scale / 2.0, (row - middle) * 4.0],
                "rotation_degrees": [0, (index * 37) % 360, 0],
                "scale": [scale, scale, scale],
            },
        });
        if moving {
            piece["collider"] = json!({ "shape": cube });
            piece["rigid_body"] = json!({
                "velocity": [0, 4 + index % 7, 0],
                "spin": [0, 1 + index % 3, 0],
            });
        }
        entities.push(piece);
        if index % 4 == 0 {
            entities.push(json!({
                "id": format!("cap-{index}"), "parent": format!("piece-{index}"), "mesh": "box",
                "material": format!("paint-{}", (index + 1) % paints.len()),
                "transform": { "translation": [0.0, 0.75, 0.0], "scale": [0.5, 0.5, 0.5] },
            }));
        }
    }
    let mut materials = vec![json!({ "id": "ground", "color": [0.55, 0.58, 0.52, 1.0], "texture": "checker" })];
    materials.extend(paints.iter().enumerate().map(|(index, color)| {
        json!({ "id": format!("paint-{index}"), "color": color, "texture": "checker" })
    }));
    json!({
        "id": "grid", "revision": 1,
        "camera": { "position": [0.0, 30.0, side as f64 * 2.6], "target": [0.0, 0.0, 0.0], "fov_degrees": 60, "near": 0.1, "far": 1000 },
        "textures": [{ "id": "checker", "width": 32, "height": 32, "rgba8": base64(&checker()) }],
        "materials": materials,
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }, { "id": "gem", "source": { "kind": "octahedron" } }],
        "entities": entities,
    })
}

fn checker() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(32 * 32 * 4);
    for y in 0..32_u32 {
        for x in 0..32_u32 {
            let shade = 150 + (x * 7 + y * 13) % 48;
            let light = ((x >> 2) + (y >> 2)) % 2 == 0;
            let rgb = if light {
                [shade + 40, shade + 30, shade]
            } else {
                [shade - 80, shade - 70, shade - 50]
            };
            pixels.extend(rgb.map(|channel| channel as u8));
            pixels.push(255);
        }
    }
    pixels
}

pub(crate) fn base64(bytes: &[u8]) -> String {
    const LETTERS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let joined = chunk
            .iter()
            .enumerate()
            .fold(0_u32, |joined, (at, &byte)| {
                joined | (u32::from(byte) << (16 - 8 * at))
            });
        for at in 0..4 {
            if at <= chunk.len() {
                text.push(char::from(LETTERS[((joined >> (18 - 6 * at)) & 63) as usize]));
            } else {
                text.push('=');
            }
        }
    }
    text
}

pub(crate) struct Pixels {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) rgba: Vec<u8>,
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn pixels(reply: &Value) -> Result<Option<Pixels>, SmokeFailure> {
    let artifact = &reply["result"]["artifact"];
    let path = artifact["path"].as_str().unwrap_or_default();
    let bytes = std::fs::read(path).unwrap_or_default();
    engine_core::smoke_check!(
        bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]),
        "capture_is_png",
        artifact.clone(),
        json!({ "mime_type": "image/png" })
    );
    engine_core::smoke_eq!(
        json!(bytes.len()),
        artifact["bytes"].clone(),
        "capture_bytes_match"
    );
    let image = crate::host::decode_image(path, &bytes);
    engine_core::smoke_check!(
        image.is_ok(),
        "capture_decodes",
        json!(image.as_ref().err()),
        json!(null)
    );
    Ok(image.ok().map(|image| Pixels {
        width: image.width,
        height: image.height,
        rgba: image.pixels,
    }))
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn pixels(_: &Value) -> Result<Option<Pixels>, SmokeFailure> {
    Ok(None)
}

pub(crate) fn differing(a: &Pixels, b: &Pixels, tolerance: u8) -> Result<usize, SmokeFailure> {
    engine_core::smoke_eq!(
        json!([b.width, b.height]),
        json!([a.width, a.height]),
        "capture_size_kept"
    );
    Ok(a
        .rgba
        .chunks_exact(4)
        .zip(b.rgba.chunks_exact(4))
        .filter(|(a, b)| (0..3).any(|channel| a[channel].abs_diff(b[channel]) > tolerance))
        .count())
}

pub(crate) fn object(value: &Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}
