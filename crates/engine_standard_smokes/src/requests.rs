use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

pub(crate) type Asked = Result<Value, SmokeFailure>;

pub(crate) const FRAME_MS: f64 = 1000.0 / 60.0;

pub(crate) fn seen(held: bool, got: Value, expected: &Value) -> Value {
    if held { expected.clone() } else { got }
}

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
    let wanted = json!({ "ok": true });
    smoke_eq!(
        seen(reply["ok"] == true, reply["error"].clone(), &wanted),
        wanted,
        check
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
    let wanted = json!({ "code": code });
    smoke_eq!(
        seen(reply["error"]["code"] == code, reply["error"].clone(), &wanted),
        wanted,
        check
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

pub(crate) fn frames(session: &mut dyn SmokeSession, count: u32, check: &'static str) -> Asked {
    control(
        session,
        "frames",
        json!({ "count": count, "interval_ms": FRAME_MS }),
        check,
    )
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

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn idle(milliseconds: f64) {
    let until =
        std::time::Instant::now() + std::time::Duration::from_secs_f64(milliseconds / 1000.0);
    while std::time::Instant::now() < until {
        std::hint::spin_loop();
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn idle(_: f64) {}

pub(crate) fn serve(session: &mut dyn SmokeSession, document: &Value) -> Result<(), SmokeFailure> {
    session.source(&document.to_string());
    control(session, "pause", json!({}), "pauses")?;
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
    let mut materials =
        vec![json!({ "id": "ground", "color": [0.55, 0.58, 0.52, 1.0], "texture": "checker" })];
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

fn base64(bytes: &[u8]) -> String {
    const LETTERS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let joined = chunk.iter().enumerate().fold(0_u32, |joined, (at, &byte)| {
            joined | (u32::from(byte) << (16 - 8 * at))
        });
        for at in 0..4 {
            if at <= chunk.len() {
                text.push(char::from(
                    LETTERS[((joined >> (18 - 6 * at)) & 63) as usize],
                ));
            } else {
                text.push('=');
            }
        }
    }
    text
}
