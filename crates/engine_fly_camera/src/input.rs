use engine_core::{SmokeFailure, SmokeSession};
use serde_json::{Value, json};

type Asked = Result<Value, SmokeFailure>;

macro_rules! check {
    ($condition:expr, $check:expr, $got:expr, $expected:expr $(,)?) => {
        if !$condition {
            return Err(SmokeFailure::new(
                $check,
                &$got,
                &$expected,
                concat!(file!(), ":", line!()),
            ));
        }
    };
}

fn accepted(session: &mut dyn SmokeSession, verb: &str, body: Value, check: &'static str) -> Asked {
    let reply = match session.request(verb, body) {
        Ok(reply) | Err(reply) => reply,
    };
    check!(
        reply["ok"] == true,
        check,
        reply["error"].clone(),
        json!({ "ok": true })
    );
    Ok(reply)
}

fn control(session: &mut dyn SmokeSession, name: &str, args: Value, check: &'static str) -> Asked {
    accepted(
        session,
        "control",
        json!({ "name": name, "args": args }),
        check,
    )
}

fn inspected(session: &mut dyn SmokeSession, component: &str, check: &'static str) -> Asked {
    let reply = accepted(session, "inspect", json!({ "component": component }), check)?;
    Ok(reply["result"]["value"].clone())
}

fn grid(side: usize) -> Value {
    let paints = [
        [0.85, 0.35, 0.2, 1.0],
        [0.25, 0.6, 0.85, 1.0],
        [0.4, 0.75, 0.3, 1.0],
        [0.9, 0.75, 0.25, 1.0],
    ];
    let ground = json!({
        "id": "ground", "mesh": "box", "material": "ground",
        "transform": { "translation": [0.0, -0.5, 0.0], "scale": [side as f64 * 6.0, 1.0, side as f64 * 6.0] },
    });
    let mut entities = vec![ground];
    let middle = (side as f64 - 1.0) / 2.0;
    for index in 0..side * side {
        let (row, column) = ((index / side) as f64, (index % side) as f64);
        let scale = 0.8 + (index % 5) as f64 * 0.3;
        entities.push(json!({
            "id": format!("piece-{index}"),
            "mesh": if index % 3 == 0 { "gem" } else { "box" },
            "material": format!("paint-{}", index % paints.len()),
            "transform": {
                "translation": [(column - middle) * 4.0, scale / 2.0, (row - middle) * 4.0],
                "rotation_degrees": [0, (index * 37) % 360, 0],
                "scale": [scale, scale, scale],
            },
        }));
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

fn axes(value: &Value) -> [f64; 3] {
    std::array::from_fn(|axis| value[axis].as_f64().unwrap_or(f64::NAN))
}

fn direction(camera: &Value) -> [f64; 3] {
    let (position, target) = (axes(&camera["position"]), axes(&camera["target"]));
    let delta: [f64; 3] = std::array::from_fn(|axis| target[axis] - position[axis]);
    let length = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
    delta.map(|value| value / length)
}

fn moved(from: &Value, to: &Value) -> [f64; 3] {
    let (from, to) = (axes(&from["position"]), axes(&to["position"]));
    std::array::from_fn(|axis| to[axis] - from[axis])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn camera(session: &mut dyn SmokeSession) -> Asked {
    inspected(session, "Camera", "inspects_camera")
}

fn steps(session: &mut dyn SmokeSession, input: Value, ticks: u32) -> Result<(), SmokeFailure> {
    control(session, "input", input, "takes_input")?;
    control(session, "step", json!({ "ticks": ticks }), "steps")?;
    Ok(())
}

pub(crate) fn input_camera(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&grid(8).to_string());
    control(session, "pause", json!({}), "pauses")?;
    let flies = inspected(session, "fly_camera", "fly_camera_added")?["enabled"] == true;
    check!(flies, "fly_camera_added", json!(flies), json!(true));
    let start = camera(session)?;
    steps(session, json!({ "forward": 1 }), 20)?;
    let forward = camera(session)?;
    let along = dot(moved(&start, &forward), direction(&start));
    check!(
        along > 0.5,
        "key_moves",
        json!({ "start": start, "forward": forward }),
        json!({ "along": "> 0.5" })
    );
    steps(session, json!({}), 10)?;
    let still = camera(session)?;
    let drift = dot(moved(&forward, &still), moved(&forward, &still)).sqrt();
    check!(drift < 1e-3, "release_stops", json!(drift), json!(1e-3));
    steps(session, json!({ "yaw": 120, "pitch": 40 }), 10)?;
    let turned = camera(session)?;
    let cosine = dot(direction(&still), direction(&turned));
    let turn = if cosine < 1.0 { cosine.acos() } else { 0.0 };
    check!(turn > 0.05, "drag_turns", json!(turn), json!(0.05));
    steps(session, json!({}), 1)?;
    Ok(())
}
