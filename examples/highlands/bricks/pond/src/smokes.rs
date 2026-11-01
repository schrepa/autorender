use engine_core::smoke::{Checked, compiled, component, control, inspect, number, query, render};
use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

const SURFACE: f64 = 2.0;
const POND_SIZE: f64 = 16.0;
const DROP: f64 = 6.0;
const FLOATS_AT: f64 = SURFACE - 0.1;
const FLOAT_WITHIN: f64 = 0.3;
const DEPTH_WITHIN: f64 = 1.0e-4;
const FLOAT_TICKS: u32 = 480;
const TICKS_AT_ONCE: u32 = 120;
const SETTLE_FRAMES: u32 = 20_000;

pub(crate) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    session.source(&scene(true).to_string());
    control(session, "pause", json!({}))?;
    settle(session)?;
    let held = component(session, "pond", "pond")?;
    smoke_eq!(held, authored(), "the_pond_spawns_from_the_scene");
    holds_water(session)?;
    floats(session)?;
    draws(session)
}

fn authored() -> Value {
    json!({"center": [0.0, 0.0], "size": [POND_SIZE, POND_SIZE], "surface": SURFACE})
}

fn scene(pond: bool) -> Value {
    let mut entities = vec![
        json!({
            "id": "ground",
            "terrain": {
                "origin": [-128, 0, -128],
                "tiles": [1, 1],
                "spacing": 1,
                "layers": [{ "type": "noise", "seed": 3, "height": 0, "wavelength": 120 }],
            },
        }),
        json!({
            "id": "body",
            "transform": { "translation": [0.0, DROP, 0.0] },
            "collider": { "shape": { "kind": "cuboid", "half_extents": [0.5, 0.5, 0.5] } },
            "rigid_body": {},
        }),
    ];
    if pond {
        entities.push(json!({ "id": "pond", "pond": authored() }));
    }
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
        "textures": [],
        "materials": [],
        "meshes": [],
        "entities": entities,
    })
}

fn settle(session: &mut dyn SmokeSession) -> Checked<Value> {
    let mut terrain = Value::Null;
    let mut resident = 0;
    for _ in 0..SETTLE_FRAMES {
        render(session)?;
        terrain = inspect(session, "terrain_stats")?;
        let ready = number(&terrain["resident_tiles"]) > 0.0
            && terrain["loading_tiles"] == 0
            && terrain["fallback_tiles"] == 0
            && terrain["baking"] == 0;
        resident = if ready { resident + 1 } else { 0 };
        if resident > 1 {
            break;
        }
    }
    smoke_eq!(resident > 1, true, "terrain_settles");
    Ok(terrain)
}

fn holds_water(session: &mut dyn SmokeSession) -> Checked<()> {
    let found = query(session, "terrain_height", json!({"points": [[0.0, 0.0]]}))?;
    let sample = &found["samples"][0];
    let ground = number(&sample["height"]);
    let water = &sample["standing_water"];
    let depth = SURFACE - ground;
    smoke_eq!(depth > 0.0, true, "the_pond_stands_above_the_ground");
    smoke_eq!(
        (number(&water["surface"]) - SURFACE).abs() <= DEPTH_WITHIN,
        true,
        "the_pond_holds_water_at_its_surface"
    );
    smoke_eq!(
        (number(&water["depth"]) - depth).abs() <= DEPTH_WITHIN,
        true,
        "the_pond_holds_water_down_to_the_ground"
    );
    Ok(())
}

fn floats(session: &mut dyn SmokeSession) -> Checked<()> {
    for _ in 0..FLOAT_TICKS / TICKS_AT_ONCE {
        session.step(TICKS_AT_ONCE);
    }
    let placed = component(session, "body", "transform")?;
    let height = number(&placed["translation"][1]);
    smoke_eq!(
        (height - FLOATS_AT).abs() <= FLOAT_WITHIN,
        true,
        "the_body_floats_on_the_pond"
    );
    Ok(())
}

fn draws(session: &mut dyn SmokeSession) -> Checked<()> {
    let shown = compiled(session)?;
    smoke_eq!(drawn(&shown), true, "the_pond_pass_draws");
    session.source(&scene(false).to_string());
    control(session, "pause", json!({}))?;
    render(session)?;
    let dry = render(session)?;
    smoke_eq!(drawn(&dry), false, "a_scene_without_ponds_skips_the_pass");
    Ok(())
}

fn drawn(rendering: &Value) -> bool {
    rendering["render_passes"].as_array().is_some_and(|passes| {
        passes
            .iter()
            .filter_map(Value::as_str)
            .any(|pass| pass == "pond" || pass.ends_with("::pond"))
    })
}
