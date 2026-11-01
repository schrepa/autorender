use engine_core::{SmokeFailure, SmokeSession, smoke_eq};
use serde_json::{Value, json};

use super::requests::seen;
use super::scene_kit::{ask, capture, control, load, render};
use super::stats::terrain;

const VIEWS: [&str; 14] = [
    "water_depth",
    "water_speed",
    "water_froude",
    "water_turbulence",
    "water_white",
    "water_reach",
    "water_span",
    "water_eddies",
    "water_roughness",
    "levels",
    "tiles",
    "materials",
    "sky",
    "detail",
];
const GROUND_POINTS: [[f64; 2]; 6] = [
    [0.5, 0.75],
    [0.4, 0.8],
    [0.6, 0.8],
    [0.5, 0.9],
    [0.3, 0.7],
    [0.7, 0.7],
];

fn fixture() -> Value {
    json!({
        "id": "terrain-views-fixture", "revision": 1,
        "camera": {"position": [0, 80, 140], "target": [0, 30, 0], "fov_degrees": 60, "near": 0.5, "far": 4000},
        "materials": [],
        "meshes": [],
        "environment": {"ground_cover": {"looks": {"meadow": {"density": 60}}}},
        "entities": [{"id": "ground", "terrain": {
            "origin": [-512, 0, -512], "tiles": [4, 4], "spacing": 1,
            "layers": [
                {"type": "noise", "seed": 3, "height": 140, "wavelength": 600},
                {"type": "stamp", "center": [512, 512], "radius": 40, "falloff": 60, "op": {"flatten": 30}},
            ],
            "surface": {
                "materials": [
                    {"id": "meadow", "color": [0.12, 0.2, 0.06]},
                    {"id": "rock", "color": [0.3, 0.28, 0.26], "roughness": 0.8},
                    {"id": "snow", "color": [0.85, 0.87, 0.9], "roughness": 0.5},
                ],
                "biomes": [
                    {"id": "lowland", "layers": [{"material": "meadow"}, {"material": "rock", "mask": {"ramp": {"map": "slope", "from": 25, "to": 35}}}]},
                    {"id": "summit", "fitness": {"ramp": {"map": "height", "from": 110, "to": 130}}, "layers": [{"material": "snow"}]},
                ],
            },
        }}],
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    load(session, &fixture())?;
    ask(session, "control", json!({"name": "pause"}))?;
    let opened = control(
        session,
        "environment",
        json!({"terrain": {"samples_per_frame": 50_000_000}}),
    )?;
    smoke_eq!(
        json!(opened["environment_changed"]),
        json!(["terrain"]),
        "terrain_settings_open"
    );
    control(
        session,
        "camera",
        json!({"position": [0, 80, 140], "target": [0, 30, 0],
            "vertical_fov_radians": std::f64::consts::FRAC_PI_3, "near": 0.5, "far": 4000}),
    )?;
    settle(session)?;
    let shaded = shown(session)?;
    for view in VIEWS {
        control(
            session,
            "environment",
            json!({"debug_view": {"view": view}}),
        )?;
        let image = shown(session)?;
        let wanted = json!({"not": shaded});
        smoke_eq!(
            seen(
                image != shaded,
                json!({"view": view, "image": image}),
                &wanted,
            ),
            wanted,
            "terrain_views_draw"
        );
    }
    let rendering = render(session)?;
    let width = rendering["resolution"]["display_width"]
        .as_f64()
        .unwrap_or(0.0);
    let height = rendering["resolution"]["display_height"]
        .as_f64()
        .unwrap_or(0.0);
    let mut ground = Value::Null;
    for [x, y] in GROUND_POINTS {
        let probed = control(
            session,
            "probe",
            json!({"x": (width * x).floor() as u64, "y": (height * y).floor() as u64}),
        )?;
        if probed["kind"] == "terrain" {
            ground = probed;
            break;
        }
    }
    let wanted = json!({"kind": "terrain"});
    smoke_eq!(
        seen(ground.is_object(), json!(ground), &wanted),
        wanted,
        "probe_finds_ground"
    );
    terrain_probe(&ground)?;
    let above = control(session, "probe", json!({"x": (width / 2.0) as u64, "y": 1}))?;
    smoke_eq!(
        json!({"terrain": above["terrain"], "water": above["water"]}),
        json!({"terrain": null, "water": null}),
        "probe_above_ground"
    );
    control(
        session,
        "environment",
        json!({"debug_view": {"view": "off"}}),
    )?;
    Ok(())
}

fn terrain_probe(ground: &Value) -> Result<(), SmokeFailure> {
    let made = &ground["terrain"];
    let tile = made["tile"]
        .as_array()
        .is_some_and(|tile| tile.iter().all(Value::is_i64));
    let patch = made["patch"].as_array().is_some_and(|patch| {
        patch
            .iter()
            .all(|value| value.as_u64().is_some_and(|value| value < 16))
    });
    let wanted = json!({"patch": "below 16"});
    smoke_eq!(
        seen(tile && patch, json!(made), &wanted),
        wanted,
        "terrain_probe_tile"
    );
    let wanted = json!({"below": 8});
    smoke_eq!(
        seen(
            ground["level"].as_u64().is_some_and(|level| level < 8),
            json!(ground["level"]),
            &wanted,
        ),
        wanted,
        "terrain_probe_level"
    );
    let shares: Vec<f64> = made["materials"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|share| share["share"].as_f64().unwrap_or(0.0))
        .collect();
    let total: f64 = shares.iter().sum();
    let wanted = json!({"sum": 1});
    smoke_eq!(
        seen(
            !shares.is_empty()
                && shares.iter().all(|&share| share > 0.0 && share <= 1.0)
                && (total - 1.0).abs() < 0.05,
            json!(made["materials"]),
            &wanted,
        ),
        wanted,
        "terrain_probe_materials"
    );
    let sky = made["sky"].as_f64().unwrap_or(-1.0);
    let detailed = made["detail"]["detailed"].as_f64().unwrap_or(-1.0);
    let averaged = made["detail"]["averaged"].as_f64().unwrap_or(f64::MAX);
    let wanted = json!({"sky": [0, 1], "detailed": "at least averaged"});
    smoke_eq!(
        seen(
            (0.0..=1.0).contains(&sky) && detailed >= averaged - 0.01,
            json!({"sky": made["sky"], "detail": made["detail"]}),
            &wanted,
        ),
        wanted,
        "terrain_probe_light"
    );
    let wanted = json!(null);
    smoke_eq!(
        seen(ground["water"].is_null(), json!(ground["water"]), &wanted),
        wanted,
        "terrain_probe_dry"
    );
    Ok(())
}

fn settle(session: &mut dyn SmokeSession) -> Result<(), SmokeFailure> {
    let mut resident = 0;
    let mut rendering = Value::Null;
    for _ in 0..600 {
        rendering = render(session)?;
        let drawn = terrain(&rendering);
        let ready = drawn["tiles"].as_u64().unwrap_or(0) > 0
            && drawn["loading_tiles"] == 0
            && drawn["fallback_tiles"] == 0;
        resident = if ready { resident + 1 } else { 0 };
        if resident > 1 {
            return Ok(());
        }
    }
    let wanted = json!({"loading_tiles": 0});
    smoke_eq!(
        seen(false, terrain(&rendering), &wanted),
        wanted,
        "terrain_settles"
    );
    Ok(())
}

fn shown(session: &mut dyn SmokeSession) -> Result<Value, SmokeFailure> {
    let mut compiling = true;
    for _ in 0..600 {
        compiling = render(session)?["pipelines_compiling"] == true;
        if !compiling {
            break;
        }
    }
    let wanted = json!(false);
    smoke_eq!(
        seen(!compiling, json!(compiling), &wanted),
        wanted,
        "pipelines_compile"
    );
    capture(session)
}
