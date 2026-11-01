use engine_core::smoke::{Checked, compiled, control, inspect, number, render};
use engine_core::{BrickList, Runs, Smoke, SmokeSession, smoke_eq};
use engine_registry::Registry;
use serde_json::{Value, json};

const FRAMES: u32 = 8;
const SETTLE_FRAMES: u32 = 600;

pub(super) const LIT_ROOM: Smoke = Smoke {
    name: "lit_room",
    runs: Runs::Gpu,
    run: lit_room,
};

pub(super) const OUTDOOR_WITHOUT_TREES: Smoke = Smoke {
    name: "outdoor_without_trees",
    runs: Runs::Gpu,
    run: outdoor_without_trees,
};

const ROOM_KEPT: [&str; 8] = [
    "engine_sun",
    "engine_lighting",
    "engine_meshes",
    "engine_shadows",
    "engine_lights",
    "engine_tonemap",
    "engine_bloom",
    "engine_dither",
];

const ROOM_LEFT_OUT: [&str; 13] = [
    "engine_terrain",
    "engine_terrain_shading",
    "engine_scatter",
    "engine_sky",
    "engine_probes",
    "engine_ground_cover",
    "engine_water",
    "engine_ambient_occlusion",
    "engine_reflections",
    "engine_particles",
    "engine_taa",
    "engine_auto_exposure",
    "engine_debug_view",
];

const OUTDOOR_KEPT: [&str; 6] = [
    "engine_terrain",
    "engine_terrain_shading",
    "engine_shadows",
    "engine_sky",
    "engine_probes",
    "engine_meshes",
];

const OUTDOOR_LEFT_OUT: &str = "engine_scatter";

pub(super) fn room(bricks: BrickList) -> BrickList {
    ROOM_LEFT_OUT
        .iter()
        .fold(bricks, |bricks, package| bricks.without_package(package))
}

pub(super) fn outdoor(bricks: BrickList) -> BrickList {
    bricks.without_package(OUTDOOR_LEFT_OUT)
}

pub(super) fn room_listed(registry: &Registry) -> bool {
    ROOM_KEPT.iter().all(|package| registry.id(package).is_some())
}

pub(super) fn outdoor_listed(registry: &Registry) -> bool {
    OUTDOOR_KEPT.iter().all(|package| registry.id(package).is_some())
}

fn opened(session: &mut dyn SmokeSession, document: &Value) -> Checked<Value> {
    session.source(&document.to_string());
    control(session, "pause", json!({}))?;
    let mut rendering = compiled(session)?;
    for _ in 0..FRAMES {
        rendering = render(session)?;
    }
    Ok(rendering)
}

fn settled(session: &mut dyn SmokeSession, key: &str, field: &str) -> Checked<Value> {
    let mut stats = inspect(session, key)?;
    for _ in 0..SETTLE_FRAMES {
        if number(&stats[field]) > 0.0 {
            break;
        }
        render(session)?;
        stats = inspect(session, key)?;
    }
    Ok(stats)
}

fn lit_room(session: &mut dyn SmokeSession) -> Checked<()> {
    let rendering = opened(session, &room_scene())?;
    let linked = &rendering["linked"];
    smoke_eq!(
        json!({
            "sun_blockers": linked["lists"]["sun_blockers"],
            "ambient": linked["lists"]["ambient"],
            "interior": linked["lists"]["interior"],
        }),
        json!({
            "sun_blockers": ["engine_shadows"],
            "ambient": [],
            "interior": [],
        }),
        "room_lists"
    );
    let nobody = linked["nobody"].as_array().cloned().unwrap_or_default();
    smoke_eq!(nobody.contains(&json!("sky")), true, "room_sky_from_contract");
    let shadows = settled(session, "shadow_stats", "draws")?;
    smoke_eq!(
        json!({"active": shadows["active"], "drawn": number(&shadows["draws"]) > 0.0}),
        json!({"active": true, "drawn": true}),
        "room_shadows_draw"
    );
    let lights = settled(session, "local_light_stats", "shaded")?;
    smoke_eq!(number(&lights["shaded"]) > 0.0, true, "room_lamp_shades");
    Ok(())
}

fn outdoor_without_trees(session: &mut dyn SmokeSession) -> Checked<()> {
    let rendering = opened(session, &outdoor_scene())?;
    let linked = &rendering["linked"];
    smoke_eq!(
        json!({
            "sun_blockers": linked["lists"]["sun_blockers"],
            "ambient": linked["lists"]["ambient"],
        }),
        json!({
            "sun_blockers": ["engine_shadows", "engine_sky", "engine_terrain_shading"],
            "ambient": ["engine_probes", "engine_sky", "engine_terrain_shading"],
        }),
        "outdoor_lists"
    );
    let named = linked["lists"].as_object().is_some_and(|lists| {
        lists
            .values()
            .filter_map(Value::as_array)
            .any(|members| members.contains(&json!(OUTDOOR_LEFT_OUT)))
    });
    smoke_eq!(named, false, "outdoor_lists_leave_scatter_out");
    let faces = linked["faces"].as_object().cloned().unwrap_or_default();
    smoke_eq!(faces.contains_key("occupancy"), false, "outdoor_no_occupancy_face");
    Ok(())
}

fn room_scene() -> Value {
    json!({
        "id": "lit-room", "revision": 1,
        "camera": {"position": [0, 3, 9], "target": [0, 1, 0], "fov_degrees": 60, "near": 0.1, "far": 200},
        "environment": {"sun": {"preset": "day"}},
        "materials": [
            {"id": "wall", "color": [0.7, 0.7, 0.68, 1]},
            {"id": "glow", "color": [1, 1, 1, 1]},
        ],
        "meshes": [{"id": "cube", "source": {"kind": "cube"}}],
        "entities": [
            {"id": "floor", "mesh": "cube", "material": "wall", "transform": {"translation": [0, -0.5, 0], "scale": [12, 1, 12]}},
            {"id": "block", "mesh": "cube", "material": "wall", "transform": {"translation": [0, 1, 0], "scale": [2, 2, 2]}},
            {"id": "back_wall", "mesh": "cube", "material": "wall", "transform": {"translation": [0, 2, -6], "scale": [12, 4, 0.3]}},
            {"id": "side_wall", "mesh": "cube", "material": "wall", "transform": {"translation": [-6, 2, 0], "scale": [0.3, 4, 12]}},
            {"id": "lamp", "mesh": "cube", "material": "glow", "light": {"color": [1, 0.8, 0.6], "intensity": 14.0}, "transform": {"translation": [3, 2, 2], "scale": [0.2, 0.2, 0.2]}},
        ],
    })
}

fn outdoor_scene() -> Value {
    json!({
        "id": "outdoor-without-trees", "revision": 1,
        "camera": {"position": [0, 80, 140], "target": [0, 30, 0], "fov_degrees": 60, "near": 0.5, "far": 4000},
        "environment": {"ground_cover": {"looks": {"meadow": {"density": 60}}}},
        "materials": [{"id": "stone", "color": [0.5, 0.48, 0.45, 1]}],
        "meshes": [{"id": "cube", "source": {"kind": "cube"}}],
        "entities": [
            {"id": "ground", "terrain": {
                "origin": [-512, 0, -512], "tiles": [4, 4], "spacing": 1,
                "layers": [
                    {"type": "noise", "seed": 3, "height": 140, "wavelength": 600},
                    {"type": "stamp", "center": [512, 512], "radius": 40, "falloff": 60, "op": {"flatten": 30}},
                ],
                "surface": {
                    "materials": [
                        {"id": "meadow", "color": [0.12, 0.2, 0.06]},
                        {"id": "rock", "color": [0.3, 0.28, 0.26], "roughness": 0.8},
                    ],
                    "biomes": [
                        {"id": "lowland", "layers": [{"material": "meadow"}, {"material": "rock", "mask": {"ramp": {"map": "slope", "from": 25, "to": 35}}}]},
                    ],
                },
            }},
            {"id": "block", "mesh": "cube", "material": "stone", "transform": {"translation": [0, 34, 0], "scale": [4, 4, 4]}},
        ],
    })
}
