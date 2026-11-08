use engine_smoke::{Checked, control, inspect, number, settled};
use engine_core::{BrickList, Runs, Smoke, SmokeSession, smoke_eq};
use engine_registry::Registry;
use serde_json::{Value, json};

pub(super) const LIT_ROOM: Smoke = Smoke {
    name: "lit_room",
    runs: Runs::Gpu,
    run: lit_room,
};

const ROOM_KEPT: [&str; 9] = [
    "engine_sun",
    "engine_lighting",
    "engine_meshes",
    "engine_shadows",
    "engine_prepass",
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

pub(super) fn room(bricks: BrickList) -> BrickList {
    ROOM_LEFT_OUT
        .iter()
        .fold(bricks, |bricks, package| bricks.without_package(package))
}

pub(super) fn room_listed(registry: &Registry) -> bool {
    ROOM_KEPT
        .iter()
        .all(|package| registry.id(package).is_some())
}

fn opened(session: &mut dyn SmokeSession, document: &Value) -> Checked<Value> {
    session.source(&document.to_string());
    control(session, "pause", json!({}))?;
    settled(session)
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
    smoke_eq!(
        nobody.contains(&json!("sky")),
        true,
        "room_sky_from_contract"
    );
    let shadows = inspect(session, "shadow_stats")?;
    smoke_eq!(
        json!({"active": shadows["active"], "drawn": number(&shadows["tiles_resident"]) > 0.0}),
        json!({"active": true, "drawn": true}),
        "room_shadows_draw"
    );
    let lights = inspect(session, "local_light_stats")?;
    smoke_eq!(number(&lights["shaded"]) > 0.0, true, "room_lamp_shades");
    Ok(())
}

fn room_scene() -> Value {
    json!({
        "id": "lit-room", "revision": 1,
        "environment": {"time_of_day": {"enabled": false}, "sun": {"sun_direction": [0.55, 0.75, 0.35]}},
        "materials": [
            {"id": "wall", "color": [0.7, 0.7, 0.68, 1]},
            {"id": "glow", "color": [1, 1, 1, 1]},
        ],
        "meshes": [{"id": "cube", "source": {"kind": "cube"}}],
        "entities": [
            {"id": "floor", "drawn": {"mesh": "cube", "material": "wall"}, "transform": {"translation": [0, -0.5, 0], "scale": [12, 1, 12]}},
            {"id": "block", "drawn": {"mesh": "cube", "material": "wall"}, "transform": {"translation": [0, 1, 0], "scale": [2, 2, 2]}},
            {"id": "back_wall", "drawn": {"mesh": "cube", "material": "wall"}, "transform": {"translation": [0, 2, -6], "scale": [12, 4, 0.3]}},
            {"id": "side_wall", "drawn": {"mesh": "cube", "material": "wall"}, "transform": {"translation": [-6, 2, 0], "scale": [0.3, 4, 12]}},
            {"id": "lamp", "drawn": {"mesh": "cube", "material": "glow"}, "light": {"color": [1, 0.8, 0.6], "intensity": 14.0}, "transform": {"translation": [3, 2, 2], "scale": [0.2, 0.2, 0.2]}},
            {"id": "camera", "transform": {"translation": [0, 3, 9], "rotation_degrees": [-12.5288, 0, 0]}, "node": {"clock": "real", "mobility": "moving"}, "camera": {"near": 0.1, "far": 200, "projection": {"Perspective": {"vertical_fov_radians": std::f64::consts::FRAC_PI_3}}}, "view": {}},
        ],
    })
}
