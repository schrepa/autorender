use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{smoke_check, smoke_eq};
use serde_json::{Value, json};

use super::scene_kit::{SceneFixture, ask, component, control, query, render};

fn fixture() -> Value {
    let cube = json!({"shape": {"kind": "cuboid", "half_extents": [0.5, 0.5, 0.5]}});
    json!({
        "id": "streaming-fixture", "revision": 1,
        "camera": {"position": [0, 60, 40], "target": [20, 20, 0], "fov_degrees": 60, "near": 0.5, "far": 20000},
        "materials": [{"id": "plain", "color": [0.8, 0.3, 0.2, 1]}],
        "meshes": [{"id": "box", "source": {"kind": "cube"}}],
        "prefabs": [{"id": "lamp", "entities": [
            {"id": "post", "mesh": "box", "material": "plain", "transform": {"translation": [0, 1, 0], "scale": [0.2, 2, 0.2]}},
            {"id": "light", "light": {"color": [1, 0.7, 0.4], "intensity": 10, "range": 10}, "transform": {"translation": [0, 2.2, 0]}},
        ]}],
        "entities": [
            {"id": "player", "mesh": "box", "material": "plain", "tags": ["player"],
             "collider": {"shape": cube["shape"], "memberships": 2, "filter": 1, "sensor": false},
             "transform": {"translation": [0, 200, 0]}},
            {"id": "ground", "terrain": {"origin": [-1024, 0, -1024], "tiles": [8, 8], "spacing": 1,
             "layers": [{"noise": {"seed": 5, "height": 40, "wavelength": 400}}]}},
        ],
        "streamed": [
            {"id": "crate-near", "mesh": "box", "material": "plain", "collider": cube, "grounded": {"offset": 0.5}, "transform": {"translation": [20, 0, 0]}},
            {"id": "crate-mid", "mesh": "box", "material": "plain", "grounded": {"offset": 0.5}, "transform": {"translation": [900, 0, 0]}},
            {"id": "spinner", "transform": {"translation": [-30, 30, 10]}},
            {"id": "spinner-blade", "parent": "spinner", "mesh": "box", "material": "plain", "transform": {"scale": [3, 0.2, 0.4]}},
            {"id": "lamp-far", "prefab": "lamp", "transform": {"translation": [1500, 0, 300]}},
            {"id": "tower", "mesh": "box", "material": "plain", "transform": {"translation": [5000, 30, 0], "scale": [10, 60, 10]}},
        ],
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let source = fixture();
    let (mut scene, _) = SceneFixture::load(session, &source)?;
    ask(session, "control", json!({"name": "pause"}))?;
    control(session, "environment", json!({"terrain": {"samples_per_frame": 50_000_000}}))?;
    let mut rendering = look(session, 0.0, 0.0)?;
    for _ in 0..400 {
        if rendering["terrain"]["loading_tiles"] == 0 && rendering["terrain"]["fallback_tiles"] == 0 {
            break;
        }
        rendering = render(session)?;
    }
    let cells = ask(session, "inspect", json!({"component": "WorldCells"}))?["value"].clone();
    let grids: Vec<Value> = cells["grids"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|grid| json!([grid["cells"], grid["loaded"]]))
        .collect();
    smoke_eq!(
        json!({"items": cells["items"], "grids": grids}),
        json!({"items": 5, "grids": [[3, 2], [1, 1], [1, 1]]}),
        "binning"
    );
    let streamed = &rendering["world_streaming"];
    smoke_eq!(
        json!([streamed["cells"], streamed["loaded"], streamed["pending"], streamed["entities"]]),
        json!([5, 4, 0, 7]),
        "binning_stats"
    );
    smoke_check!(!loaded(session, "crate-mid")?, "middle_crate_out_of_range", json!("crate-mid"), json!("unloaded"));
    grounded(session)?;
    let post = component(session, "lamp-far.post", "Hierarchy")?;
    smoke_eq!(json!(post["parent"]), json!("lamp-far"), "items");
    look(session, 1000.0, 0.0)?;
    smoke_check!(!loaded(session, "crate-near")?, "far_crate_goes", json!("crate-near"), json!("unloaded"));
    let missed = query(session, "raycast", down())?;
    smoke_check!(
        missed["hit"]["entity"]["id"] != "crate-near",
        "unloaded_crate_no_longer_collides",
        json!(missed["hit"]),
        json!({"not": "crate-near"})
    );
    let mut fewer = source.clone();
    if let Some(items) = fewer["streamed"].as_array_mut() {
        items.truncate(1);
    }
    let reloaded = scene.reload(session, &fewer);
    smoke_check!(reloaded.is_ok(), "reload_accepted", json!(reloaded.as_ref().err()), json!(null));
    let cells = ask(session, "inspect", json!({"component": "WorldCells"}))?["value"].clone();
    smoke_eq!(json!(cells["items"]), json!(1), "reload");
    refusals(session, &mut scene, &source)
}

fn grounded(session: &mut dyn SmokeSession) -> SmokeResult {
    let samples = query(session, "terrain_height", json!({"points": [[20, 0]]}))?;
    let ground = samples["samples"][0].clone();
    let crate_at = component(session, "crate-near", "Transform")?;
    let height = ground["height"].as_f64().unwrap_or(f64::NAN);
    smoke_check!(
        crate_at["translation"][1]
            .as_f64()
            .is_some_and(|y| (y - (height + 0.5)).abs() < 1e-3),
        "grounded",
        json!(crate_at["translation"]),
        json!({"y": height + 0.5})
    );
    let held = component(session, "crate-near", "Grounded")?;
    let level = if ground["level"].is_null() { json!(0) } else { ground["level"].clone() };
    smoke_eq!(
        json!({"offset": held["offset"], "level": held["level"]}),
        json!({"offset": 0.5, "level": level}),
        "grounded_inspects"
    );
    let hit = query(session, "raycast", down())?["hit"].clone();
    smoke_eq!(
        json!({"id": hit["entity"]["id"], "active": hit["entity"]["in_active_scene"]}),
        json!({"id": "crate-near", "active": true}),
        "streamed_crate_collides"
    );
    Ok(())
}

fn refusals(session: &mut dyn SmokeSession, scene: &mut SceneFixture, source: &Value) -> SmokeResult {
    for (case, code, path) in [
        ("terrain", "invalid_streamed", "$.streamed[6].terrain"),
        ("parent_across_lists", "unknown_parent", "$.streamed[5].parent"),
        ("grounded_child", "invalid_grounded", "$.streamed[3].grounded"),
        ("duplicate_id", "duplicate_id", "$.streamed[0].id"),
    ] {
        let mut broken = source.clone();
        match case {
            "terrain" => {
                let land = json!({"id": "land", "terrain": source["entities"][1]["terrain"]});
                if let Some(items) = broken["streamed"].as_array_mut() {
                    items.push(land);
                }
            }
            "parent_across_lists" => broken["streamed"][5]["parent"] = json!("player"),
            "grounded_child" => broken["streamed"][3]["grounded"] = json!({}),
            _ => broken["streamed"][0]["id"] = json!("player"),
        }
        let refused = scene.reload(session, &broken).err().unwrap_or_default();
        smoke_eq!(
            json!({"case": case, "code": refused["code"], "path": refused["path"]}),
            json!({"case": case, "code": code, "path": path}),
            "refusals"
        );
    }
    Ok(())
}

fn down() -> Value {
    json!({"origin": [20, 300, 0], "direction": [0, -1, 0], "max_distance": 600})
}

fn look(session: &mut dyn SmokeSession, x: f64, z: f64) -> Result<Value, SmokeFailure> {
    let changed = control(
        session,
        "camera",
        json!({
            "position": [x, 60.0, z + 40.0], "target": [x + 20.0, 20.0, z],
            "vertical_fov_radians": std::f64::consts::FRAC_PI_3, "near": 0.5, "far": 20000,
        }),
    )?;
    Ok(changed["rendering"].clone())
}

fn loaded(session: &mut dyn SmokeSession, id: &str) -> Result<bool, SmokeFailure> {
    let reply = session.request("inspect", json!({"entity": id, "component": "Transform"}));
    let missing = reply
        .as_ref()
        .err()
        .map(|refused| refused["error"]["code"].clone());
    smoke_check!(
        missing.as_ref().is_none_or(|code| code == "unknown_entity"),
        "loaded_inspects",
        json!(missing),
        json!("unknown_entity")
    );
    Ok(reply.is_ok())
}
