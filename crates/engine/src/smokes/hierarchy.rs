use engine_core::glam::{Mat4, Quat, Vec3};
use engine_core::smoke::{SmokeFailure, SmokeResult, SmokeSession};
use engine_core::{edit_transform, smoke_check, smoke_eq};
use serde_json::{Value, json};

use super::scene_kit::{
    SceneFixture, ask, capture, component, near_all, numbers, query, refuse, residency,
};

const SCENE: &str = "hierarchy-fixture";
const PREFAB: &str = "garden-lantern";
const PARTS: [&str; 2] = ["lantern", "planter"];
const EPSILON: f64 = 0.0002;

fn fixture() -> Value {
    json!({
        "id": SCENE, "revision": 1,
        "camera": {"position": [0, 9, 16], "target": [0, 0, -2], "fov_degrees": 55, "near": 0.1, "far": 500},
        "materials": [
            {"id": "ground", "color": [0.45, 0.5, 0.42, 1]},
            {"id": "cloth", "color": [0.2, 0.35, 0.7, 1]},
            {"id": "brass", "color": [0.85, 0.65, 0.3, 1]},
            {"id": "clay", "color": [0.6, 0.35, 0.25, 1]},
        ],
        "meshes": [
            {"id": "box", "source": {"kind": "cube"}},
            {"id": "lamp", "source": {"kind": "octahedron"}},
            {"id": "pot", "source": {"kind": "plane"}},
        ],
        "prefabs": [{"id": PREFAB, "entities": [
            {"id": "lantern", "mesh": "lamp", "material": "brass",
             "transform": {"translation": [0, 1.4, 0], "rotation_degrees": [17, 31, -9], "scale": [0.6, 0.8, 0.6]}},
            {"id": "planter", "mesh": "pot", "material": "clay",
             "transform": {"translation": [0.6, 0.2, 0], "scale": [1.2, 1, 0.8]}},
        ]}],
        "entities": [
            {"id": "ground", "mesh": "box", "material": "ground", "transform": {"translation": [0, -0.5, 0], "scale": [30, 1, 30]}},
            {"id": "player", "tags": ["player"], "mesh": "box", "material": "cloth", "transform": {"translation": [0, 0.5, 4]}},
            {"id": "player-accessory", "parent": "player", "mesh": "box", "material": "brass",
             "transform": {"translation": [0, 0.8, 0], "scale": [0.3, 0.3, 0.3]}},
            {"id": "west-lantern", "prefab": PREFAB, "transform": {"translation": [-4, 0, -2]}},
            {"id": "east-lantern", "prefab": PREFAB, "transform": {"translation": [4, 0, -2], "rotation_degrees": [0, -20, 0]}},
        ],
    })
}

pub(super) fn run(session: &mut dyn SmokeSession) -> SmokeResult {
    let source = fixture();
    let (mut scene, loaded) = SceneFixture::load(session, &source)?;
    smoke_eq!(json!(loaded["rendering"]["invalid_transforms"]), json!(0), "fixture_transforms_valid");
    ask(session, "control", json!({"name": "pause"}))?;
    discovery(session)?;
    children(session)?;
    let initial = residency(session)?;
    let rendering = query(session, "assets", json!({}))?["rendering"].clone();
    let entities = total(session, json!({"limit": 1}))?;
    smoke_eq!(json!(rendering["invalid_transforms"]), json!(0), "initial_transforms_valid");
    motion(session)?;
    refusals(session, &mut scene, &source, &initial)?;
    let transformed = matrices(session, &mut scene, &source, &initial)?;
    instances(session, &mut scene, &transformed, &initial, (&rendering, entities))?;
    removal(session, &mut scene, &transformed, &initial, (&rendering, entities))?;
    reload(session, &mut scene, &source)?;
    smoke_eq!(residency(session)?, initial, "repeated_reload_residency");
    Ok(())
}

fn discovery(session: &mut dyn SmokeSession) -> SmokeResult {
    let listing = ask(session, "describe", json!({}))?;
    for name in ["Hierarchy", "Transform", "GlobalTransform"] {
        let listed = listing["types"]
            .as_array()
            .is_some_and(|types| types.iter().any(|kind| kind["name"] == name));
        smoke_check!(listed, "descriptor_discovery", json!(name), json!("listed"));
        let described = ask(session, "describe", json!({"name": name}))?;
        smoke_check!(
            described["descriptor"]["schema"].is_object(),
            "descriptor_schema",
            json!(described["descriptor"]),
            json!({"schema": {}})
        );
    }
    let schema = ask(session, "describe", json!({"name": "Hierarchy"}))?["descriptor"]["schema"].clone();
    for field in [
        "parent",
        "parent_scene",
        "parent_alive",
        "parent_in_active_scene",
        "authored_parent",
        "prefab_id",
        "prefab_instance",
        "prefab_local_id",
    ] {
        smoke_check!(
            schema["properties"][field].is_object(),
            "hierarchy_schema_fields",
            json!(field),
            json!("described")
        );
    }
    Ok(())
}

fn children(session: &mut dyn SmokeSession) -> SmokeResult {
    let first = ask(
        session,
        "inspect",
        json!({"parent": "west-lantern", "component": "Hierarchy", "limit": 1}),
    )?;
    smoke_check!(
        first["total"] == 2 && first["entities"].as_array().map(Vec::len) == Some(1) && first["next_offset"] == 1,
        "direct_children_pagination",
        json!({"total": first["total"], "next_offset": first["next_offset"]}),
        json!({"total": 2, "next_offset": 1})
    );
    let second = ask(
        session,
        "inspect",
        json!({"parent": "west-lantern", "component": "Hierarchy", "limit": 1, "offset": 1}),
    )?;
    smoke_check!(
        second["total"] == 2 && second["next_offset"].is_null(),
        "direct_children_last_page",
        json!({"total": second["total"], "next_offset": second["next_offset"]}),
        json!({"total": 2, "next_offset": null})
    );
    let mut rows: Vec<Value> = [&first, &second]
        .into_iter()
        .flat_map(|page| page["entities"].as_array().cloned().unwrap_or_default())
        .collect();
    rows.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let ids: Vec<Value> = rows.iter().map(|row| row["id"].clone()).collect();
    smoke_eq!(json!(ids), json!(["west-lantern.lantern", "west-lantern.planter"]), "direct_children");
    for row in &rows {
        let hierarchy = &row["components"]["Hierarchy"];
        let local = hierarchy["prefab_local_id"].as_str().unwrap_or_default();
        smoke_eq!(
            json!({
                "parent": hierarchy["parent"], "parent_scene": hierarchy["parent_scene"],
                "parent_alive": hierarchy["parent_alive"], "parent_in_active_scene": hierarchy["parent_in_active_scene"],
                "authored_parent": hierarchy["authored_parent"], "prefab_id": hierarchy["prefab_id"],
                "prefab_instance": hierarchy["prefab_instance"], "id": row["id"],
            }),
            json!({
                "parent": "west-lantern", "parent_scene": SCENE, "parent_alive": true, "parent_in_active_scene": true,
                "authored_parent": "west-lantern", "prefab_id": PREFAB, "prefab_instance": "west-lantern",
                "id": format!("west-lantern.{local}"),
            }),
            "prefab_metadata"
        );
    }
    let selector = refuse(session, "inspect", json!({"entity": "player", "parent": "player"}))?;
    smoke_eq!(json!(selector["code"]), json!("invalid_selector"), "invalid_selector");
    let carried = ask(session, "inspect", json!({"parent": "player", "limit": 10}))?;
    let carries = carried["entities"]
        .as_array()
        .is_some_and(|rows| rows.iter().any(|row| row["id"] == "player-accessory"));
    smoke_check!(carries, "player_children", json!(carried["entities"]), json!(["player-accessory"]));
    let root = component(session, "west-lantern", "Hierarchy")?;
    smoke_check!(
        root["parent"].is_null() && root["parent_alive"].is_null() && root["parent_in_active_scene"].is_null(),
        "root_hierarchy",
        json!(root),
        json!({"parent": null, "parent_alive": null, "parent_in_active_scene": null})
    );
    let placement = ask(session, "inspect", json!({"entity": "west-lantern"}))?;
    let components = &placement["entity"]["components"];
    smoke_check!(
        components.as_array().is_some_and(|names| !names.iter().any(|name| name == "Mesh3d")),
        "placement_root_draws_nothing",
        json!(components),
        json!("no Mesh3d")
    );
    Ok(())
}

fn motion(session: &mut dyn SmokeSession) -> SmokeResult {
    let local = component(session, "player-accessory", "Transform")?;
    let player_before = component(session, "player", "GlobalTransform")?;
    let accessory_before = component(session, "player-accessory", "GlobalTransform")?;
    let tick_before = tick(session)?;
    let world = session.app().world_mut();
    let player = engine_core::authored_entity(world, "player");
    let moved = player.is_some_and(|player| {
        edit_transform(world, player, |transform| {
            transform.translation += Vec3::new(0.0, 0.0, -0.65);
        })
        .is_ok()
    });
    smoke_check!(moved, "player_moves", json!(player.is_some()), json!(true));
    session.step(6);
    smoke_eq!(json!(tick(session)? - tick_before), json!(6), "exact_step");
    let player_after = component(session, "player", "GlobalTransform")?;
    let accessory_after = component(session, "player-accessory", "GlobalTransform")?;
    let delta = |after: &Value, before: &Value| -> Vec<f64> {
        numbers(&after["translation"])
            .iter()
            .zip(numbers(&before["translation"]))
            .map(|(after, before)| after - before)
            .collect()
    };
    let player_delta = delta(&player_after, &player_before);
    let accessory_delta = delta(&accessory_after, &accessory_before);
    let length = player_delta.iter().map(|value| value * value).sum::<f64>().sqrt();
    smoke_check!(length > 0.1, "player_displaced", json!(player_delta), json!({"length": 0.1}));
    smoke_check!(
        near_all(&json!(accessory_delta), &player_delta, EPSILON),
        "exact_step_parent_motion",
        json!(accessory_delta),
        json!(player_delta)
    );
    smoke_eq!(
        component(session, "player-accessory", "Transform")?,
        local,
        "unchanged_local_transform"
    );
    Ok(())
}

fn refusals(
    session: &mut dyn SmokeSession,
    scene: &mut SceneFixture,
    source: &Value,
    initial: &Value,
) -> SmokeResult {
    let player = component(session, "player", "Transform")?;
    let accessory = component(session, "player-accessory", "GlobalTransform")?;
    for (kind, code) in [
        ("cycle", "hierarchy_cycle"),
        ("missing_parent", "unknown_parent"),
        ("qualified_id_collision", "duplicate_id"),
        ("parented_body", "invalid_rigid_body"),
    ] {
        let mut broken = source.clone();
        let entities = &mut broken["entities"];
        match kind {
            "cycle" => entities[3]["parent"] = json!("west-lantern.lantern"),
            "missing_parent" => entities[2]["parent"] = json!("absent-parent"),
            "parented_body" => {
                entities[2]["collider"] = json!({"shape": {"kind": "cuboid", "half_extents": [0.5, 0.5, 0.5]}});
                entities[2]["rigid_body"] = json!({});
            }
            _ => {
                if let Some(list) = entities.as_array_mut() {
                    list.push(json!({"id": "west-lantern.lantern"}));
                }
            }
        }
        let refused = scene.reload(session, &broken).err().unwrap_or_default();
        smoke_eq!(json!(refused["code"]), json!(code), "invalid_reload_code");
        smoke_check!(
            refused["path"].as_str().is_some_and(|path| path.starts_with('$')),
            "invalid_reload_path",
            json!(refused),
            json!({"path": "$"})
        );
        smoke_eq!(component(session, "player", "Transform")?, player.clone(), "invalid_reload_preservation");
        smoke_eq!(
            component(session, "player-accessory", "GlobalTransform")?,
            accessory.clone(),
            "invalid_reload_keeps_children"
        );
        smoke_eq!(residency(session)?, initial.clone(), "invalid_reload_keeps_assets");
    }
    Ok(())
}

fn matrices(
    session: &mut dyn SmokeSession,
    scene: &mut SceneFixture,
    source: &Value,
    initial: &Value,
) -> Result<Value, SmokeFailure> {
    reload(session, scene, source)?;
    let original = capture(session)?;
    let east: Vec<String> = ["east-lantern".to_owned()]
        .into_iter()
        .chain(PARTS.iter().map(|part| format!("east-lantern.{part}")))
        .collect();
    let mut east_before = Vec::new();
    for id in &east {
        east_before.push(component(session, id, "GlobalTransform")?);
    }
    let mut transformed = source.clone();
    transformed["entities"][3]["transform"] = json!({
        "translation": [-6, 0.25, -4], "rotation_degrees": [0, 37, 0], "scale": [1.6, 0.8, 0.65],
    });
    reload(session, scene, &transformed)?;
    let root = local_matrix(&component(session, "west-lantern", "Transform")?);
    for part in PARTS {
        let id = format!("west-lantern.{part}");
        let child = local_matrix(&component(session, &id, "Transform")?);
        let global = component(session, &id, "GlobalTransform")?;
        let expected: Vec<f64> = (root * child)
            .to_cols_array()
            .iter()
            .map(|&value| f64::from(value))
            .collect();
        smoke_eq!(json!(global["valid"]), json!(true), "world_matrix_valid");
        smoke_check!(
            near_all(&global["matrix"], &expected, EPSILON),
            "full_world_matrix_with_shear",
            json!(global["matrix"]),
            json!(expected)
        );
        let matrix = numbers(&global["matrix"]);
        smoke_check!(
            matrix.len() == 16 && near_all(&global["translation"], &matrix[12..15], EPSILON),
            "world_translation",
            json!(global["translation"]),
            json!(matrix.get(12..15))
        );
    }
    let shear = numbers(&component(session, "west-lantern.lantern", "GlobalTransform")?["matrix"]);
    let dot = shear.get(0..3).zip(shear.get(4..7)).map_or(0.0, |(x, y)| {
        x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>()
    });
    smoke_check!(dot.abs() > 0.01, "fixture_exercises_shear", json!(dot), json!({"more_than": 0.01}));
    for (id, before) in east.iter().zip(east_before) {
        smoke_eq!(component(session, id, "GlobalTransform")?, before, "independent_instances");
    }
    smoke_eq!(residency(session)?, initial.clone(), "transform_keeps_assets");
    let moved = capture(session)?;
    smoke_check!(moved != original, "transform_changes_image", json!(moved), json!({"not": original}));
    Ok(transformed)
}

fn instances(
    session: &mut dyn SmokeSession,
    scene: &mut SceneFixture,
    transformed: &Value,
    initial: &Value,
    (rendering, entities): (&Value, u64),
) -> SmokeResult {
    let mut triple = transformed.clone();
    if let Some(list) = triple["entities"].as_array_mut() {
        list.push(json!({"id": "third-lantern", "prefab": PREFAB,
            "transform": {"translation": [0, 0, -7], "scale": [0.7, 0.7, 0.7]}}));
    }
    let parts = PARTS.len() as u64;
    reload(session, scene, &triple)?;
    smoke_eq!(
        json!(total(session, json!({"parent": "third-lantern", "component": "Hierarchy", "limit": 10}))?),
        json!(parts),
        "third_placement_children"
    );
    smoke_eq!(json!(total(session, json!({"limit": 1}))?), json!(entities + parts + 1), "third_placement_entities");
    for replacement in 0..4 {
        if replacement > 0 {
            reload(session, scene, &triple)?;
        }
        smoke_eq!(residency(session)?, initial.clone(), "shared_assets");
        let assets = query(session, "assets", json!({}))?["rendering"].clone();
        smoke_eq!(
            json!([assets["draw_calls"], assets["instances"]]),
            json!([rendering["draw_calls"], rendering["instances"].as_u64().unwrap_or(0) + parts]),
            "shared_assets_and_draws"
        );
        if replacement > 0 {
            let warm = ask(session, "control", json!({"name": "render"}))?["rendering"].clone();
            smoke_eq!(
                json!([warm["mesh_upload_bytes"], warm["texture_upload_bytes"], warm["invalid_transforms"]]),
                json!([0, 0, 0]),
                "warm_upload_reuse"
            );
        }
    }
    capture(session)?;
    Ok(())
}

fn removal(
    session: &mut dyn SmokeSession,
    scene: &mut SceneFixture,
    transformed: &Value,
    initial: &Value,
    (rendering, entities): (&Value, u64),
) -> SmokeResult {
    let parts = PARTS.len() as u64;
    reload(session, scene, transformed)?;
    smoke_eq!(residency(session)?, initial.clone(), "instance_removal_assets");
    smoke_eq!(json!(total(session, json!({"limit": 1}))?), json!(entities), "instance_removal");
    let gone = refuse(session, "inspect", json!({"entity": "third-lantern"}))?;
    smoke_eq!(json!(gone["code"]), json!("unknown_entity"), "instance_removal_unknown");
    let mut removed = transformed.clone();
    if let Some(list) = removed["entities"].as_array_mut() {
        list.retain(|entity| entity["id"] != "west-lantern" && entity["id"] != "east-lantern");
    }
    reload(session, scene, &removed)?;
    let retired = query(session, "assets", json!({}))?;
    let drawn = &retired["rendering"];
    smoke_eq!(
        json!(drawn["instances"]),
        json!(rendering["instances"].as_u64().unwrap_or(0).saturating_sub(parts * 2)),
        "removed_instances"
    );
    smoke_eq!(
        json!(total(session, json!({"limit": 1}))?),
        json!(entities.saturating_sub((parts + 1) * 2)),
        "removed_entities"
    );
    for id in ["west-lantern", "west-lantern.lantern", "east-lantern.planter"] {
        let gone = refuse(session, "inspect", json!({"entity": id}))?;
        smoke_eq!(json!(gone["code"]), json!("unknown_entity"), "removed_unknown");
    }
    smoke_eq!(json!(retired["resident_cpu_bytes"]), json!(initial["cpu"]), "removed_cpu_assets_stay");
    smoke_check!(
        drawn["resident_meshes"].as_u64() < initial["gpu_meshes"].as_u64()
            && drawn["resident_mesh_bytes"].as_u64() < initial["mesh_gpu"].as_u64(),
        "unused_gpu_retirement",
        json!(drawn),
        json!({"fewer_than": initial})
    );
    smoke_eq!(
        json!([drawn["resident_textures"], drawn["resident_texture_bytes"], drawn["invalid_transforms"]]),
        json!([0, 0, 0]),
        "removed_textures_and_transforms"
    );
    capture(session)?;
    Ok(())
}

fn reload(
    session: &mut dyn SmokeSession,
    scene: &mut SceneFixture,
    document: &Value,
) -> Result<Value, SmokeFailure> {
    let reloaded = scene.reload(session, document);
    smoke_check!(reloaded.is_ok(), "reload_accepted", json!(reloaded.as_ref().err()), json!(null));
    let rendering = reloaded.unwrap_or_default()["rendering"].clone();
    smoke_eq!(json!(rendering["invalid_transforms"]), json!(0), "reload_transforms_valid");
    Ok(rendering)
}

fn total(session: &mut dyn SmokeSession, selector: Value) -> Result<u64, SmokeFailure> {
    Ok(ask(session, "inspect", selector)?["total"].as_u64().unwrap_or(0))
}

fn tick(session: &mut dyn SmokeSession) -> Result<u64, SmokeFailure> {
    let status = query(session, "status", json!({}))?;
    Ok(status["observation"]["tick"]
        .as_str()
        .and_then(|tick| tick.parse().ok())
        .unwrap_or(0))
}

fn local_matrix(transform: &Value) -> Mat4 {
    let vector = |key: &str| {
        let values = numbers(&transform[key]);
        Vec3::new(
            values.first().copied().unwrap_or(0.0) as f32,
            values.get(1).copied().unwrap_or(0.0) as f32,
            values.get(2).copied().unwrap_or(0.0) as f32,
        )
    };
    let rotation = numbers(&transform["rotation"]);
    let rotation = Quat::from_xyzw(
        rotation.first().copied().unwrap_or(0.0) as f32,
        rotation.get(1).copied().unwrap_or(0.0) as f32,
        rotation.get(2).copied().unwrap_or(0.0) as f32,
        rotation.get(3).copied().unwrap_or(1.0) as f32,
    );
    Mat4::from_scale_rotation_translation(vector("scale"), rotation, vector("translation"))
}
