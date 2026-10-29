use std::collections::{BTreeMap, BTreeSet};

use engine_core::smoke::{SmokeResult, SmokeSession};
use serde_json::{Value, json};

use super::requests::{accepted, control, names, spawn, spawned, status};

const BRICKS: &str = include_str!("../../../../bricks.json");
const FEATURES: [&str; 6] = [
    "terrain",
    "world_streaming",
    "navigation",
    "time_of_day",
    "collision",
    "physics",
];

type Reads = BTreeMap<String, Vec<String>>;

fn reads() -> Reads {
    let listed: Value = serde_json::from_str(BRICKS).unwrap_or_default();
    listed["bricks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|brick| Some((brick["name"].as_str()?.to_owned(), names(&brick["reads"]))))
        .collect()
}

fn needs(reads: &Reads, name: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let mut pending: Vec<String> = reads.get(name).cloned().unwrap_or_default();
    while let Some(next) = pending.pop() {
        if found.insert(next.clone()) {
            pending.extend(reads.get(&next).cloned().unwrap_or_default());
        }
    }
    found
}

fn core() -> Value {
    json!({
        "id": "plugins-fixture", "revision": 1,
        "camera": { "position": [0, 6, 12], "target": [0, 0, 0], "fov_degrees": 55, "near": 0.1, "far": 1000 },
        "materials": [{ "id": "plain", "color": [0.7, 0.6, 0.5, 1] }],
        "meshes": [{ "id": "box", "source": { "kind": "cube" } }],
        "entities": [{ "id": "floor", "mesh": "box", "material": "plain", "transform": { "translation": [0, -0.5, 0], "scale": [20, 1, 20] } }],
    })
}

fn ground() -> Value {
    json!({
        "id": "ground",
        "terrain": { "origin": [-128, 0, -128], "tiles": [2, 2], "spacing": 1, "layers": [{ "noise": { "seed": 3, "height": 8, "wavelength": 120 } }] },
    })
}

fn cuboid() -> Value {
    json!({ "shape": { "kind": "cuboid", "half_extents": [0.5, 0.5, 0.5] } })
}

fn author(scene: &mut Value, name: &str) -> &'static str {
    let has_ground = scene["entities"]
        .as_array()
        .is_some_and(|entities| entities.iter().any(|entity| entity["id"] == "ground"));
    let entities = &mut scene["entities"];
    let push = |entities: &mut Value, entity: Value| {
        if let Some(entities) = entities.as_array_mut() {
            entities.push(entity);
        }
    };
    match name {
        "terrain" => {
            push(entities, ground());
            "$.entities[1].terrain"
        }
        "world_streaming" => {
            scene["streamed"] = json!([{ "id": "crate", "mesh": "box", "material": "plain", "transform": { "translation": [4, 0.5, 0] } }]);
            "$.streamed"
        }
        "navigation" => {
            if !has_ground {
                push(entities, ground());
            }
            push(entities, json!({ "id": "walker", "nav_agent": {} }));
            "$.entities[2].nav_agent"
        }
        "time_of_day" => {
            scene["environment"] = json!({ "time_of_day": {} });
            "$.environment.time_of_day"
        }
        "collision" => {
            entities[0]["collider"] = cuboid();
            "$.entities[0].collider"
        }
        _ => {
            push(
                entities,
                json!({
                    "id": "body", "mesh": "box", "material": "plain", "transform": { "translation": [0, 2, 0] },
                    "collider": cuboid(), "rigid_body": {},
                }),
            );
            "$.entities[1].rigid_body"
        }
    }
}

fn authored(names: &[&str]) -> Value {
    let mut scene = core();
    for name in names {
        author(&mut scene, name);
    }
    scene
}

fn ray() -> Value {
    json!({ "origin": [0, 5, 0], "direction": [0, -1, 0], "max_distance": 20 })
}

fn request(name: &str) -> (&'static str, Value) {
    match name {
        "terrain" => ("query", json!({ "name": "terrain_height", "args": { "points": [[0, 0]] } })),
        "world_streaming" => ("inspect", json!({ "component": "WorldCells" })),
        "navigation" => ("inspect", json!({ "component": "NavAgents" })),
        "time_of_day" => ("inspect", json!({ "component": "TimeOfDay" })),
        "collision" => ("query", json!({ "name": "raycast", "args": ray() })),
        _ => ("inspect", json!({ "component": "PhysicsWorld" })),
    }
}

fn refusal(name: &str) -> (&'static str, Value) {
    match name {
        "world_streaming" => ("control", json!({ "name": "world_changes", "args": { "clear": true } })),
        "time_of_day" => ("control", json!({ "name": "environment", "args": { "time_of_day": {} } })),
        "physics" => ("control", json!({ "name": "environment", "args": { "physics": { "enabled": true } } })),
        _ => request(name),
    }
}

fn restarted(session: &mut dyn SmokeSession, without: &[&str], standard: &[&'static str]) -> SmokeResult {
    session.restart(without);
    control(session, "pause", json!({}), "pauses")?;
    let added: Vec<&str> = standard
        .iter()
        .copied()
        .filter(|name| !without.contains(name))
        .collect();
    let features = status(session, "status_answers")?["result"]["features"].clone();
    engine_core::smoke_eq!(
        json!({ "without": without, "features": names(&features) }),
        json!({ "without": without, "features": added }),
        "status_names_bricks_added"
    );
    Ok(())
}

fn left_out(session: &mut dyn SmokeSession, name: &str, reads: &Reads, standard: &[&'static str]) -> SmokeResult {
    let without: Vec<&str> = standard
        .iter()
        .copied()
        .filter(|&other| other == name || needs(reads, other).contains(name))
        .collect();
    restarted(session, &without, standard)?;
    control(session, "render", json!({}), "renders_without_brick")?;
    let (verb, body) = refusal(name);
    let refused = super::requests::reply(session, verb, body);
    engine_core::smoke_eq!(
        json!({ "brick": name, "code": refused["error"]["code"] }),
        json!({ "brick": name, "code": "missing_plugin" }),
        "missing_brick_refused"
    );
    let mut scene = core();
    let path = author(&mut scene, name);
    let authored = spawned(session, &scene).err().unwrap_or_default();
    engine_core::smoke_eq!(
        json!({ "brick": name, "code": authored["code"], "path": authored["path"] }),
        json!({ "brick": name, "code": "missing_plugin", "path": path }),
        "authoring_names_missing_brick"
    );
    Ok(())
}

fn alone(session: &mut dyn SmokeSession, name: &str, reads: &Reads, features: &[&str], standard: &[&'static str]) -> SmokeResult {
    let needed = needs(reads, name);
    let others: Vec<&str> = features
        .iter()
        .copied()
        .filter(|&other| other != name && !needed.contains(other))
        .collect();
    restarted(session, &others, standard)?;
    spawn(session, &authored(&[name]))?;
    control(session, "render", json!({}), "renders_with_brick")?;
    let (verb, body) = request(name);
    accepted(session, verb, body, "brick_answers")?;
    Ok(())
}

pub(crate) fn plugins(session: &mut dyn SmokeSession) -> SmokeResult {
    let reads = reads();
    let standard: Vec<&'static str> = crate::bricks::standard().names().collect();
    let features: Vec<&str> = standard
        .iter()
        .copied()
        .filter(|name| FEATURES.contains(name))
        .collect();
    engine_core::smoke_eq!(
        json!(features),
        json!(FEATURES),
        "every_feature_brick_authored"
    );
    for &name in &features {
        left_out(session, name, &reads, &standard)?;
        alone(session, name, &reads, &features, &standard)?;
    }
    restarted(session, &[], &standard)?;
    spawn(session, &authored(&features))?;
    control(session, "render", json!({}), "renders_every_brick")?;
    Ok(())
}
