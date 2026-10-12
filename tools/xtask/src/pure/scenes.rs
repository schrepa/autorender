//! The scenes corpus: scene documents through the format's round trip,
//! and its refusals. Each document (the tracked fixtures and the corpus's
//! own, tools/xtask/pure/scenes.json) parses and validates, serializes,
//! and parses back to the same document. Each refusal fails with its
//! rule's error code at its path, while parsing, validating against its
//! files or instantiating into a world. Every error code the scene format
//! answers with (each `ErrorCode::*` engine_scene names) has a refusal, or
//! is one [`UNREACHED`] lists.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use engine::bevy_ecs::world::World;
use engine::{AssetBudget, AssetLibrary, Engine, EngineConfig, ErrorCode};
use engine_scene::{MAX_SOURCE_BYTES, SceneDocument, SceneError, SceneFiles};
use serde_json::{Value, json};

use super::Observations;

/// The corpus's own documents and refusals.
const FIXTURES: &str = "tools/xtask/pure/scenes.json";

/// Error codes engine_scene names that no document reaches, with why.
/// Sorted; a code a refusal reaches leaves it.
const UNREACHED: &[(&str, &str)] = &[];

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let fixtures: Value = serde_json::from_str(&read(root, FIXTURES)?)
        .map_err(|error| format!("{FIXTURES} is not JSON: {error}"))?;
    for (name, document) in fixtures["documents"]
        .as_object()
        .ok_or_else(|| format!("{FIXTURES} has no documents"))?
    {
        // A string names a tracked document by its path.
        let source = match document.as_str() {
            Some(path) => read(root, path)?,
            None => document.to_string(),
        };
        round_trip(name, &source, observed);
    }
    let mut refused = BTreeSet::new();
    for refusal in fixtures["refusals"]
        .as_array()
        .ok_or_else(|| format!("{FIXTURES} has no refusals"))?
    {
        let rule = refusal["rule"].as_str().unwrap_or_default();
        let path = refusal["path"].as_str().unwrap_or_default();
        let name = format!("refusal {rule} at {path}");
        let generated = (refusal["chain"].as_u64(), refusal["prefab"].as_u64());
        let source = match (&refusal["source"], generated) {
            // A source one byte over the bound.
            (Value::String(source), _) if source == "OVERSIZED" => " ".repeat(MAX_SOURCE_BYTES + 1),
            (Value::String(source), _) => source.clone(),
            // A parent chain of that many entities.
            (_, (Some(length), _)) => chain(length),
            // A prefab of that many entities.
            (_, (_, Some(count))) => prefab(count),
            _ => refusal["document"].to_string(),
        };
        let stage = refusal["stage"].as_str().unwrap_or("parse");
        let error = match refuse(&source, stage, refusal["world"].as_str()) {
            Ok(()) => {
                observed.check(false, || format!("{name}: the document was accepted"));
                continue;
            }
            Err(error) => error,
        };
        let code = error.code.as_str();
        observed.check(code == rule && error.path == path, || {
            format!(
                "{name}: expected {rule} at {path}, found {code} at {}: {}",
                error.path, error.message
            )
        });
        refused.insert(code.to_owned());
        observed.record(
            name,
            json!({"code": code, "path": error.path, "message": error.message}),
        );
    }
    // Every error code the format answers with has a refusal, or is listed
    // as reached by no document.
    let rules = scene_rules(root)?;
    for rule in &rules {
        let unreached = UNREACHED.iter().any(|(code, _)| code == rule);
        observed.check(refused.contains(rule) != unreached, || {
            if unreached {
                format!("{rule} is in UNREACHED, but a refusal reaches it")
            } else {
                format!("no refusal in {FIXTURES} answers {rule}, which engine_scene names")
            }
        });
    }
    observed.check(
        UNREACHED.windows(2).all(|pair| pair[0].0 < pair[1].0)
            && UNREACHED.iter().all(|(code, _)| rules.contains(*code)),
        || "UNREACHED must be sorted and name only codes engine_scene names".into(),
    );
    Ok(())
}

/// A document of `length` entities, each the parent of the next.
fn chain(length: u64) -> String {
    let entities: Vec<Value> = (0..length)
        .map(|index| match index {
            0 => json!({"id": "e0"}),
            _ => json!({"id": format!("e{index}"), "parent": format!("e{}", index - 1)}),
        })
        .collect();
    json!({"id": "chain", "revision": 1, "materials": [], "meshes": [], "entities": entities})
        .to_string()
}

/// A document whose one prefab holds `count` entities.
fn prefab(count: u64) -> String {
    let entities: Vec<Value> = (0..count)
        .map(|index| json!({"id": format!("e{index}")}))
        .collect();
    json!({
        "id": "prefab",
        "revision": 1,
        "materials": [],
        "meshes": [],
        "prefabs": [{"id": "large", "entities": entities}],
        "entities": [],
    })
    .to_string()
}

fn read(root: &Path, path: &str) -> Result<String, String> {
    fs::read_to_string(root.join(path)).map_err(|error| format!("cannot read {path}: {error}"))
}

/// Parses and validates `source`, serializes it, and parses that back:
/// the second serialization equals the first. Observes the serialized
/// document's hash and size.
fn round_trip(name: &str, source: &str, observed: &mut Observations) {
    let written = SceneDocument::from_json(source).and_then(|document| {
        let first = document.to_json()?;
        let second = SceneDocument::from_json(&first)?.to_json()?;
        Ok((document, first, second))
    });
    match written {
        Ok((document, first, second)) => {
            observed.check(first == second, || {
                format!("document {name}: serializing its parse changes it")
            });
            observed.hash(format!("document {name}"), first.as_bytes());
            observed.record(
                format!("document {name} counts"),
                json!({
                    "entities": document.entities.len(),
                    "streamed": document.streamed.len(),
                    "prefabs": document.prefabs.len(),
                    "materials": document.materials.len(),
                    "meshes": document.meshes.len(),
                    "textures": document.textures.len(),
                    "bytes": first.len(),
                }),
            );
        }
        Err(error) => observed.check(false, || {
            format!(
                "document {name} is refused: {} at {}: {}",
                error.code.as_str(),
                error.path,
                error.message
            )
        }),
    }
}

/// `source` through `stage`: `parse` (parse and validate), `files`
/// (validate against no files) or `instantiate` (into `world`).
fn refuse(source: &str, stage: &str, world: Option<&str>) -> Result<(), SceneError> {
    let document = SceneDocument::from_json(source)?;
    match stage {
        "files" => document.validate_with(&SceneFiles::new()),
        "instantiate" => {
            let mut engine;
            let mut bare;
            let world: &mut World = match world {
                // A world without an asset library.
                Some("bare") => {
                    bare = World::new();
                    &mut bare
                }
                // An asset library holding no meshes.
                Some("no meshes") => {
                    bare = World::new();
                    bare.insert_resource(AssetLibrary::new(AssetBudget {
                        max_meshes: 0,
                        ..AssetBudget::default()
                    }));
                    &mut bare
                }
                // An engine of one transform node.
                Some("one node") => {
                    engine = Engine::new(EngineConfig {
                        max_transform_nodes: 1,
                        ..EngineConfig::default()
                    });
                    engine.app.world_mut()
                }
                // An engine without its features' plugins.
                _ => {
                    engine = Engine::default();
                    engine.app.world_mut()
                }
            };
            document.instantiate(world).map(drop)
        }
        _ => Ok(()),
    }
}

/// The error codes engine_scene answers with: each `ErrorCode::*` its
/// sources name, by code.
fn scene_rules(root: &Path) -> Result<BTreeSet<String>, String> {
    let source = root.join("crates/engine_scene/src");
    let mut named = BTreeSet::new();
    for entry in
        fs::read_dir(&source).map_err(|error| format!("cannot read engine_scene: {error}"))?
    {
        let path = entry.map_err(|error| error.to_string())?.path();
        let text = fs::read_to_string(&path).map_err(|error| error.to_string())?;
        for (at, _) in text.match_indices("ErrorCode::") {
            let rest = &text[at + "ErrorCode::".len()..];
            let end = rest
                .find(|c: char| !c.is_alphanumeric())
                .unwrap_or(rest.len());
            named.insert(rest[..end].to_owned());
        }
    }
    let mut rules = BTreeSet::new();
    for variant in named {
        let code = ErrorCode::ALL
            .iter()
            .find(|(code, _, _)| format!("{code:?}") == variant)
            .map(|(_, name, _)| (*name).to_owned())
            .ok_or_else(|| format!("engine_scene names ErrorCode::{variant}, which is no code"))?;
        rules.insert(code);
    }
    Ok(rules)
}
