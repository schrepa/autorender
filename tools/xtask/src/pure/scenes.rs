use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use bevy_ecs::world::World;
use engine::{AssetBudget, AssetLibrary, Engine, EngineConfig, Error, ErrorCode};
use engine_scene::{MAX_SOURCE_BYTES, SceneDocument, SceneFiles, SceneKinds};
use serde_json::{Value, json};

use super::Observations;

const FIXTURES: &str = "tools/xtask/pure/scenes.json";

const SCENE_SOURCES: &[&str] = &[
    "crates/engine_scene/src",
    "crates/engine_grounding/src/lib.rs",
    "crates/engine_splines/src/scene.rs",
    "crates/engine_splines/src/spline_mesh.rs",
    "crates/engine_world_streaming/src/scene.rs",
];

const UNREACHED: &[(&str, &str)] = &[];

pub fn run(root: &Path, observed: &mut Observations) -> Result<(), String> {
    let fixtures: Value = serde_json::from_str(&read(root, FIXTURES)?)
        .map_err(|error| format!("{FIXTURES} is not JSON: {error}"))?;
    let kinds = crate::bricks::standard_kinds().map_err(|error| error.to_string())?;
    for (name, document) in fixtures["documents"]
        .as_object()
        .ok_or_else(|| format!("{FIXTURES} has no documents"))?
    {
        let source = match document.as_str() {
            Some(path) => read(root, path)?,
            None => document.to_string(),
        };
        round_trip(name, &source, &kinds, observed);
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
            (Value::String(source), _) if source == "OVERSIZED" => " ".repeat(MAX_SOURCE_BYTES + 1),
            (Value::String(source), _) => source.clone(),
            (_, (Some(length), _)) => chain(length),
            (_, (_, Some(count))) => prefab(count),
            _ => refusal["document"].to_string(),
        };
        let stage = refusal["stage"].as_str().unwrap_or("parse");
        let skips = refusal["skipped"] == true;
        let (error, skipped) = match refuse(&source, stage, refusal["world"].as_str(), &kinds) {
            Ok(skipped) => match skipped.into_iter().next() {
                Some(skip) => (skip, true),
                None => {
                    observed.check(false, || format!("{name}: the document was accepted"));
                    continue;
                }
            },
            Err(error) => (error, false),
        };
        let code = error.code.as_str();
        observed.check(code == rule && error.path == path && skipped == skips, || {
            let answer = if skipped { "skipped" } else { "refused" };
            format!(
                "{name}: expected {rule} at {path}, found {code} at {} {answer}",
                error.path
            )
        });
        refused.insert(code.to_owned());
        observed.record(name, json!(error));
    }
    let rules = scene_rules(root)?;
    for rule in &rules {
        let unreached = UNREACHED.iter().any(|(code, _)| code == rule);
        observed.check(refused.contains(rule) != unreached, || {
            if unreached {
                format!("{rule} is in UNREACHED, but a refusal reaches it")
            } else {
                format!("no refusal in {FIXTURES} answers {rule}, which the scene loader names")
            }
        });
    }
    observed.check(
        UNREACHED.windows(2).all(|pair| pair[0].0 < pair[1].0)
            && UNREACHED.iter().all(|(code, _)| rules.contains(*code)),
        || "UNREACHED must be sorted and name only codes the scene loader names".into(),
    );
    Ok(())
}

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

fn round_trip(name: &str, source: &str, kinds: &SceneKinds, observed: &mut Observations) {
    let written = SceneDocument::from_json(source).and_then(|document| {
        document.validate(kinds)?;
        let first = document.to_json()?;
        let again = SceneDocument::from_json(&first)?;
        again.validate(kinds)?;
        let second = again.to_json()?;
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
                "document {name} is refused: {} at {}",
                error.code.as_str(),
                error.path
            )
        }),
    }
}

fn refuse(
    source: &str,
    stage: &str,
    world: Option<&str>,
    kinds: &SceneKinds,
) -> Result<Vec<Error>, Error> {
    let document = SceneDocument::from_json(source)?;
    document.validate(kinds)?;
    match stage {
        "files" => document
            .validate_with(kinds, &SceneFiles::new())
            .map(|()| Vec::new()),
        "instantiate" => {
            let mut engine;
            let mut bare;
            let world: &mut World = match world {
                Some("bare") => {
                    bare = World::new();
                    &mut bare
                }
                Some("no meshes") => {
                    bare = World::new();
                    bare.insert_resource(AssetLibrary::new(AssetBudget {
                        max_meshes: 0,
                        ..AssetBudget::default()
                    }));
                    &mut bare
                }
                Some("one node") => {
                    engine = Engine::new(EngineConfig {
                        max_transform_nodes: 1,
                        ..EngineConfig::default()
                    });
                    engine.app.world_mut()
                }
                Some("standard") => {
                    engine = crate::bricks::standard_engine()?;
                    engine.app.world_mut()
                }
                _ => {
                    engine = Engine::default();
                    engine.app.world_mut()
                }
            };
            document
                .instantiate(world)
                .map(|scene| scene.skipped().to_vec())
        }
        _ => Ok(Vec::new()),
    }
}

fn scene_rules(root: &Path) -> Result<BTreeSet<String>, String> {
    let mut paths = Vec::new();
    for listed in SCENE_SOURCES {
        let path = root.join(listed);
        if !path.is_dir() {
            paths.push(path);
            continue;
        }
        let entries =
            fs::read_dir(&path).map_err(|error| format!("cannot read {listed}: {error}"))?;
        for entry in entries {
            paths.push(entry.map_err(|error| error.to_string())?.path());
        }
    }
    let mut named = BTreeSet::new();
    for path in paths {
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
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
            .find(|(code, _)| format!("{code:?}") == variant)
            .map(|(_, name)| (*name).to_owned())
            .ok_or_else(|| {
                format!("the scene loader names ErrorCode::{variant}, which is no code")
            })?;
        rules.insert(code);
    }
    Ok(rules)
}
