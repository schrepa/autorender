use std::any::TypeId;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use engine::bevy_ecs::prelude::World;
use engine::bevy_ecs::schedule::graph::Direction;
use engine::bevy_ecs::schedule::{NodeId, Schedules, SystemSet};
use engine::bevy_ecs::system::System;
use engine::bricks::{AddBrick, BrickRecord, BrickRegistry, BrickSet};
use serde_json::{Value, json};

pub const BRICKS_FILE: &str = "bricks.json";

pub fn standard_engine() -> engine::Engine {
    let mut engine = engine::Engine::new(engine::EngineConfig::default());
    engine.app.add_bricks(engine::bricks::standard());
    engine
}

pub fn environment_sections(world: &World) -> Vec<(&'static str, &'static str)> {
    world
        .get_resource::<BrickRegistry>()
        .into_iter()
        .flat_map(BrickRegistry::sections)
        .map(|section| (section.key, section.name))
        .collect()
}

struct Package {
    folder: PathBuf,
    dependencies: Vec<String>,
}

fn packages(root: &Path) -> Result<BTreeMap<String, Package>, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps", "--locked"])
        .current_dir(root)
        .output()
        .map_err(|error| format!("cannot run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let metadata: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("cargo metadata is not JSON: {error}"))?;
    let members: BTreeSet<String> = metadata["packages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|package| package["name"].as_str().map(str::to_owned))
        .collect();
    let mut packages = BTreeMap::new();
    for package in metadata["packages"].as_array().into_iter().flatten() {
        let Some(name) = package["name"].as_str() else {
            continue;
        };
        let folder = package["manifest_path"]
            .as_str()
            .and_then(|manifest| Path::new(manifest).parent())
            .map(Path::to_path_buf)
            .ok_or_else(|| format!("cargo metadata gives {name} no manifest"))?;
        let mut dependencies: Vec<String> = package["dependencies"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|dependency| dependency["kind"].is_null())
            .filter_map(|dependency| dependency["name"].as_str())
            .filter(|dependency| members.contains(*dependency))
            .map(str::to_owned)
            .collect();
        dependencies.sort();
        dependencies.dedup();
        packages.insert(
            name.to_owned(),
            Package {
                folder,
                dependencies,
            },
        );
    }
    Ok(packages)
}

fn accessed(world: &mut World, names: &[&'static str]) -> BTreeMap<&'static str, BTreeSet<TypeId>> {
    let labels: Vec<_> = world
        .resource::<Schedules>()
        .iter()
        .map(|(_, schedule)| schedule.label())
        .collect();
    let mut accessed: BTreeMap<&'static str, BTreeSet<TypeId>> = BTreeMap::new();
    for label in labels {
        let Some(mut schedule) = world.resource_mut::<Schedules>().remove(label) else {
            continue;
        };
        let graph = schedule.graph_mut();
        for &name in names {
            let Some(set) = graph.system_sets.get_key(BrickSet(name).intern()) else {
                continue;
            };
            let systems: Vec<_> = graph
                .hierarchy()
                .graph()
                .neighbors_directed(NodeId::Set(set), Direction::Outgoing)
                .filter_map(|node| match node {
                    NodeId::System(key) => Some(key),
                    NodeId::Set(_) => None,
                })
                .collect();
            for key in systems {
                let Some(system) = graph.systems.get_mut(key) else {
                    continue;
                };
                let access = system.initialize(world);
                let Ok(ids) = access.combined_access().try_iter_access() else {
                    continue;
                };
                let types = accessed.entry(name).or_default();
                for kind in ids {
                    if let Some(type_id) = world
                        .components()
                        .get_info(*kind.index())
                        .and_then(|info| info.type_id())
                    {
                        types.insert(type_id);
                    }
                }
            }
        }
        world.resource_mut::<Schedules>().insert(schedule);
    }
    accessed
}

fn record(
    brick: &BrickRecord,
    registry: &BrickRegistry,
    packages: &BTreeMap<String, Package>,
    types: Option<&BTreeSet<TypeId>>,
) -> Result<Value, String> {
    let package = packages.get(brick.crate_name).ok_or_else(|| {
        format!(
            "the {} brick's crate {} is no workspace package",
            brick.name, brick.crate_name
        )
    })?;
    let version: String = engine_sources::crate_source(&package.folder)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let reads: BTreeSet<&str> = types
        .into_iter()
        .flatten()
        .filter_map(|&type_id| registry.owner_of(type_id))
        .map(|owner| owner.name)
        .filter(|&owner| owner != brick.name)
        .collect();
    Ok(json!({
        "name": brick.name,
        "type": brick.type_path,
        "crate": brick.crate_name,
        "version": version,
        "slot": brick.slot,
        "lists": brick.lists,
        "records": brick.records,
        "settings": brick
            .settings
            .iter()
            .map(|section| json!({
                "key": section.key,
                "name": section.name,
                "opt_in": section.opt_in,
            }))
            .collect::<Vec<_>>(),
        "stats": brick.stats.iter().map(|stats| stats.key).collect::<Vec<_>>(),
        "operations": brick
            .operations
            .iter()
            .map(|operation| json!({
                "name": operation.name,
                "control": operation.control,
            }))
            .collect::<Vec<_>>(),
        "inspectors": brick.inspectors.iter().map(|inspector| inspector.name).collect::<Vec<_>>(),
        "system_set": brick.name,
        "resources": brick.resources.iter().map(|owned| owned.name).collect::<Vec<_>>(),
        "components": brick.components.iter().map(|owned| owned.name).collect::<Vec<_>>(),
        "reads": reads,
        "dependencies": package.dependencies,
    }))
}

pub fn bricks_json(root: &Path) -> Result<String, String> {
    let packages = packages(root)?;
    let mut engine = standard_engine();
    let world = engine.app.world_mut();
    let names: Vec<&'static str> = world.resource::<BrickRegistry>().names().collect();
    let accessed = accessed(world, &names);
    let registry = world.resource::<BrickRegistry>();
    let bricks = registry
        .bricks()
        .iter()
        .map(|brick| record(brick, registry, &packages, accessed.get(brick.name)))
        .collect::<Result<Vec<_>, _>>()?;
    let text = serde_json::to_string_pretty(&json!({ "bricks": bricks }))
        .map_err(|error| format!("cannot write {BRICKS_FILE}: {error}"))?;
    Ok(text + "\n")
}
