use std::any::TypeId;
use std::collections::{BTreeMap, BTreeSet};

use engine::Engine;
use engine::bevy_ecs::prelude::World;
use engine::bevy_ecs::schedule::graph::Direction;
use engine::bevy_ecs::schedule::{NodeId, Schedules, SystemSet};
use engine::bevy_ecs::system::System;
use engine::bricks::{BrickRecord, BrickRegistry, BrickSet};
use engine_render::faces::face_of;
use engine_render::linker::face_imports;
use engine_render::pieces::{PieceDecl, place};
use engine_render::render_bricks::{RenderBrick, render_bricks};
use serde_json::{Value, json};

use crate::workspace::joined;
use crate::{Package, Repository};

pub const FILE: &str = "bricks.json";

pub struct Recorded {
    pub name: String,
    pub krate: String,
    pub reads: Vec<String>,
    pub paths: Vec<String>,
    pub smokes: Vec<(String, bool)>,
    pub parts: Vec<String>,
}

pub fn recorded(repository: &Repository) -> Result<Vec<Recorded>, String> {
    let path = repository.root.join(FILE);
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("cannot read {FILE}: {error}"))?;
    let file: Value =
        serde_json::from_str(&text).map_err(|error| format!("{FILE} is not JSON: {error}"))?;
    let strings = |value: &Value| -> Vec<String> {
        value
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()
    };
    Ok(file["bricks"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|brick| Recorded {
            name: brick["name"].as_str().unwrap_or_default().to_owned(),
            krate: brick["crate"].as_str().unwrap_or_default().to_owned(),
            reads: strings(&brick["reads"]),
            paths: strings(&brick["paths"]),
            smokes: brick["smokes"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|smoke| {
                    let name = smoke["name"].as_str()?.to_owned();
                    Some((name, smoke["gpu"] == true))
                })
                .collect(),
            parts: strings(&brick["parts"]),
        })
        .collect())
}

pub fn brick_crates<'a>(
    repository: &Repository,
    crates: impl IntoIterator<Item = &'a str>,
) -> BTreeSet<String> {
    crates
        .into_iter()
        .filter(|name| {
            repository
                .of_crate(name)
                .is_some_and(|package| !package.core())
        })
        .map(str::to_owned)
        .collect()
}

pub fn registered_crates(engine: &Engine) -> Vec<&'static str> {
    engine
        .app
        .world()
        .get_resource::<BrickRegistry>()
        .into_iter()
        .flat_map(BrickRegistry::bricks)
        .map(|brick| brick.crate_name)
        .collect()
}

fn dependencies(repository: &Repository, package: &Package) -> Vec<String> {
    let mut dependencies: Vec<String> = package
        .dependencies
        .iter()
        .filter(|dependency| dependency.kind == crate::DependencyKind::Normal)
        .filter(|dependency| {
            repository
                .package(&dependency.package)
                .is_some_and(|known| known.member || !known.engine.is_null())
        })
        .map(|dependency| dependency.package.clone())
        .collect();
    dependencies.sort();
    dependencies.dedup();
    dependencies
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

fn paths(brick: &BrickRecord, package: &Package) -> Vec<String> {
    let Some(folder) = package.path.as_deref() else {
        return Vec::new();
    };
    if !package.core() {
        return vec![joined(folder, "**")];
    }
    let path = brick.type_path.split('<').next().unwrap_or(brick.type_path);
    let segments: Vec<&str> = path.split("::").collect();
    match segments.get(1..segments.len().saturating_sub(1)) {
        Some(modules) if !modules.is_empty() => {
            let module = modules.join("/");
            vec![
                joined(folder, &format!("src/{module}.rs")),
                joined(folder, &format!("src/{module}/**")),
            ]
        }
        Some(_) | None => vec![joined(folder, "src/lib.rs")],
    }
}

struct Rendered {
    faces: Vec<&'static str>,
    needs: Vec<Value>,
    defaults: Vec<&'static str>,
    pieces: Vec<Value>,
    parts: Vec<&'static str>,
}

impl Rendered {
    fn of(render: Option<&RenderBrick>) -> Self {
        let mut faces: Vec<&'static str> = Vec::new();
        for face in render.into_iter().flat_map(RenderBrick::faces) {
            if !faces.contains(&face) {
                faces.push(face);
            }
        }
        let parts = render.map_or(&[][..], |render| render.parts.as_slice());
        Self {
            needs: needs(render, &faces),
            defaults: render
                .into_iter()
                .flat_map(RenderBrick::defaults)
                .filter_map(|module| face_of(module.path))
                .collect(),
            pieces: parts
                .iter()
                .filter_map(|part| part.decl.piece)
                .map(|piece| json!({ "name": piece.name, "size": piece.size }))
                .collect(),
            parts: parts.iter().map(|part| part.decl.name).collect(),
            faces,
        }
    }
}

fn needs(render: Option<&RenderBrick>, offered: &[&str]) -> Vec<Value> {
    let Some(render) = render else {
        return Vec::new();
    };
    let mut needs: Vec<(String, bool)> = Vec::new();
    for module in render.modules() {
        for face in face_imports(module.source) {
            if offered.contains(&face.as_str()) || needs.iter().any(|(known, _)| *known == face) {
                continue;
            }
            let default = render
                .defaults()
                .any(|default| face_of(default.path) == Some(face.as_str()));
            needs.push((face, default));
        }
    }
    for part in &render.parts {
        for &name in part.decl.needs {
            if !needs.iter().any(|(known, _)| known == name) {
                needs.push((name.to_owned(), false));
            }
        }
    }
    needs
        .into_iter()
        .map(|(face, default)| json!({ "face": face, "default": default }))
        .collect()
}

fn slots(registry: &BrickRegistry) -> Vec<Value> {
    let mut names: Vec<&'static str> = Vec::new();
    for slot in registry.bricks().iter().filter_map(|brick| brick.slot) {
        if !names.contains(&slot) {
            names.push(slot);
        }
    }
    names
        .into_iter()
        .map(|slot| {
            json!({
                "name": slot,
                "filler": registry.filler(slot).map(|brick| brick.name),
                "fillers": registry.fillers(slot),
            })
        })
        .collect()
}

fn lists(registry: &BrickRegistry) -> Vec<Value> {
    let mut names: Vec<&'static str> = Vec::new();
    for &list in registry.bricks().iter().flat_map(|brick| &brick.lists) {
        if !names.contains(&list) {
            names.push(list);
        }
    }
    names
        .into_iter()
        .map(|list| {
            let members: Vec<&str> = registry
                .bricks()
                .iter()
                .filter(|brick| brick.lists.contains(&list))
                .map(|brick| brick.name)
                .collect();
            json!({ "name": list, "members": members })
        })
        .collect()
}

pub fn pieces(registry: &BrickRegistry) -> Vec<(&'static str, PieceDecl)> {
    render_bricks(registry)
        .flat_map(|(_, render)| render.parts.iter())
        .filter_map(|part| Some((part.decl.brick, part.decl.piece?)))
        .collect()
}

fn frame(registry: &BrickRegistry) -> Result<Value, String> {
    let pieces = pieces(registry);
    let (placed, size) = place(&pieces).map_err(|error| error.to_string())?;
    let listed: Vec<Value> = pieces
        .iter()
        .zip(&placed)
        .map(|((brick, piece), at)| {
            json!({ "name": piece.name, "brick": brick, "offset": at.offset, "size": at.size })
        })
        .collect();
    Ok(json!({ "pieces": listed, "size": size }))
}

pub fn installed_passes(registry: &BrickRegistry) -> Vec<&'static str> {
    engine_render::parts::installed_parts(registry)
        .iter()
        .flat_map(|part| part.decl.passes)
        .map(|pass| pass.name)
        .collect()
}

fn record(
    brick: &BrickRecord,
    registry: &BrickRegistry,
    repository: &Repository,
    types: Option<&BTreeSet<TypeId>>,
) -> Result<Value, String> {
    let package = repository.of_crate(brick.crate_name).ok_or_else(|| {
        format!(
            "the {} brick's crate {} is no package of the workspace",
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
    let rendered = Rendered::of(brick.extension::<RenderBrick>());
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
        "faces": rendered.faces,
        "needs": rendered.needs,
        "defaults": rendered.defaults,
        "pieces": rendered.pieces,
        "parts": rendered.parts,
        "passes": brick.passes,
        "smokes": brick
            .smokes
            .iter()
            .map(|smoke| json!({ "name": smoke.name, "gpu": smoke.gpu }))
            .collect::<Vec<_>>(),
        "paths": paths(brick, package),
        "system_set": brick.name,
        "resources": brick.resources.iter().map(|owned| owned.name).collect::<Vec<_>>(),
        "components": brick.components.iter().map(|owned| owned.name).collect::<Vec<_>>(),
        "reads": reads,
        "dependencies": dependencies(repository, package),
    }))
}

pub fn bricks_json(repository: &Repository, engine: &mut Engine) -> Result<String, String> {
    let world = engine.app.world_mut();
    let names: Vec<&'static str> = world
        .get_resource::<BrickRegistry>()
        .map(|registry| registry.names().collect())
        .unwrap_or_default();
    let accessed = accessed(world, &names);
    let Some(registry) = world.get_resource::<BrickRegistry>() else {
        return Ok(serde_json::to_string_pretty(&json!({ "bricks": [] }))
            .map_err(|error| format!("cannot write {FILE}: {error}"))?
            + "\n");
    };
    registry
        .check_passes(&installed_passes(registry))
        .map_err(|error| error.to_string())?;
    let bricks = registry
        .bricks()
        .iter()
        .map(|brick| record(brick, registry, repository, accessed.get(brick.name)))
        .collect::<Result<Vec<_>, _>>()?;
    let text = serde_json::to_string_pretty(&json!({
        "bricks": bricks,
        "slots": slots(registry),
        "lists": lists(registry),
        "frame": frame(registry)?,
    }))
    .map_err(|error| format!("cannot write {FILE}: {error}"))?;
    Ok(text + "\n")
}
