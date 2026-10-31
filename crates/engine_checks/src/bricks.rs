use std::any::TypeId;
use std::collections::{BTreeMap, BTreeSet};

use engine::Engine;
use engine::bevy_ecs::prelude::World;
use engine::bevy_ecs::schedule::graph::Direction;
use engine::bevy_ecs::schedule::{NodeId, Schedules, SystemSet};
use engine::bevy_ecs::system::System;
use engine::bricks::{
    Absent, BrickRecord, BrickRegistry, BrickSet, DetailConsumer, DetailConsumers, PlaceKind,
};
use engine::scene::SceneComponents;
use engine_render::linker::face_imports;
use engine_render::pieces::{PieceDecl, place};
use engine_render::places::{Answer, contract, place_module};
use engine_render::render_bricks::{RenderBrick, render_bricks};
use serde_json::{Value, json};

use crate::workspace::joined;
use crate::{Package, Repository};

type Smokes = Vec<(String, bool)>;

pub const FILE: &str = "bricks.json";

pub struct Recorded {
    pub name: String,
    pub krate: String,
    pub reads: Vec<String>,
    pub paths: Vec<String>,
    pub smokes: Smokes,
    pub place_smokes: Smokes,
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
    let smokes = |value: &Value| -> Smokes {
        value
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|smoke| {
                let name = smoke["name"].as_str()?.to_owned();
                Some((name, smoke["gpu"] == true))
            })
            .collect()
    };
    let places: Vec<(Vec<String>, Smokes)> = file["places"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|place| (strings(&place["members"]), smokes(&place["smokes"])))
        .collect();
    Ok(file["bricks"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|brick| {
            let name = brick["name"].as_str().unwrap_or_default().to_owned();
            let place_smokes = places
                .iter()
                .filter(|(members, _)| members.contains(&name))
                .flat_map(|(_, held)| held.iter().cloned())
                .collect();
            Recorded {
                krate: brick["crate"].as_str().unwrap_or_default().to_owned(),
                reads: strings(&brick["reads"]),
                paths: strings(&brick["paths"]),
                smokes: smokes(&brick["smokes"]),
                place_smokes,
                parts: strings(&brick["parts"]),
                name,
            }
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
    needs: Vec<Value>,
    pieces: Vec<Value>,
    parts: Vec<&'static str>,
}

impl Rendered {
    fn of(brick: &BrickRecord, registry: &BrickRegistry) -> Self {
        let render = brick.extension::<RenderBrick>();
        let parts = render.map_or(&[][..], |render| render.parts.as_slice());
        Self {
            needs: needs(brick, registry, render),
            pieces: parts
                .iter()
                .filter_map(|part| part.decl.piece)
                .map(|piece| json!({ "name": piece.name, "size": piece.size }))
                .collect(),
            parts: parts.iter().map(|part| part.decl.name).collect(),
        }
    }
}

fn answers_empty(registry: &BrickRegistry, name: &str) -> bool {
    registry.place(name).is_some_and(|place| match place.kind {
        PlaceKind::Slot => place.face.absent == Absent::Nobody,
        PlaceKind::List => true,
    })
}

fn needs(
    brick: &BrickRecord,
    registry: &BrickRegistry,
    render: Option<&RenderBrick>,
) -> Vec<Value> {
    let Some(render) = render else {
        return Vec::new();
    };
    let held: Vec<&str> = brick
        .fills
        .iter()
        .map(|filled| filled.place)
        .chain(brick.joins.iter().map(|joined| joined.place))
        .collect();
    let mut needs: Vec<String> = Vec::new();
    for module in render.modules() {
        for face in face_imports(module.source) {
            if !held.contains(&face.as_str()) && !needs.contains(&face) {
                needs.push(face);
            }
        }
    }
    for part in &render.parts {
        for &name in part.decl.needs {
            if !needs.iter().any(|known| known == name) {
                needs.push(name.to_owned());
            }
        }
    }
    needs
        .into_iter()
        .map(|face| {
            let default = answers_empty(registry, &face);
            json!({ "face": face, "default": default })
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

pub(crate) fn place_face(place: &engine_core::Place) -> Value {
    let face = &place.face;
    let wgsl = face.wgsl.map(|source| match contract(source) {
        Ok(found) => json!({
            "module": place_module(place.name),
            "types": found.types,
            "answers": found
                .shared
                .iter()
                .chain(&found.answers)
                .map(Answer::signature)
                .collect::<Vec<_>>(),
            "absent": face.absent.name(),
        }),
        Err(error) => json!({ "error": error }),
    });
    json!({
        "cpu": face.cpu.map(|side| (side.name)()),
        "gpu": face.gpu.map(|side| (side.name)()),
        "wgsl": wgsl,
    })
}

fn places(registry: &BrickRegistry) -> Vec<Value> {
    registry
        .places()
        .into_iter()
        .map(|place| {
            let kind = match place.kind {
                PlaceKind::Slot => "slot",
                PlaceKind::List => "list",
            };
            json!({
                "name": place.name,
                "kind": kind,
                "members": registry.members(place.name),
                "face": place_face(&place),
                "smokes": place
                    .smokes
                    .iter()
                    .map(|smoke| json!({ "name": smoke.name, "gpu": smoke.gpu }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect()
}

type Kind = (&'static str, &'static str);

type Consumer = (&'static str, DetailConsumer);

fn record(
    brick: &BrickRecord,
    registry: &BrickRegistry,
    repository: &Repository,
    types: Option<&BTreeSet<TypeId>>,
    (kinds, consumers): (&[Kind], &[Consumer]),
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
    let rendered = Rendered::of(brick, registry);
    Ok(json!({
        "name": brick.name,
        "type": brick.type_path,
        "crate": brick.crate_name,
        "version": version,
        "fills": brick
            .fills
            .iter()
            .map(|filled| json!({
                "place": filled.place,
                "cpu": filled.cpu,
                "gpu": filled.gpu.map(|offer| offer.name()),
                "wgsl": filled.wgsl,
            }))
            .collect::<Vec<_>>(),
        "joins": brick
            .joins
            .iter()
            .map(|joined| json!({
                "place": joined.place,
                "cpu": joined.cpu,
                "wgsl": joined.wgsl.map(|member| json!({
                    "module": member.module,
                    "functions": member.functions,
                    "reads": member.reads,
                })),
            }))
            .collect::<Vec<_>>(),
        "consumers": consumers
            .iter()
            .filter(|(owner, _)| *owner == brick.name)
            .map(|(_, consumer)| json!({
                "name": consumer.name,
                "setting": consumer.setting,
                "stat": consumer.stat,
            }))
            .collect::<Vec<_>>(),
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
        "needs": rendered.needs,
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
        "components": kinds
            .iter()
            .filter(|&&(owner, _)| owner == brick.name)
            .map(|&(_, kind)| kind)
            .collect::<Vec<_>>(),
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
    let kinds: Vec<(&'static str, &'static str)> = world
        .get_resource::<SceneComponents>()
        .map(|held| {
            held.kinds()
                .iter()
                .map(|&(brick, kind)| (brick, kind.name))
                .collect()
        })
        .unwrap_or_default();
    let consumers: Vec<(&'static str, DetailConsumer)> = world
        .get_resource::<DetailConsumers>()
        .map(|held| {
            held.consumers()
                .map(|(brick, consumer)| (brick, *consumer))
                .collect()
        })
        .unwrap_or_default();
    let bricks = registry
        .bricks()
        .iter()
        .map(|brick| {
            record(
                brick,
                registry,
                repository,
                accessed.get(brick.name),
                (&kinds, &consumers),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let text = serde_json::to_string_pretty(&json!({
        "bricks": bricks,
        "places": places(registry),
        "frame": frame(registry)?,
    }))
    .map_err(|error| format!("cannot write {FILE}: {error}"))?;
    Ok(text + "\n")
}
