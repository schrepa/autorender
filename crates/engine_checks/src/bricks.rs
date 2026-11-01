use std::any::TypeId;
use std::collections::BTreeMap;

use bevy_ecs::query::ComponentAccessKind;
use bevy_ecs::schedule::graph::Direction;
use bevy_ecs::schedule::{NodeId, Schedules, SystemSet};
use bevy_ecs::system::System;
use bevy_ecs::world::World;
use engine::Engine;
use engine_core::{BrickList, PlaceDecl, Runs, Smoke};
use engine_place::PlaceKind;
use engine_registry::{BrickRecord, BrickSet, Registry};
use engine_render::Installation;
use serde_json::{Value, json};

use crate::install::List;
use crate::{Package, Repository};

pub const FILE: &str = "bricks.json";

type Scheduled = Vec<(&'static str, String)>;

#[derive(Default)]
struct Uses {
    writers: Scheduled,
    readers: Scheduled,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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

fn registry(world: &World) -> Result<&Registry, String> {
    world
        .get_resource::<Registry>()
        .ok_or_else(|| "the engine holds no brick registry".to_owned())
}

fn uses(
    world: &mut World,
    packages: &[&'static str],
    slots: &[TypeId],
) -> BTreeMap<TypeId, Uses> {
    let labels: Vec<_> = world
        .resource::<Schedules>()
        .iter()
        .map(|(_, schedule)| schedule.label())
        .collect();
    let mut uses: BTreeMap<TypeId, Uses> = BTreeMap::new();
    for label in labels {
        let Some(mut schedule) = world.resource_mut::<Schedules>().remove(label) else {
            continue;
        };
        let named = format!("{label:?}");
        let graph = schedule.graph_mut();
        for &package in packages {
            let Some(set) = graph.system_sets.get_key(BrickSet(package).intern()) else {
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
                let Ok(kinds) = access.combined_access().try_iter_access() else {
                    continue;
                };
                for kind in kinds {
                    let Some(type_id) = world
                        .components()
                        .get_info(*kind.index())
                        .and_then(|info| info.type_id())
                        .filter(|type_id| slots.contains(type_id))
                    else {
                        continue;
                    };
                    let used = uses.entry(type_id).or_default();
                    let side = match kind {
                        ComponentAccessKind::Exclusive(_) => &mut used.writers,
                        ComponentAccessKind::Shared(_) | ComponentAccessKind::Archetypal(_) => {
                            &mut used.readers
                        }
                    };
                    if !side
                        .iter()
                        .any(|(brick, at)| *brick == package && *at == named)
                    {
                        side.push((package, named.clone()));
                    }
                }
            }
        }
        world.resource_mut::<Schedules>().insert(schedule);
    }
    uses
}

fn smokes(smokes: &[Smoke]) -> Vec<Value> {
    smokes
        .iter()
        .map(|smoke| json!({ "name": smoke.name, "gpu": smoke.runs == Runs::Gpu }))
        .collect()
}

fn scheduled(used: &Scheduled) -> Vec<Value> {
    used.iter()
        .map(|(brick, schedule)| json!({ "brick": brick, "schedule": schedule }))
        .collect()
}

fn kind(decl: &PlaceDecl) -> &'static str {
    match decl.kind() {
        PlaceKind::Slot => "slot",
        PlaceKind::List => "list",
        PlaceKind::Component => "component",
    }
}

fn record(
    brick: &BrickRecord,
    repository: &Repository,
    listed: &BrickList,
    installed: &Installation,
) -> Value {
    let jobs: Vec<&'static str> = listed
        .listed()
        .iter()
        .filter(|entry| entry.package() == brick.package)
        .flat_map(|entry| entry.jobs().iter().map(|kind| kind.name()))
        .collect();
    let parts: Vec<Value> = installed
        .parts
        .iter()
        .filter(|part| part.brick == brick.package)
        .map(|part| json!({ "part": part.part, "passes": part.passes }))
        .collect();
    json!({
        "name": brick.package,
        "type": brick.type_name(),
        "version": hex(&brick.source()),
        "fills": brick.fills().iter().map(|filled| filled.place.name()).collect::<Vec<_>>(),
        "joins": brick.joins().iter().map(|joined| joined.place.name()).collect::<Vec<_>>(),
        "settings": brick
            .settings_entries()
            .iter()
            .map(|entry| json!({ "key": entry.key, "type": entry.type_name }))
            .collect::<Vec<_>>(),
        "stats": brick
            .stats_entries()
            .iter()
            .map(|entry| json!({ "key": entry.key, "type": entry.type_name }))
            .collect::<Vec<_>>(),
        "operations": brick
            .operations()
            .iter()
            .map(|operation| json!({ "name": operation.name, "control": operation.control }))
            .collect::<Vec<_>>(),
        "piece": brick
            .piece_entry()
            .map(|piece| json!({ "name": piece.wgsl_name, "size": piece.size })),
        "resources": brick.owned().iter().map(|owned| owned.name).collect::<Vec<_>>(),
        "jobs": jobs,
        "parts": parts,
        "smokes": smokes(brick.smokes()),
        "dependencies": repository
            .package(brick.package)
            .map(|package| dependencies(repository, package))
            .unwrap_or_default(),
    })
}

fn places(registry: &Registry, uses: &BTreeMap<TypeId, Uses>) -> Vec<Value> {
    registry
        .catalog()
        .iter()
        .map(|decl| {
            let used = uses.get(&decl.type_id());
            json!({
                "name": decl.name(),
                "kind": kind(decl),
                "filler": registry.filler_named(decl.name()),
                "members": registry.members(decl.type_id()).collect::<Vec<_>>(),
                "contract": decl.wgsl().is_some(),
                "smokes": smokes(decl.smokes()),
                "writers": used.map(|used| scheduled(&used.writers)),
                "readers": used.map(|used| scheduled(&used.readers)),
            })
        })
        .collect()
}

fn frame(installed: &Installation) -> Value {
    json!({
        "pieces": installed
            .pieces
            .iter()
            .map(|piece| json!({
                "brick": piece.brick,
                "name": piece.name,
                "offset": piece.offset,
                "size": piece.size,
            }))
            .collect::<Vec<_>>(),
        "size": installed.arena,
        "layouts": installed.layouts,
        "frame_variants": installed.frame_variants,
    })
}

pub fn bricks_json(
    repository: &Repository,
    list: List,
    engine: &mut Engine,
) -> Result<String, String> {
    let installed = crate::install::noop(engine)?;
    let world = engine.app.world_mut();
    let (packages, slots) = {
        let registry = registry(world)?;
        let packages: Vec<&'static str> = registry.names().collect();
        let slots: Vec<TypeId> = registry
            .catalog()
            .iter()
            .filter(|decl| decl.kind() == PlaceKind::Slot)
            .map(|decl| decl.type_id())
            .collect();
        (packages, slots)
    };
    let uses = uses(world, &packages, &slots);
    let registry = registry(world)?;
    let listed = list(BrickList::default());
    let bricks: Vec<Value> = registry
        .bricks()
        .iter()
        .map(|brick| record(brick, repository, &listed, &installed))
        .collect();
    let text = serde_json::to_string_pretty(&json!({
        "bricks": bricks,
        "places": places(registry, &uses),
        "left_out": registry.left_out(),
        "frame": frame(&installed),
    }))
    .map_err(|error| format!("cannot write {FILE}: {error}"))?;
    Ok(text + "\n")
}
