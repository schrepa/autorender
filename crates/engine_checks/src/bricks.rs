use bevy_ecs::world::World;
use engine::Engine;
use engine_core::{BrickList, PlaceDecl, ScenePasses, Smoke};
use engine_place::PlaceKind;
use engine_registry::{BrickRecord, ReadKind, Registry};
use engine_render::Installation;
use serde_json::{Value, json};

use crate::install::List;
use crate::{Package, Repository};

pub const FILE: &str = "bricks.json";

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

fn smokes(smokes: &[Smoke]) -> Vec<Value> {
    smokes
        .iter()
        .map(|smoke| json!({ "name": smoke.name }))
        .collect()
}

fn kind(decl: &PlaceDecl) -> &'static str {
    match decl.kind() {
        PlaceKind::Slot => "slot",
        PlaceKind::List => "list",
        PlaceKind::Component => "component",
        PlaceKind::Ledger => "ledger",
    }
}

fn record(
    brick: &BrickRecord,
    registry: &Registry,
    repository: &Repository,
    (listed, installed): (&BrickList, &Installation),
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
        "reads": registry
            .reads()
            .iter()
            .filter(|read| read.reader == brick.package)
            .map(|read| json!({ "kind": read.kind, "place": read.place, "providers": read.providers }))
            .collect::<Vec<_>>(),
        "smokes": smokes(brick.smokes()),
        "dependencies": repository
            .package(brick.package)
            .map(|package| dependencies(repository, package))
            .unwrap_or_default(),
    })
}

fn places(registry: &Registry) -> Vec<Value> {
    registry
        .catalog()
        .iter()
        .map(|decl| {
            let used = registry.place_uses(decl);
            json!({
                "name": decl.name(),
                "kind": kind(decl),
                "filler": registry.filler_named(decl.name()),
                "members": registry.members(decl.type_id()).collect::<Vec<_>>(),
                "contract": decl.wgsl().is_some(),
                "smokes": smokes(decl.smokes()),
                "writers": used.writers,
                "readers": used.readers,
                "face_readers": registry
                    .reads()
                    .iter()
                    .filter(|read| read.kind == ReadKind::Face && read.place == decl.name())
                    .map(|read| read.reader)
                    .collect::<Vec<_>>(),
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
    let world = engine.app.world();
    let kind_passes = world
        .get_resource::<ScenePasses>()
        .map(ScenePasses::kind_table)
        .unwrap_or_default();
    let registry = registry(world)?;
    let listed = list(BrickList::default());
    let bricks: Vec<Value> = registry
        .bricks()
        .iter()
        .map(|brick| record(brick, registry, repository, (&listed, &installed)))
        .collect();
    let text = serde_json::to_string_pretty(&json!({
        "bricks": bricks,
        "places": self::places(registry),
        "left_out": registry.left_out(),
        "unmet": registry.unmet(),
        "kind_passes": kind_passes,
        "ownership": registry.ownership(),
        "frame": frame(&installed),
    }))
    .map_err(|error| format!("cannot write {FILE}: {error}"))?;
    Ok(text + "\n")
}
