//! Terrain's development tools: the inspectors of terrains, their streaming,
//! prefetch places and the development view.

use engine_core::TerrainDebug;
use engine_core::bevy_ecs::prelude::{Entity, World};
use serde_json::{Value, json};

use crate::inspect::Inspector;
use crate::views::TerrainDebugView;
use crate::{Context, DevTools, InspectorDescriptor, InspectorKind};

/// Registers terrain's inspectors.
pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools.register_component::<engine_core::TerrainPrefetch>(
        InspectorDescriptor::generated(schemas, "TerrainPrefetch")
            .with_default(json!(engine_core::TerrainPrefetch::default())),
        |prefetch| json!(prefetch),
    )
    .expect("built-in descriptor is bounded");
    tools.insert(Inspector {
        descriptor: InspectorDescriptor::generated(schemas, "Terrain"),
        kind: InspectorKind::Component,
        has: Box::new(|world, entity| {
            entity.is_some_and(|entity| world.get::<engine_core::Terrain>(entity).is_some())
        }),
        read: Box::new(terrain_view),
    })
    .expect("built-in descriptor is bounded");
    tools.register_resource::<TerrainDebug>(
        InspectorDescriptor::generated(schemas, "TerrainDebug"),
        |debug| {
            json!(TerrainDebugView {
                freeze: debug.freeze.map(|point| point.to_array()),
                view: debug.view,
            })
        },
    )
    .expect("built-in descriptor is bounded");
    tools.register_resource::<engine_core::Terrains>(
        InspectorDescriptor::generated(schemas, "Terrains"),
        |terrains| {
            let stats = terrains.stats();
            json!(crate::views::TerrainsView {
                terrains: stats.terrains,
                resident_tiles: stats.resident_tiles,
                loading_tiles: stats.loading_tiles,
                shaped_tiles: stats.shaped_tiles,
                waiting_tiles: stats.waiting_tiles,
                cached_tiles: stats.cached_tiles,
                cached_bytes: stats.cached_bytes,
                restored_tiles: stats.restored_tiles,
                idle_tiles: stats.idle_tiles,
                banded_tiles: stats.banded_tiles,
                unplaced_tiles: stats.unplaced_tiles,
                dropped_tiles: stats.dropped_tiles,
                patches: stats.patches,
                fallback_patches: stats.fallback_patches,
                baking: stats.baking,
                bridges: stats.bridges,
                tunnels: stats.tunnels,
                falls: stats.falls,
                surroundings_tiles: stats.surroundings_tiles,
                surroundings_patches: stats.surroundings_patches,
            })
        },
    )
    .expect("built-in descriptor is bounded");
}

/// Adds the schemas of terrain's inspectors.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert("TerrainPrefetch".into(), super::schema::<engine_core::TerrainPrefetch>());
    schemas.insert("Terrain".into(), super::schema::<crate::views::TerrainView>());
    schemas.insert("TerrainDebug".into(), super::schema::<crate::views::TerrainDebugView>());
    schemas.insert("Terrains".into(), super::schema::<crate::views::TerrainsView>());
}

/// The view of the terrain on `entity`: its tiles by level, its
/// streaming's errors and ranges, its bakes, paths and surroundings.
fn terrain_view(world: &World, entity: Option<Entity>, context: &Context<'_>) -> Option<Value> {
        let scene = context.scene;
        let entity = entity?;
        let terrain = world.get::<engine_core::Terrain>(entity)?;
        let mut levels = vec![0_u32; terrain.pyramid.levels() as usize];
        let (mut level_errors, mut level_ranges) = (Vec::new(), Vec::new());
        let (mut hydrology, mut erosion, mut paths) = (None, None, Vec::new());
        if let Some(terrains) = world.get_resource::<engine_core::Terrains>() {
            for tile in terrains.tiles(entity) {
                levels[usize::from(tile.key().level)] += 1;
            }
            if let Some((errors, ranges)) = terrains.level_detail(entity) {
                level_errors = errors.to_vec();
                level_ranges = ranges.to_vec();
            }
            let holds = |wanted: fn(&engine_core::Layer) -> bool| {
                matches!(&terrain.source, engine_core::TerrainSource::Generated(layers)
                    if layers.layers().iter().any(wanted))
            };
            let bakes = terrains.bakes(entity);
            if holds(|layer| matches!(layer, engine_core::Layer::Hydrology(_))) {
                let stats = bakes.hydrology;
                hydrology = Some(crate::views::HydrologyView {
                    baking: bakes.baking,
                    samples: stats.map(|stats| stats.samples),
                    channel_samples: stats.map(|stats| stats.channel_samples),
                    lake_samples: stats.map(|stats| stats.lake_samples),
                    max_area: stats.map(|stats| stats.max_area),
                    falls: stats.map(|stats| stats.falls),
                });
            }
            // Each path layer's profile, once baked.
            if let Some(layers) = terrains.layers(entity) {
                for (index, layer) in layers.layers().iter().enumerate() {
                    if let engine_core::Layer::Path(stage) = layer {
                        let stats = stage.baked.as_ref().map(|profile| profile.stats);
                        paths.push(crate::views::PathView {
                            layer: index,
                            baked: stats.is_some(),
                            length: stats.map(|stats| stats.length),
                            samples: stats.map(|stats| stats.samples),
                            steepest: stats.map(|stats| stats.steepest),
                            deepest_cut: stats.map(|stats| stats.deepest_cut),
                            highest_fill: stats.map(|stats| stats.highest_fill),
                            bridges: stats.map(|stats| stats.bridges),
                            bridged: stats.map(|stats| stats.bridged),
                            tunnels: stats.map(|stats| stats.tunnels),
                            tunneled: stats.map(|stats| stats.tunneled),
                        });
                    }
                }
            }
            if holds(|layer| matches!(layer, engine_core::Layer::Erosion(_))) {
                let stats = bakes.erosion;
                erosion = Some(crate::views::ErosionView {
                    baking: bakes.baking,
                    samples: stats.map(|stats| stats.samples),
                    rounds: stats.map(|stats| stats.rounds),
                    deepest: stats.map(|stats| stats.deepest),
                    highest: stats.map(|stats| stats.highest),
                });
            }
        }
        Some(json!(crate::views::TerrainView {
            terrain: engine_scene::SceneTerrain::from_runtime(
                terrain,
                None,
                |handle| {
                    scene.and_then(|scene| {
                        scene
                            .materials()
                            .find(|(_, candidate)| *candidate == handle)
                            .map(|(id, _)| id.to_owned())
                    })
                },
                |handle| {
                    scene.and_then(|scene| {
                        scene
                            .meshes()
                            .find(|(_, candidate)| *candidate == handle)
                            .map(|(id, _)| id.to_owned())
                    })
                }
            ),
            levels: terrain.pyramid.levels(),
            extent: terrain.pyramid.extent().to_array(),
            resident_tiles_by_level: levels,
            level_errors,
            level_ranges,
            hydrology,
            erosion,
            paths,
            edits: terrain
                .edits
                .iter()
                .map(|edit| crate::views::TerrainEditView {
                    id: edit.id.to_string(),
                    stamp: edit.stamp,
                })
                .collect(),
            surroundings: world
                .get_resource::<engine_core::Terrains>()
                .and_then(|terrains| terrains.surroundings(entity))
                .map(|status| crate::views::SurroundingsView {
                    reach: status.reach,
                    tiles: status.tiles,
                    spacing: status.spacing,
                    resident_tiles: status.resident_tiles,
                    patches: status.patches,
                }),
        }))
}
