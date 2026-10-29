use super::*;
use engine_terrain::{FALLS_SHOWN, FallView, RiverFlowView};

impl DevTools {
    pub(super) fn register_terrain(&mut self, schemas: &Value) {
        self.insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Terrain"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<engine_terrain::Terrain>(entity).is_some())
            }),
            read: Box::new(terrain_view),
            brick: Some("terrain"),
        })
        .expect("descriptor_bounded");
    }
}

fn terrain_view(world: &World, entity: Option<Entity>, context: &Context<'_>) -> Option<Value> {
    let scene = context.scene;
    let entity = entity?;
    let terrain = world.get::<engine_terrain::Terrain>(entity)?;
    let mut levels = vec![0_u32; terrain.pyramid.levels() as usize];
    let (mut level_errors, mut level_ranges) = (Vec::new(), Vec::new());
    let (mut hydrology, mut erosion, mut paths) = (None, None, Vec::new());
    if let Some(terrains) = world.get_resource::<engine_terrain::Terrains>() {
        for tile in terrains.tiles(entity) {
            levels[usize::from(tile.key().level)] += 1;
        }
        if let Some((errors, ranges)) = terrains.level_detail(entity) {
            level_errors = errors.to_vec();
            level_ranges = ranges.to_vec();
        }
        let holds =
            |wanted: fn(&engine_terrain::Layer) -> bool| terrain.layers.layers().iter().any(wanted);
        let bakes = terrains.bakes(entity);
        if holds(|layer| matches!(layer, engine_terrain::Layer::Hydrology(_))) {
            hydrology = Some(hydrology_view(terrains, entity));
        }
        if let Some(layers) = terrains.layers(entity) {
            for (index, layer) in layers.layers().iter().enumerate() {
                if let engine_terrain::Layer::Path(stage) = layer {
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
        if holds(|layer| matches!(layer, engine_terrain::Layer::Erosion(_))) {
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
        rivers: terrain
            .rivers
            .iter()
            .map(|drawn| crate::views::TerrainRiverView {
                id: drawn.id.to_string(),
                river: drawn.river.clone(),
            })
            .collect(),
        surroundings: world
            .get_resource::<engine_terrain::Terrains>()
            .and_then(|terrains| terrains.surroundings(entity))
            .map(|status| crate::views::SurroundingsView {
                reach: status.reach,
                tiles: status.tiles,
                spacing: status.spacing,
                resident_tiles: status.resident_tiles,
                drawn: status.drawn,
            }),
    }))
}

fn hydrology_view(
    terrains: &engine_terrain::Terrains,
    entity: Entity,
) -> crate::views::HydrologyView {
    let bakes = terrains.bakes(entity);
    let stats = bakes.hydrology;
    let baked = terrains.layers(entity);
    let found = baked.as_deref().and_then(engine_terrain::Layers::hydrology);
    crate::views::HydrologyView {
        baking: bakes.baking,
        samples: stats.map(|stats| stats.samples),
        channel_samples: stats.map(|stats| stats.channel_samples),
        lake_samples: stats.map(|stats| stats.lake_samples),
        max_area: stats.map(|stats| stats.max_area),
        falls: stats.map(|stats| stats.falls),
        given: stats.map(|stats| stats.given),
        uncut: stats.map(|stats| stats.uncut),
        rivers: baked
            .as_deref()
            .map(|layers| layers.river_listing(crate::views::RIVERS_SHOWN))
            .unwrap_or_default()
            .into_iter()
            .map(|river| crate::views::RiverView {
                name: river.name,
                given: river.given,
                courses: river.courses,
                length: river.length,
                first: river.first,
                last: river.last,
                width: river.width,
                depth: river.depth,
                flood: river.flood,
                given_width: river.given_width,
                given_depth: river.given_depth,
                given_flood: river.given_flood,
                uncut: river.uncut,
            })
            .collect(),
        flow: found.map(RiverFlowView::of),
        fall_list: found
            .map(|found| {
                found
                    .falls
                    .iter()
                    .take(FALLS_SHOWN)
                    .map(|fall| FallView {
                        top: fall.top,
                        bottom: fall.bottom,
                        width: fall.width,
                        speed: fall.speed,
                        discharge: fall.discharge,
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}
