mod controls;
mod queries;
mod results;

use engine_core::bevy_ecs;
use engine_core::bevy_ecs::prelude::{Entity, Resource, World};
use engine_core::{Engine, TerrainDebug};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
#[cfg(not(target_arch = "wasm32"))]
use engine_settings::{rule::same, stated};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::EngineControl;
use crate::inspect::Inspector;
#[cfg(not(target_arch = "wasm32"))]
use crate::protocol::stamp_rule;
use crate::results::result;
use crate::views::TerrainDebugView;
use crate::{Context, DevError, DevTools, InspectorDescriptor, InspectorKind};
use controls::{TerrainEditing, draw_river, edit_terrain};
pub use queries::TerrainKey;
pub(crate) use queries::terrain_key;
pub use queries::{TerrainHeightQuery, TerrainRouteQuery, TerrainSweepQuery};
pub use results::{
    HeightSample, Reach, RouteFound, StandingWater, SweepExample, SweepLayers, SweepRays,
    TerrainEdited, TerrainHeightResult, TerrainRouteResult, TerrainSweepResult, Worst,
};

pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
    tools
        .register_component::<engine_core::TerrainPrefetch>(
            InspectorDescriptor::generated(schemas, "TerrainPrefetch")
                .with_default(json!(engine_core::TerrainPrefetch::default())),
            |prefetch| json!(prefetch),
        )
        .expect("built-in descriptor is bounded");
    tools
        .insert(Inspector {
            descriptor: InspectorDescriptor::generated(schemas, "Terrain"),
            kind: InspectorKind::Component,
            has: Box::new(|world, entity| {
                entity.is_some_and(|entity| world.get::<engine_core::Terrain>(entity).is_some())
            }),
            read: Box::new(terrain_view),
        })
        .expect("built-in descriptor is bounded");
    tools
        .register_resource::<TerrainDebug>(
            InspectorDescriptor::generated(schemas, "TerrainDebug"),
            |debug| {
                json!(TerrainDebugView {
                    freeze: debug.freeze.map(|point| point.to_array()),
                })
            },
        )
        .expect("built-in descriptor is bounded");
    tools
        .register_resource::<engine_core::Terrains>(
            InspectorDescriptor::generated(schemas, "Terrains"),
            |terrains| {
                let stats = terrains.stats();
                let (failed_jobs, latest_job_failure) = terrains.job_failures();
                json!(crate::views::TerrainsView {
                    terrains: stats.terrains,
                    resident_tiles: stats.resident_tiles,
                    loading_tiles: stats.loading_tiles,
                    waiting_tiles: stats.waiting_tiles,
                    cached_tiles: stats.cached_tiles,
                    cached_bytes: stats.cached_bytes,
                    restored_tiles: stats.restored_tiles,
                    prefetching_tiles: stats.prefetching_tiles,
                    dropped_tiles: stats.dropped_tiles,
                    fallback_tiles: stats.fallback_tiles,
                    baking: stats.baking,
                    bridges: stats.bridges,
                    tunnels: stats.tunnels,
                    falls: stats.falls,
                    surroundings_tiles: stats.surroundings_tiles,
                    water: terrains
                        .waters()
                        .next()
                        .and_then(|(entity, origin, layers)| {
                            let found = layers.hydrology()?;
                            let held = terrains.river_flow(entity);
                            let world = |point: [f32; 3]| {
                                [
                                    point[0] + origin.x,
                                    point[1] + origin.y,
                                    point[2] + origin.z,
                                ]
                            };
                            Some(crate::views::TerrainWaterView {
                                flow: river_flow(found),
                                lakes: lakes(found, origin),
                                sights: sights(found, origin),
                                blocks: held.map_or(0, |flow| flow.blocks().len()),
                                pools: held.map_or_else(Vec::new, |flow| pools(flow, origin)),
                                falls: found
                                    .falls
                                    .iter()
                                    .take(crate::views::FALLS_SHOWN)
                                    .map(|fall| crate::views::FallView {
                                        top: world(fall.top),
                                        bottom: world(fall.bottom),
                                        width: fall.width,
                                        speed: fall.speed,
                                        discharge: fall.discharge,
                                    })
                                    .collect(),
                            })
                        }),
                    failed_jobs,
                    latest_job_failure: latest_job_failure.map(str::to_owned),
                })
            },
        )
        .expect("built-in descriptor is bounded");
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
    schemas.insert(
        "TerrainPrefetch".into(),
        super::schema::<engine_core::TerrainPrefetch>(),
    );
    schemas.insert(
        "Terrain".into(),
        super::schema::<crate::views::TerrainView>(),
    );
    schemas.insert(
        "TerrainDebug".into(),
        super::schema::<crate::views::TerrainDebugView>(),
    );
    schemas.insert(
        "Terrains".into(),
        super::schema::<crate::views::TerrainsView>(),
    );
}

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
        let holds =
            |wanted: fn(&engine_core::Layer) -> bool| terrain.layers.layers().iter().any(wanted);
        let bakes = terrains.bakes(entity);
        if holds(|layer| matches!(layer, engine_core::Layer::Hydrology(_))) {
            hydrology = Some(hydrology_view(terrains, entity));
        }
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
            .get_resource::<engine_core::Terrains>()
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

fn hydrology_view(terrains: &engine_core::Terrains, entity: Entity) -> crate::views::HydrologyView {
    let bakes = terrains.bakes(entity);
    let stats = bakes.hydrology;
    let baked = terrains.layers(entity);
    let found = baked.as_deref().and_then(engine_core::Layers::hydrology);
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
        flow: found.map(river_flow),
        fall_list: found
            .map(|found| {
                found
                    .falls
                    .iter()
                    .take(crate::views::FALLS_SHOWN)
                    .map(|fall| crate::views::FallView {
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

fn pools(
    flow: &engine_core::terrain_data::RiverFlow,
    origin: engine_core::glam::Vec3,
) -> Vec<crate::views::PoolView> {
    flow.pools()
        .iter()
        .map(|pool| {
            let (point, rise) = flow.pool_place(pool);
            let (blocked, crest, area) = flow.holding(pool);
            crate::views::PoolView {
                at: [
                    point[0] + origin.x,
                    pool.level + origin.y,
                    point[1] + origin.z,
                ],
                rise,
                inflow: pool.inflow,
                outflow: pool.outflow,
                blocked,
                crest: crest.map(|crest| crest + origin.y),
                area,
            }
        })
        .collect()
}

const LAKES_SHOWN: usize = 8;

fn lakes(
    found: &engine_core::terrain_data::Hydrology,
    origin: engine_core::glam::Vec3,
) -> Vec<crate::views::LakeView> {
    found
        .lake_sights(LAKES_SHOWN)
        .into_iter()
        .map(|lake| {
            let at = |[x, z]: [f32; 2]| [x + origin.x, lake.level + origin.y, z + origin.z];
            crate::views::LakeView {
                level: lake.level + origin.y,
                area: lake.area,
                middle: at(lake.middle),
                shore: at(lake.shore),
            }
        })
        .collect()
}

fn sights(
    found: &engine_core::terrain_data::Hydrology,
    origin: engine_core::glam::Vec3,
) -> Vec<crate::views::RiverSight> {
    found
        .river_sights()
        .into_iter()
        .map(|sight| crate::views::RiverSight {
            kind: sight.kind,
            point: [
                sight.point[0] + origin.x,
                sight.point[1] + origin.y,
                sight.point[2] + origin.z,
            ],
            direction: sight.direction,
            half: sight.half,
            speed: sight.speed,
            discharge: sight.discharge,
        })
        .collect()
}

fn river_flow(found: &engine_core::terrain_data::Hydrology) -> crate::views::RiverFlowView {
    let flow = found.flow_summary();
    crate::views::RiverFlowView {
        rivers: flow.rivers,
        lakes: flow.lakes,
        points: flow.points,
        max_discharge: flow.max_discharge,
        fastest: flow.fastest,
        rapid_share: flow.rapid_share,
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainControl {
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::TerrainSettings::rule_of("subdivision_error")))))]
    pub subdivision_error: Option<f32>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::TerrainSettings::rule_of("pixel_error")))))]
    pub pixel_error: Option<f32>,
    #[serde(default)]
    pub freeze: Option<bool>,
}

fn terrain_setting(value: f32, name: &'static str) -> Result<f32, DevError> {
    let rule = engine_core::TerrainSettings::rule_of(name).expect("terrain settings declare it");
    engine_settings::Setting::check(&value, name, rule)
        .map(|()| value)
        .map_err(|invalid| {
            DevError::new(
                ErrorCode::OutOfRange,
                format!("$.{name}"),
                invalid.to_string(),
            )
        })
}

impl EngineControl for TerrainControl {
    fn changed(&self) -> &'static str {
        "terrain changed"
    }

    fn apply(&self, engine: &mut Engine, _: Option<&SceneInstance>) -> Result<Value, DevError> {
        let world = engine.app.world_mut();
        let subdivision_error = self
            .subdivision_error
            .map(|value| terrain_setting(value, "subdivision_error"))
            .transpose()?;
        let pixel_error = self
            .pixel_error
            .map(|value| terrain_setting(value, "pixel_error"))
            .transpose()?;
        if subdivision_error.is_some() || pixel_error.is_some() {
            let mut settings = world
                .get_resource::<engine_core::TerrainSettings>()
                .copied()
                .unwrap_or_default();
            if let Some(subdivision_error) = subdivision_error {
                settings.subdivision_error = subdivision_error;
            }
            if let Some(pixel_error) = pixel_error {
                settings.pixel_error = pixel_error;
            }
            world.insert_resource(settings);
        }
        let eye = world.resource::<engine_core::Camera>().eye();
        let mut debug = world
            .get_resource::<engine_core::TerrainDebug>()
            .copied()
            .unwrap_or_default();
        if let Some(freeze) = self.freeze {
            debug.freeze = freeze.then_some(debug.freeze.unwrap_or(eye));
        }
        world.insert_resource(debug);
        let settings = world.get_resource::<engine_core::TerrainSettings>();
        Ok(result::terrain(TerrainControlResult {
            temporary: true,
            terrain_changed: true,
            frozen_at: debug.freeze.map(|eye| eye.to_array()),
            subdivision_error: settings.map(|settings| settings.subdivision_error),
            pixel_error: settings.map(|settings| settings.pixel_error),
        }))
    }
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainControlResult {
    pub temporary: bool,
    pub terrain_changed: bool,
    pub frozen_at: Option<[f32; 3]>,
    pub subdivision_error: Option<f32>,
    pub pixel_error: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainEditControl {
    #[serde(default)]
    pub terrain: Option<String>,
    #[cfg_attr(
        not(target_arch = "wasm32"),
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
    )]
    pub id: String,
    pub center: [f64; 2],
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stamp_rule("radius")))]
    pub radius: f64,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stamp_rule("falloff")))]
    pub falloff: f64,
    pub change: engine_core::terrain_data::StampOp,
}

impl EngineControl for TerrainEditControl {
    fn changed(&self) -> &'static str {
        "terrain edited"
    }

    fn apply(&self, engine: &mut Engine, scene: Option<&SceneInstance>) -> Result<Value, DevError> {
        editing(
            engine,
            scene,
            self.terrain.as_deref(),
            TerrainEditing::Edit {
                id: self.id.clone(),
                center: self.center,
                radius: self.radius,
                falloff: self.falloff,
                change: self.change,
            },
        )
        .map(result::terrain_edit)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainRiverControl {
    #[serde(default)]
    pub terrain: Option<String>,
    #[cfg_attr(
        not(target_arch = "wasm32"),
        schemars(transform = engine_settings::stated(engine_settings::rule::id()))
    )]
    pub id: String,
    #[serde(default)]
    pub points: Vec<[f64; 2]>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::GivenRiver::rule_of("width")))))]
    pub width: Vec<f64>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::GivenRiver::rule_of("depth")))))]
    pub depth: Vec<f64>,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::GivenRiver::rule_of("discharge")))))]
    pub discharge: Vec<f64>,
    #[serde(default)]
    pub name: Option<String>,
}

impl EngineControl for TerrainRiverControl {
    fn changed(&self) -> &'static str {
        "terrain edited"
    }

    fn apply(&self, engine: &mut Engine, scene: Option<&SceneInstance>) -> Result<Value, DevError> {
        draw_river(engine, scene, self).map(result::terrain_river)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainUndoControl {
    #[serde(default)]
    pub terrain: Option<String>,
}

impl EngineControl for TerrainUndoControl {
    fn changed(&self) -> &'static str {
        "terrain edited"
    }

    fn apply(&self, engine: &mut Engine, scene: Option<&SceneInstance>) -> Result<Value, DevError> {
        editing(engine, scene, self.terrain.as_deref(), TerrainEditing::Undo)
            .map(result::terrain_undo)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainRedoControl {
    #[serde(default)]
    pub terrain: Option<String>,
}

impl EngineControl for TerrainRedoControl {
    fn changed(&self) -> &'static str {
        "terrain edited"
    }

    fn apply(&self, engine: &mut Engine, scene: Option<&SceneInstance>) -> Result<Value, DevError> {
        editing(engine, scene, self.terrain.as_deref(), TerrainEditing::Redo)
            .map(result::terrain_redo)
    }
}

#[derive(Resource, Default)]
struct TerrainRedo(Vec<(Entity, engine_core::TerrainEdit)>);

fn editing(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    name: Option<&str>,
    editing: TerrainEditing,
) -> Result<TerrainEdited, DevError> {
    let mut redo = engine
        .app
        .world_mut()
        .remove_resource::<TerrainRedo>()
        .unwrap_or_default();
    let result = edit_terrain(engine, scene, name, &mut redo.0, editing);
    engine.app.world_mut().insert_resource(redo);
    result
}
