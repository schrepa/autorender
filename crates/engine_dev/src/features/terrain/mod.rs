//! Terrain's development tools: the inspectors of terrains, their streaming,
//! prefetch places and the development view; the controls of the view and
//! of edits; and the queries of heights, sweeps and routes.

mod controls;
mod queries;

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
use controls::{TerrainEditing, edit_terrain};
pub use queries::TerrainKey;
pub(crate) use queries::terrain_key;
pub use queries::{TerrainHeightQuery, TerrainRouteQuery, TerrainSweepQuery};

/// Registers terrain's inspectors.
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
                    view: debug.view,
                })
            },
        )
        .expect("built-in descriptor is bounded");
    tools
        .register_resource::<engine_core::Terrains>(
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
                    failed_reads: stats.failed_reads,
                    latest_read_failure: terrains.latest_read_failure().map(str::to_owned),
                    refused_falls: stats.refused_falls,
                    latest_fall_refusal: terrains.latest_fall_refusal().map(str::to_owned),
                })
            },
        )
        .expect("built-in descriptor is bounded");
}

/// Adds the schemas of terrain's inspectors.
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

/// Temporarily set terrain's level of detail and view and
/// render while paused: `pixel_error` replaces the settings', `freeze`
/// keeps streaming and patches following the camera's
/// current position while the camera moves on, and `view` shows terrain
/// shaded or its levels, normals, tiles and patches, materials, sky
/// visibility or detail. Omitted fields stay as they
/// are.
#[derive(Debug, Deserialize, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct TerrainControl {
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), schemars(transform = stated(same(engine_core::TerrainSettings::rule_of("pixel_error")))))]
    pub pixel_error: Option<f32>,
    #[serde(default)]
    pub freeze: Option<bool>,
    #[serde(default)]
    pub view: Option<engine_core::TerrainView>,
}

impl EngineControl for TerrainControl {
    fn changed(&self) -> &'static str {
        "terrain changed"
    }

    fn apply(&self, engine: &mut Engine, _: Option<&SceneInstance>) -> Result<Value, DevError> {
        let world = engine.app.world_mut();
        if let Some(pixel_error) = self.pixel_error {
            // The settings' own rule.
            let rule = engine_core::TerrainSettings::rule_of("pixel_error")
                .expect("terrain settings declare pixel_error");
            if let Err(invalid) = engine_settings::Setting::check(&pixel_error, "pixel_error", rule)
            {
                return Err(DevError::new(
                    ErrorCode::OutOfRange,
                    "$.pixel_error",
                    invalid.to_string(),
                ));
            }
            let mut settings = world
                .get_resource::<engine_core::TerrainSettings>()
                .copied()
                .unwrap_or_default();
            settings.pixel_error = pixel_error;
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
        if let Some(view) = self.view {
            debug.view = view;
        }
        world.insert_resource(debug);
        Ok(result::terrain(TerrainControlResult {
            temporary: true,
            terrain_changed: true,
            frozen_at: debug.freeze.map(|eye| eye.to_array()),
            view: self.view.map(|_| debug.view),
            pixel_error: world
                .get_resource::<engine_core::TerrainSettings>()
                .map(|settings| settings.pixel_error),
        }))
    }
}

/// What `terrain` answers: terrain's view and level of detail after.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainControlResult {
    /// Controls change the session's world, not the scene's source.
    pub temporary: bool,
    /// Always true: terrain draws as asked.
    pub terrain_changed: bool,
    /// Where streaming stays frozen, where it does (world space).
    pub frozen_at: Option<[f32; 3]>,
    /// The view terrain shows, where the control named one.
    pub view: Option<engine_core::TerrainView>,
    /// The settings' pixel error, where terrain settings exist.
    pub pixel_error: Option<f32>,
}

/// Temporarily edit a generated terrain over its layers and bakes
/// and render while paused: `change` raises or lowers the
/// ground by meters, levels it to a world height or cuts holes in it, at
/// full strength within `radius` meters of world `center` `[x, z]` and
/// fading over `falloff` more, under `id` (letters, digits, `_`, `-` and
/// `.`), which repeating leaves the terrain alone. Only the tiles
/// under it regenerate. Returns whether it applied, the edits standing,
/// where tiles regenerate (world `min` and `max` `[x, z]`) and the most
/// the ground moved (meters). `terrain` names a scene's terrain entity
/// (the first terrain when omitted). The `Terrain` inspector lists the
/// edits as stamp layers to save into the scene.
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
    }
}

/// Take the terrain's last edit away and render while
/// paused; `terrain` as for `terrain_edit`.
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
    }
}

/// Put back the last edit undo took away, unless an edit came since,
/// and render while paused.
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
    }
}

/// The terrain edits undo took away in this session, latest last: redo
/// puts them back.
#[derive(Resource, Default)]
struct TerrainRedo(Vec<(Entity, engine_core::TerrainEdit)>);

/// Runs `editing` on the terrain `name` names in `scene`, with the session's
/// redo stack.
fn editing(
    engine: &mut Engine,
    scene: Option<&SceneInstance>,
    name: Option<&str>,
    editing: TerrainEditing,
) -> Result<Value, DevError> {
    let mut redo = engine
        .app
        .world_mut()
        .remove_resource::<TerrainRedo>()
        .unwrap_or_default();
    let result = edit_terrain(engine, scene, name, &mut redo.0, editing);
    engine.app.world_mut().insert_resource(redo);
    result
}
