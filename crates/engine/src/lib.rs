#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

pub use engine_collision as collision;
pub use engine_navigation as navigation;
pub use engine_physics as physics;
pub use engine_profile as profiling;
pub use engine_scene as scene;
pub use engine_settings as settings;
pub use engine_time_of_day as time_of_day;
pub use engine_world_streaming as world_streaming;
pub mod camera;
pub mod game;
pub use engine_core::{Allowed, Detail, Error, Expected};
pub use engine_settings::JS_ERROR_NAME;

pub type Result<T, E = Error> = std::result::Result<T, E>;
#[doc(hidden)]
pub mod surface;
pub use game::{Game, Inspectors};
#[cfg(not(target_arch = "wasm32"))]
pub use surface::runtime_typescript;
#[cfg(not(target_arch = "wasm32"))]
pub mod tooling;

pub use engine_core::collections;
pub use engine_core::command_line;
pub use engine_core::{bevy_ecs, glam, smoke, smoke_check, smoke_eq};

pub mod ground {
    pub use engine_core::{Pyramid, TILE_QUADS, TILE_SAMPLES, TileKey};
    pub use engine_places::{
        GroundBridge, GroundGrid, GroundObstacle, GroundRead, GroundRecord, GroundRegions,
        GroundSlot, GroundSource, GroundWorker,
    };
}

pub mod solids {
    pub use engine_places::{
        FieldSamples, SolidField, SolidShape, SolidsFace, SolidsMember, WorldSolid, WorldSolids,
        WorldSolidsList,
    };
}

mod games {
    pub use engine_ambient_occlusion::AmbientOcclusionBrick;
    pub use engine_auto_exposure::{AutoExposure, AutoExposureBrick};
    pub use engine_bloom::{Bloom, BloomBrick};
    pub use engine_core::Culling;
    pub use engine_core::{
        Aabb, ActionId, ActionInput, AlphaImage, AlphaMode, AmbientOcclusion, App, AssetBudget,
        AssetLibrary, Camera, Clouds, Cube, DebugView, DebugViewMode, Dither, DynamicResolution,
        DynamicResolutionStats, Engine, EngineConfig, EngineEntityCommands, Filter,
        FixedPostSimulation, FixedPreSimulation, FixedSimulation, FixedSimulationSystems,
        FixedTime, Foliage, FrameReport, FrameTime, GlobalTransform, GlobalTransforms, Grading,
        Grounding, Haze, Highlights, InputState, Interior, InterpolateTransform, LevelOfDetail,
        LightPreset, LightRegion, LocalLight, LocalLightSettings, MakeSources, MakeTextures,
        Material, MaterialHandle, Mesh, Mesh3d, MeshHandle, MeshVertex, NightSky, OcclusionCulling,
        Parent, ParticleBlend, ParticleEmitter, ParticleSettings, Plugin, PluginGroup,
        Presentation, PresentationState, PresentedTime, PresentedTransform, PresentedTransforms,
        ProvidedAssets, Records, ReflectionProbe, ReflectionProbeSettings, Resolution,
        ResolutionStats, Sampler, Shadows, Sky, SkyLighting, SkyOcclusion, SplineMesh, Spot,
        StreamingFocus, SubpixelCulling, Sunlight, Surface, SurfaceLight, TemporalAntialiasing,
        Texture, TextureHandle, TextureRef, TextureSet, TextureSource, TextureUsage, Tint,
        TopLayer, Transform, UvTransform, ViewSync, ViewSyncSystems, Wind, WorldSync,
        WorldSyncSystems, Wrap, edit_transform, math, reset_all_interpolation, reset_interpolation,
        set_parent, set_parents, trim_cards,
    };
    pub use engine_debug_view::DebugViewBrick;
    pub use engine_dither::DitherBrick;
    pub use engine_ground_cover::GroundCoverBrick;
    pub use engine_grounding::{Grounded, GroundingBrick};
    pub use engine_lighting::LightingModelBrick;
    pub use engine_lights::LightsBrick;
    pub use engine_meshes::MeshesBrick;
    pub use engine_navigation::{
        NavAgent, NavAgents, NavQuery, NavStatus, NavSystems, NavWalk, Navigation, NavigationBrick,
    };
    pub use engine_particles::ParticlesBrick;
    pub use engine_physics::Physics;
    pub use engine_places::{
        Ground, GroundFace, GroundPoint, HeightHeld, WaterBodies, WaterBodiesList, WaterFace,
        WaterKind, WaterObstacle, WaterObstacles, WaterSample, WaterSurfaces, WaterWorker,
    };
    pub use engine_probes::ReflectionProbesBrick;
    pub use engine_reflections::{Reflections, ReflectionsBrick};
    pub use engine_scatter::ScatterBrick;
    pub use engine_shadows::ShadowsBrick;
    pub use engine_sky::SkyBrick;
    pub use engine_sun::SunBrick;
    pub use engine_taa::TaaBrick;
    pub use engine_terrain::{
        Cover, GroundCover, Occupancy, Terrain, TerrainBrick, TerrainSample, TerrainSettings,
        Terrains, Water,
    };
    pub use engine_terrain_shading::TerrainShadingBrick;
    pub use engine_time_of_day::{TimeOfDay, TimeOfDayBrick};
    pub use engine_tonemap::TonemapBrick;
    pub use engine_water::WaterBrick;
    pub use engine_world_streaming::{
        StreamSet, Streamed, StreamedId, StreamedItem, WorldCells, WorldChanges, WorldStreaming,
        WorldStreamingBrick, remove_streamed,
    };
}
pub use games::*;

pub use engine_core::{
    DetailShares, DrawnKind, EnvironmentSection, ErrorCode, HierarchyReport, InvalidSetting,
    LodLevel, MAX_EMITTER_PARTICLES, MAX_HIERARCHY_DEPTH, MAX_LIGHT_RANGE, MAX_MOVING_TILES,
    MAX_SHADOW_LEVELS, MAX_SPLINE_POINTS, MAX_SPOT_DEGREES, MAX_SPREAD_DEGREES,
    MAX_TEXTURE_DIMENSION, MAX_TRANSFORM_NODES, MIN_SPOT_DEGREES, MaterialShare, MeshLods,
    PixelProbe, PluginGroupBuilder, SplineChunks, SunLookup, TerrainProbe, TextureFamily,
    TextureFormat, TextureSampling, TransformNodeLimit, Viewport, WaterProbe, WaterReach,
    section_in,
};
pub use engine_terrain::{MAX_BOUNCE_LEVELS, MAX_OCCUPANCY_LEVELS};

pub mod bricks {
    pub use engine_core::{
        Absent, AddBrick, AuthoredId, Brick, BrickCost, BrickCosts, BrickList, BrickRecord,
        BrickRegistry, BrickRow, BrickSet, BrickStats, BrickTable, Bricks, CORE_PASSES, CORE_ROW,
        CpuList, CpuSlot, DetailConsumer, DetailConsumersList, DrawState, ExplainEntry,
        ExplainRecord, ExplainRequest, ExplainedEntity, Explainer, Extracted, Face, FaceOffer,
        FaceType, Fill, Filled, GPU_IDLE_ROW, InspectedKind, InspectorEntry, Join, Joined, List,
        Member, Members, Operation, OperationEntry, OperationScene, Owned, Place, PlaceKind,
        SettingsEntry, Slot, StatsEntry, UNATTRIBUTED_ROW, WARMUP_FRAMES, authored_entity,
        authored_id, bounds, brick_name, face_name, missing_brick, parse_json, parse_value,
        parse_value_at, source_line,
    };
    pub use engine_places::PhysicsSlot;

    pub fn standard() -> BrickList {
        BrickList::default()
            .with::<engine_terrain::TerrainBrick>()
            .with::<engine_grounding::GroundingBrick>()
            .with::<engine_core::Wind>()
            .with::<engine_world_streaming::WorldStreamingBrick>()
            .with::<engine_navigation::NavigationBrick>()
            .with::<engine_time_of_day::TimeOfDayBrick>()
            .with::<engine_collision::Collision>()
            .with::<engine_physics::PhysicsBrick>()
            .with::<engine_sun::SunBrick>()
            .with::<engine_lighting::LightingModelBrick>()
            .with::<engine_meshes::MeshesBrick>()
            .with::<engine_terrain_shading::TerrainShadingBrick>()
            .with::<engine_scatter::ScatterBrick>()
            .with::<engine_shadows::ShadowsBrick>()
            .with::<engine_lights::LightsBrick>()
            .with::<engine_sky::SkyBrick>()
            .with::<engine_probes::ReflectionProbesBrick>()
            .with::<engine_ground_cover::GroundCoverBrick>()
            .with::<engine_water::WaterBrick>()
            .with::<engine_ambient_occlusion::AmbientOcclusionBrick>()
            .with::<engine_reflections::ReflectionsBrick>()
            .with::<engine_bloom::BloomBrick>()
            .with::<engine_dither::DitherBrick>()
            .with::<engine_particles::ParticlesBrick>()
            .with::<engine_taa::TaaBrick>()
            .with::<engine_auto_exposure::AutoExposureBrick>()
            .with::<engine_tonemap::TonemapBrick>()
            .with::<engine_debug_view::DebugViewBrick>()
            .with::<engine_core::LevelOfDetail>()
            .with::<engine_core::Culling>()
            .with::<engine_core::Resolution>()
            .with::<engine_layer_noise::LayerNoiseBrick>()
            .with::<engine_layer_stamps::LayerStampsBrick>()
            .with::<engine_layer_heightmap::LayerHeightmapBrick>()
            .with::<engine_layer_paths::LayerPathsBrick>()
            .with::<engine_layer_erosion::LayerErosionBrick>()
            .with::<engine_layer_hydrology::LayerHydrologyBrick>()
            .with::<engine_layer_relief::LayerReliefBrick>()
            .with::<crate::host::SceneViewsBrick>()
    }

    pub fn standard_names(names: &str, path: &str) -> crate::Result<Vec<&'static str>> {
        standard().check_names(names.split(','), path)
    }
}

pub mod terrain {
    pub use engine_layer_erosion as layer_erosion;
    pub use engine_layer_heightmap as layer_heightmap;
    pub use engine_layer_hydrology as layer_hydrology;
    pub use engine_layer_noise as layer_noise;
    pub use engine_layer_paths as layer_paths;
    pub use engine_layer_relief as layer_relief;
    pub use engine_layer_stamps as layer_stamps;
    pub use engine_terrain as data;
    pub use engine_terrain::scene::{
        SceneColliderShape, SceneSurroundings, SceneTerrain, SceneTerrainAsset, SceneTerrainLod,
        SceneTerrainMaterial, SceneTerrainPart, SceneTerrainSurface,
    };
    pub use engine_terrain::{
        AUTHORED_LAYER, AuthoredPlacement, BRIDGE_SOLIDS, Biome, BiomeLayer, BiomeRules, Biomes,
        Blend, Control, DECK_DEPTH, DECK_MARGIN, DECK_RISE, Expr, Facing, FarPlacement, GivenRiver,
        HeightImage, Hydrology, HydrologySample, LayerKind, Layers, MAX_COLLIDER_EXTENT,
        MAX_PLACEMENT_LODS, MAX_PLACEMENT_PARTS, MIN_COLLIDER_EXTENT, MotionDetail, NoiseMask,
        Placed, PlacedCollider, PlacedShape, PlacementAsset, PlacementChoice, PlacementLayer,
        PlacementLod, PlacementPart, Pyramid, Route, RouteOptions, Stamp, StampOp, SurfaceMaterial,
        SurfaceProjection, Surroundings, SurroundingsStatus, TerrainBakes, TerrainEdit,
        TerrainLayer, TerrainLayerKinds, TerrainPrefetch, TerrainRiver, TerrainSurface, Tile,
        TileKey, expr, layer,
    };
}

pub mod render {
    pub use engine_core::{
        GUST_QUICK_ACROSS, GUST_QUICK_RATE, GUST_QUICK_SHARE, GUST_QUICK_TRAVEL, GUST_SLOW_RATE,
        GUST_SLOW_SHARE, GUST_SLOW_TRAVEL, GpuCurrent, GpuPresentation, GradingValues,
        MAX_NORMAL_INVERSE_NORM, MaterialBinding, NO_PARENT, NODE_ACTIVE, NODE_BLEND,
        NODE_PARENT_ACTIVE, NODE_VALID, NodeInfo, NodePose, PoseRows, PresentationMode,
        RenderEmitter, RenderFrame, RenderInstance, RenderLight, RenderLights, RenderMaterial,
        RenderMesh, RenderParticles, RenderReflectionProbe, RenderReflectionProbes, RenderSnapshot,
        RenderTexture, SkyLightingValues, SunValues, SurfaceLightValues, TransformIndex,
        TransformPropagation, presentation_bounds, propagate_transforms, valid_global_matrix,
    };
    pub use engine_terrain::bounce::TerrainBounce;
    pub use engine_terrain::lighting::TerrainLighting;
    pub use engine_terrain::{
        MacroMap, RenderPlacementAsset, RenderPlacementLayer, RenderPlacementPart,
        RenderSurfaceMaps, RenderTerrain, RenderTerrains,
    };
    pub use engine_world_streaming::{CellView, STREAMING_GRIDS, StreamingGrid};
}

pub mod stats {
    pub use engine_core::{
        Applied, CloudStats, CullingStats, Cut, DetailStats, ExposureStats, FrameStats,
        GroundCoverStats, JobStats, LightStats, LinkedStats, Names, OccupancyStats, ParticleStats,
        ProbeStats, RenderStats, ShadowStats, SkyStats, TerrainBounceStats, TerrainRenderStats,
        WaterStats,
    };
    pub use engine_layer_erosion::ErosionStats;
    pub use engine_layer_paths::PathStats;
    pub use engine_navigation::NavStats;
    pub use engine_physics::PhysicsStats;
    pub use engine_terrain::{HydrologyStats, TerrainStats};
    pub use engine_world_streaming::CellStats;
}

pub mod host;
#[cfg(feature = "dev")]
pub mod smokes;

#[cfg(feature = "dev")]
pub use engine_dev as dev;

mod app_host;
#[cfg(feature = "dev")]
mod dev_host;
mod game_host;
pub use engine_render::{AdapterInfo, RenderTarget};
#[cfg(not(target_arch = "wasm32"))]
pub use engine_render::{GpuMark, GpuPreference, NativeWindow, start_native_instances_with};
pub use game_host::GameHost;

pub fn shader_modules() -> Vec<engine_render::linker::ShaderModule> {
    use bricks::AddBrick;
    use engine_render::linker::ShaderModule;
    use engine_render::places::{PLACE_OWNER, place_module};
    let mut engine = Engine::default();
    engine.app.add_bricks(bricks::standard());
    let mut modules = engine_render::shaders::MODULES.to_vec();
    let Some(registry) = engine.app.world().get_resource::<bricks::BrickRegistry>() else {
        return modules;
    };
    for (_, render) in engine_render::render_bricks::render_bricks(registry) {
        for module in render.modules() {
            if !modules.iter().any(|known| known.path == module.path) {
                modules.push(*module);
            }
        }
    }
    let declared = engine_render::composite::POST_PLACES.iter().copied();
    for place in declared.chain(registry.places()) {
        let Some(source) = place.face.wgsl else {
            continue;
        };
        let path = place_module(place.name);
        if !modules.iter().any(|known| known.path == path) {
            modules.push(ShaderModule {
                path: Box::leak(path.into_boxed_str()),
                brick: PLACE_OWNER,
                source,
            });
        }
    }
    modules
}

#[cfg(target_arch = "wasm32")]
pub mod browser;

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
pub mod __private {
    pub use js_sys;
    pub use wasm_bindgen;
    pub use wasm_bindgen::JsValue;
    pub use wasm_bindgen::prelude::wasm_bindgen as bindgen;
    pub use web_sys;
}

#[cfg(feature = "dev")]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_dev {
    ($($item:item)*) => { $($item)* };
}

#[cfg(not(feature = "dev"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_dev {
    ($($item:item)*) => {};
}

#[cfg(feature = "profiling")]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_profiling {
    ($($item:item)*) => { $($item)* };
}

#[cfg(not(feature = "profiling"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __if_profiling {
    ($($item:item)*) => {};
}

pub mod prelude {
    pub use crate::bricks::AddBrick;
    pub use crate::camera::{CameraSystems, FlyCameraController, FlyCameraPlugin};
    pub use crate::games::*;
    pub use crate::{Game, Inspectors};
    pub use engine_collision::{
        Collider, ColliderShape, Collision, CollisionLayers, CollisionRay, CollisionSystems,
        CollisionWorld, QueryFilter, SensorFilter, ShapeCastOptions, refresh_collision_world,
    };
    pub use engine_core::bevy_ecs::{
        self,
        bundle::Bundle,
        change_detection::{DetectChanges, DetectChangesMut, Mut, Ref},
        component::Component,
        entity::Entity,
        error::{BevyError, Result},
        event::{EntityEvent, Event},
        lifecycle::{Add, Despawn, Insert, Remove},
        name::Name,
        observer::{Observer, On},
        query::{Added, AnyOf, Changed, Has, Or, QueryState, With, Without},
        resource::Resource,
        schedule::{
            ApplyDeferred, IntoScheduleConfigs, IntoSystemSet, Schedule, Schedules,
            SystemCondition, SystemSet, common_conditions::*,
        },
        system::{
            Command, Commands, EntityCommand, EntityCommands, In, IntoSystem, Local, ParamSet,
            Populated, Query, Res, ResMut, Single, System,
        },
        world::{EntityMut, EntityRef, EntityWorldMut, FromWorld, World},
    };
    pub use engine_core::glam::{Mat4, Quat, Vec2, Vec3, Vec4};
    pub use engine_physics::{BodyMotion, PhysicsBrick, PhysicsSystems, PhysicsWorld, RigidBody};
    pub use engine_scene::{SceneDocument, SceneEntity, SceneInstance};
}
