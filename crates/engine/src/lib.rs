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

mod games {
    pub use engine_core::{
        Aabb, ActionId, ActionInput, AlphaImage, AlphaMode, AmbientOcclusion, App, AssetBudget,
        AssetLibrary, Atmosphere, AutoExposure, Bloom, Camera, Clouds, Cube, DebugView,
        DebugViewMode, Dither, DynamicResolution, DynamicResolutionStats, Engine, EngineConfig,
        EngineEntityCommands, Environment, Filter, FixedPostSimulation, FixedPreSimulation,
        FixedSimulation, FixedSimulationSystems, FixedTime, Foliage, FrameReport, FrameTime,
        GlobalTransform, GlobalTransforms, Ground, GroundFace, GroundPoint, Grounded, Grounding,
        Haze, HeightHeld, Highlights, InputState, Interior, InterpolateTransform, LevelOfDetail,
        LightRegion, LocalLight, LocalLightSettings, MakeSources, MakeTextures, Material,
        MaterialHandle, Mesh, Mesh3d, MeshHandle, MeshVertex, NightSky, OcclusionCulling, Parent,
        ParticleBlend, ParticleEmitter, ParticleSettings, Plugin, PluginGroup, Presentation,
        PresentationState, PresentedTime, PresentedTransform, PresentedTransforms, ProvidedAssets,
        Records, ReflectionProbe, ReflectionProbeSettings, Reflections, Resolution,
        ResolutionStats, Sampler, Shadows, Sky, SkyOcclusion, SplineMesh, Spot, StreamingFocus,
        SubpixelCulling, Surface, TemporalAntialiasing, Texture, TextureHandle, TextureRef,
        TextureSet, TextureSource, TextureUsage, Tint, TopLayer, Transform, UvTransform, ViewSync,
        ViewSyncSystems, WaterKind, WaterObstacle, WaterObstacles, WaterSample, WorldSync,
        WorldSyncSystems, Wrap, edit_transform, math,
        reset_all_interpolation, reset_interpolation, set_parent, set_parents, trim_cards,
    };
    pub use engine_core::{
        Culling, LightingModel, Lights, Particles, PostEffects, ReflectionProbes,
    };
    pub use engine_navigation::{
        NavAgent, NavAgents, NavQuery, NavStatus, NavSystems, NavWalk, Navigation, NavigationBrick,
    };
    pub use engine_physics::Physics;
    pub use engine_terrain::{
        Cover, GroundCover, Occupancy, Terrain, TerrainBrick, TerrainSample, TerrainSettings,
        Terrains, Water, Wind,
    };
    pub use engine_time_of_day::{TimeOfDay, TimeOfDayBrick};
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
};
pub use engine_terrain::{MAX_BOUNCE_LEVELS, MAX_OCCUPANCY_LEVELS};

pub mod bricks {
    pub use engine_core::{
        AddBrick, AuthoredId, Brick, BrickCost, BrickCosts, BrickList, BrickRecord, BrickRegistry,
        BrickRow, BrickSet, BrickStats, BrickTable, Bricks, CORE_PASSES, CORE_ROW, DrawState,
        ExplainEntry, ExplainRecord, ExplainRequest, ExplainedEntity, Explainer, Extracted,
        GPU_IDLE_ROW, InspectedKind, InspectorEntry, List, Operation, OperationEntry,
        OperationScene, Owned, SettingsEntry, Slot, StatsEntry, UNATTRIBUTED_ROW, WARMUP_FRAMES,
        authored_entity, authored_id, bounds, brick_name, missing_brick, parse_json, parse_value,
        parse_value_at, source_line,
    };

    pub fn standard() -> BrickList {
        BrickList::default()
            .with::<engine_terrain::TerrainBrick>()
            .with::<engine_world_streaming::WorldStreamingBrick>()
            .with::<engine_navigation::NavigationBrick>()
            .with::<engine_time_of_day::TimeOfDayBrick>()
            .with::<engine_collision::Collision>()
            .with::<engine_physics::PhysicsBrick>()
            .with::<engine_core::Lights>()
            .with::<engine_core::Particles>()
            .with::<engine_core::ReflectionProbes>()
            .with::<engine_core::Sky>()
            .with::<engine_core::Shadows>()
            .with::<engine_core::PostEffects>()
            .with::<engine_core::LevelOfDetail>()
            .with::<engine_core::Culling>()
            .with::<engine_core::Resolution>()
            .with::<engine_core::DebugView>()
            .with::<engine_core::LightingModel>()
            .with::<engine_layer_noise::LayerNoiseBrick>()
            .with::<engine_layer_stamps::LayerStampsBrick>()
            .with::<engine_layer_heightmap::LayerHeightmapBrick>()
            .with::<engine_layer_paths::LayerPathsBrick>()
            .with::<engine_layer_erosion::LayerErosionBrick>()
            .with::<engine_layer_hydrology::LayerHydrologyBrick>()
            .with::<engine_layer_relief::LayerReliefBrick>()
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
    pub use engine_terrain::{
        AUTHORED_LAYER, AuthoredPlacement, Biome, BiomeLayer, BiomeRules, Biomes, Blend, Control,
        DECK_DEPTH, DECK_MARGIN, DECK_RISE, Expr, Facing, FarPlacement, GivenRiver, HeightImage,
        Hydrology, HydrologySample, LayerKind, Layers, MAX_COLLIDER_EXTENT, MAX_PLACEMENT_LODS,
        MAX_PLACEMENT_PARTS, MIN_COLLIDER_EXTENT, MotionDetail, NoiseMask, Placed, PlacedCollider,
        PlacedShape, PlacementAsset, PlacementChoice, PlacementLayer, PlacementLod, PlacementPart,
        Pyramid, Route, RouteOptions, Stamp, StampOp, SurfaceMaterial, SurfaceProjection,
        Surroundings, SurroundingsStatus, TerrainBakes, TerrainEdit, TerrainLayer,
        TerrainLayerKinds, TerrainPrefetch, TerrainRiver, TerrainSurface, Tile, TileKey, expr,
        layer,
    };
}

pub mod render {
    pub use engine_core::{
        GpuCurrent, GpuPresentation, MAX_NORMAL_INVERSE_NORM, MaterialBinding, NO_PARENT,
        NODE_ACTIVE, NODE_BLEND, NODE_PARENT_ACTIVE, NODE_VALID, NodeInfo, NodePose, PoseRows,
        PresentationMode, RenderEmitter, RenderFrame, RenderInstance, RenderLight, RenderLights,
        RenderMaterial, RenderMesh, RenderParticles, RenderReflectionProbe, RenderReflectionProbes,
        RenderSnapshot, RenderTexture, TransformIndex, TransformPropagation, presentation_bounds,
        propagate_transforms, valid_global_matrix,
    };
    pub use engine_terrain::bounce::TerrainBounce;
    pub use engine_terrain::lighting::TerrainLighting;
    pub use engine_terrain::{
        GUST_QUICK_ACROSS, GUST_QUICK_RATE, GUST_QUICK_SHARE, GUST_QUICK_TRAVEL, GUST_SLOW_RATE,
        GUST_SLOW_SHARE, GUST_SLOW_TRAVEL, MacroMap, RenderPlacementAsset, RenderPlacementLayer,
        RenderPlacementPart, RenderSurfaceMaps, RenderTerrain, RenderTerrains, TerrainEnvironment,
    };
    pub use engine_world_streaming::{CellView, STREAMING_GRIDS, StreamingGrid};
}

pub mod stats {
    pub use engine_core::{
        Applied, CloudStats, CullingStats, DetailStats, Drawn, ExposureStats, FrameStats,
        GroundCoverStats, JobStats, LightStats, Names, OccupancyStats, ProbeStats, RenderStats,
        ShadowStats, SkyStats, TerrainBounceStats, TerrainRenderStats, WaterStats,
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
pub use engine_render::shaders::MODULES as SHADER_MODULES;
pub use engine_render::{AdapterInfo, RENDER_PASSES, RenderTarget};
#[cfg(not(target_arch = "wasm32"))]
pub use engine_render::{GpuMark, GpuPreference, NativeWindow, start_native_instances_with};
pub use game_host::GameHost;

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
