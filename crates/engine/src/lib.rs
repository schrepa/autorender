#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

pub use engine_collision as collision;
pub use engine_physics as physics;
pub use engine_profile as profiling;
pub use engine_scene as scene;
pub use engine_settings as settings;
pub mod camera;
mod error;
pub mod game;
mod plugins;
pub use error::{Error, JS_ERROR_NAME, Result};
#[doc(hidden)]
pub mod surface;
pub use game::{Game, Inspectors};
pub use plugins::EnginePlugins;
#[cfg(not(target_arch = "wasm32"))]
pub use surface::runtime_typescript;
#[cfg(not(target_arch = "wasm32"))]
pub mod tooling;

pub use engine_core::collections;
pub use engine_core::environment_sections;
pub use engine_core::{bevy_ecs, glam};

mod games {
    pub use engine_core::{
        Aabb, ActionId, ActionInput, AlphaImage, AlphaMode, AmbientOcclusion, App, AssetBudget,
        AssetError, AssetLibrary, Atmosphere, AutoExposure, Bloom, Camera, Clouds, Cover, Cube,
        DebugView, DebugViewMode, Decal, DecalBlend, DecalSettings, Deformation, DeformationPlugin,
        Deformer, Dither, Engine, EngineConfig, EngineEntityCommands, Environment, Feature, Filter,
        FixedPostSimulation, FixedPreSimulation, FixedSimulation, FixedSimulationSystems,
        FixedTickEnd, FixedTickEndSystems, FixedTime, Floating, Foliage, FrameReport, FrameTime,
        GlobalTransform, GlobalTransforms, Ground, GroundCover, GroundSample, Grounded, Grounding,
        Haze, HeightHeld, HierarchyError, Highlights, InputState, Interior, InterpolateTransform,
        InterpolationError, LevelOfDetail, LightRegion, LocalLight, LocalLightSettings,
        MakeSources, MakeTextures, Material, MaterialHandle, Mesh, Mesh3d, MeshHandle, MeshVertex,
        NavAgent, NavAgents, NavQuery, NavStatus, NavSystems, NavWalk, Navigation,
        NavigationPlugin, NightSky, OcclusionCulling, Occupancy, Ocean, OceanPlugin, OceanSystems,
        Parent, ParticleBlend, ParticleEmitter, ParticleSettings, Physics, Plugin, PluginGroup,
        Presentation, PresentationState, PresentedTime, PresentedTransform, PresentedTransforms,
        Records, ReflectionProbe, ReflectionProbeSettings, Reflections, Refraction, Sampler,
        Shadows, Sky, SkyOcclusion, SplineMesh, Spot, StreamSet, Streamed, StreamedId,
        StreamedItem, StreamingFocus, SubpixelCulling, Surface, TemporalAntialiasing, Terrain,
        TerrainPlugin, TerrainSample, TerrainSettings, Terrains, Texture, TextureHandle,
        TextureRef, TextureSet, TextureSource, TextureUsage, TimeOfDay, TimeOfDayPlugin, Tint,
        TopLayer, Track, TrackDepth, Tracks, Transform, TransformEditError, Transmission,
        UvTransform, ViewSync, ViewSyncSystems, Water, WaterKind, WaterObstacle, WaterObstacles,
        WaterSample, WaterSource, WaterSources, Wind, Windows, WorldCells, WorldChanges,
        WorldStreaming, WorldStreamingPlugin, WorldSync, WorldSyncSystems, Wrap, edit_transform,
        math, remove_streamed, reset_all_interpolation, reset_interpolation, set_parent,
        set_parents, trim_cards,
    };
}
pub use games::*;

pub use engine_core::{
    DetailShares, DrawnKind, EnvironmentSection, ErrorCode, Features, HierarchyReport,
    InvalidSetting, LodLevel, MAX_BOUNCE_LEVELS, MAX_EMITTER_PARTICLES, MAX_HIERARCHY_DEPTH,
    MAX_LIGHT_RANGE, MAX_MOVING_TILES, MAX_OCCUPANCY_LEVELS, MAX_OCEAN_WAVES, MAX_SHADOW_LEVELS,
    MAX_SPLINE_POINTS, MAX_SPOT_DEGREES, MAX_SPREAD_DEGREES, MAX_TEXTURE_DIMENSION,
    MAX_TRACK_DEPTH, MAX_TRANSFORM_NODES, MIN_SPOT_DEGREES, MaterialShare, MeshLods, OceanPoint,
    OceanSurface, OceanWave, OceanWaves, PixelProbe, PluginGroupBuilder, SplineChunks, SunLookup,
    TerrainProbe, TextureFamily, TextureFormat, TextureSampling, TransformNodeLimit, Viewport,
    WaterProbe, WaterReach, has_feature,
};

pub mod terrain {
    pub use engine_core::terrain_data as data;
    pub use engine_core::{
        AUTHORED_LAYER, AuthoredPlacement, Biome, BiomeError, BiomeLayer, BiomeRules, Biomes,
        Blend, Control, DECK_DEPTH, DECK_MARGIN, DECK_RISE, Erosion, ErosionLayer, ErosionStage,
        Facing, FarPlacement, GivenRiver, HeightImage, HeightmapLayer, HummocksLayer, Hydrology,
        HydrologyLayer, HydrologySample, HydrologyStage, Layer, LayerError, Layers,
        MAX_COLLIDER_EXTENT, MAX_PLACEMENT_LODS, MAX_PLACEMENT_PARTS, MIN_COLLIDER_EXTENT, Mask,
        MotionDetail, NoiseLayer, NoiseMask, PathLayer, PathProfile, PathStage, Placed,
        PlacedCollider, PlacedShape, PlacementAsset, PlacementChoice, PlacementLayer, PlacementLod,
        PlacementPart, Pyramid, PyramidError, Remap, RiversLayer, Route, RouteOptions, Stamp,
        StampOp, StepsLayer, SurfaceMaterial, SurfaceProjection, Surroundings, SurroundingsStatus,
        TerrainBakes, TerrainEdit, TerrainPrefetch, TerrainRiver, TerrainSurface, Tile, TileKey,
    };
}

pub mod render {
    pub use engine_core::{
        CellView, DEFORMATION_GRAIN, DeformationMap, GUST_QUICK_ACROSS, GUST_QUICK_RATE,
        GUST_QUICK_SHARE, GUST_QUICK_TRAVEL, GUST_SLOW_RATE, GUST_SLOW_SHARE, GUST_SLOW_TRAVEL,
        GpuCurrent, GpuPresentation, MAX_NORMAL_INVERSE_NORM, MacroMap, MaterialBinding, NO_PARENT,
        NODE_ACTIVE, NODE_BLEND, NODE_PARENT_ACTIVE, NODE_VALID, NodeInfo, NodePose, PoseRows,
        PresentationMode, RIM_REACH, RIM_START, RenderDecal, RenderDecals, RenderDeformation,
        RenderEmitter, RenderFrame, RenderInstance, RenderLight, RenderLights, RenderMaterial,
        RenderMesh, RenderParticles, RenderPlacementAsset, RenderPlacementLayer,
        RenderPlacementPart, RenderReflectionProbe, RenderReflectionProbes, RenderSnapshot,
        RenderSurfaceMaps, RenderTerrain, RenderTerrains, RenderTexture, STREAMING_GRIDS,
        StreamingGrid, TerrainBounce, TerrainLighting, TransformIndex, TransformPropagation,
        carve_profile, presentation_bounds, propagate_transforms, rim_profile, valid_global_matrix,
    };
}

pub mod stats {
    pub use engine_core::{
        Applied, CellStats, CloudStats, CullingStats, DeformationStats, DetailStats, Drawn,
        ErosionStats, ExposureStats, FrameStats, GroundCoverStats, HydrologyStats, JobStats,
        LightStats, Names, NavStats, OccupancyStats, OceanStats, PathStats, PhysicsStats,
        ProbeStats, RenderStats, ShadowStats, SkyStats, TerrainBounceStats, TerrainRenderStats,
        TerrainStats, WaterStats,
    };
}

pub mod host;

#[cfg(feature = "dev")]
pub use engine_dev as dev;

mod app_host;
#[cfg(feature = "dev")]
mod dev_host;
mod game_host;
pub use engine_render::shaders::MODULES as SHADER_MODULES;
pub use engine_render::{AdapterInfo, RenderError, RenderTarget};
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
    pub use crate::camera::{CameraSystems, FlyCameraController, FlyCameraPlugin};
    pub use crate::games::*;
    pub use crate::{EnginePlugins, Game, Inspectors};
    pub use engine_collision::{
        Collider, ColliderShape, CollisionLayers, CollisionPlugin, CollisionRay, CollisionSystems,
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
    pub use engine_physics::{BodyMotion, PhysicsPlugin, PhysicsSystems, PhysicsWorld, RigidBody};
    pub use engine_scene::{SceneDocument, SceneEntity, SceneInstance};
}
