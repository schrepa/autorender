//! Browser-first 3D engine façade. Gameplay intentionally uses Bevy ECS.
//!
//! The root and [`prelude`] hold what games use; [`terrain`] holds terrain
//! authoring, [`render`] the renderer's contract, [`stats`] the frame
//! statistics and [`host`] what hosts wire.

/// Opt-in collision geometry, filtered queries, and fixed-tick sensor events.
pub use engine_collision as collision;
/// Optional bounded profiling scopes; calls compile away without `profiling`.
pub use engine_profile as profiling;
/// Versioned authoring, validation, and explicit scene ownership.
pub use engine_scene as scene;
/// Settings declared once with their rules (`settings!`), as the engine's own are.
pub use engine_settings as settings;
pub mod camera;
mod error;
pub mod game;
mod plugins;
pub use error::{Error, Result};
#[doc(hidden)]
pub mod surface;
pub use game::{Game, Inspectors};
pub use plugins::EnginePlugins;
#[cfg(not(target_arch = "wasm32"))]
pub use surface::runtime_typescript;
/// What tooling generates from Rust, for apps' `schema` binaries.
#[cfg(not(target_arch = "wasm32"))]
pub mod tooling;

// The ECS and math the engine is built on, and its deterministic
// trigonometry (`math::sin`, not `f32::sin`).
pub use engine_core::{bevy_ecs, glam, math};
// Every environment section, for code listing them (`environment_sections!(callback)`).
pub use engine_core::environment_sections;

// What games use: the engine and its schedules, time, input, transforms,
// objects, assets, the look's settings and the stores gameplay reads.
pub use engine_core::{
    Aabb, AlphaMode, AnimatedProperty, AnimationChannel, AnimationClip, AnimationSample,
    AssetBudget, AssetError, AssetLibrary, ClipHandle, Filter, Grounding, KeyInterpolation,
    Keyframes, LodLevel, Material, MaterialHandle, Mesh, MeshHandle, MeshLods, MeshVertex, Sampler,
    Surface, Texture, TextureFormat, TextureHandle, TextureRef, TopLayer, Transmission,
    UvTransform, VirtualMesh, Wrap, mesh_lods,
};
pub use engine_core::{
    ActionId, ActionInput, AnimationPlayer, AnimationSystems, App, Atmosphere, Camera, Cube,
    Engine, EngineConfig, ErrorCode, FixedPostSimulation, FixedPreSimulation, FixedSimulation,
    FixedSimulationSystems, FixedTickEnd, FixedTickEndSystems, FixedTime, FrameReport, FrameTime,
    InputState, Plugin, PluginGroup, PluginGroupBuilder, Presentation, PresentedTime, ViewSync,
    ViewSyncSystems, WorldSync, WorldSyncSystems,
};
// The engine's features, each a plugin of `EnginePlugins`, and the ground
// what stands on it reads.
pub use engine_core::{
    AmbientOcclusion, Bloom, Clouds, ContactShadows, DecalSettings, Deformation, Environment,
    EnvironmentSection, GroundCover, InvalidSetting, LevelOfDetail, LocalLightSettings, Navigation,
    OcclusionCulling, Occupancy, Ocean, ParticleSettings, ReflectionProbeSettings, Reflections,
    Refraction, Shadows, Sky, SubpixelCulling, SunShafts, TemporalAntialiasing, TerrainSettings,
    TimeOfDay, VirtualGeometry, Water, Wind, WorldStreaming,
};
pub use engine_core::{
    AnimationPlugin, DeformationPlugin, Feature, Features, Ground, GroundSample, HeightHeld,
    NavigationPlugin, OceanPlugin, TerrainPlugin, TimeOfDayPlugin, WorldStreamingPlugin,
    has_feature,
};
// The fallback mesh size `Mesh::from_virtual` takes by default.
pub use engine_core::DEFAULT_FALLBACK_TRIANGLES;
pub use engine_core::{
    Cover, Decal, DecalBlend, Deformer, EngineEntityCommands, Floating, GlobalTransform,
    GlobalTransforms, Grounded, HierarchyError, HierarchyReport, Interior, InterpolateTransform,
    InterpolationError, LocalLight, MAX_HIERARCHY_DEPTH, MAX_TRANSFORM_NODES, Mesh3d, NavAgent,
    NavAgents, NavStatus, NavSystems, NavWalk, OceanPoint, OceanSurface, OceanSystems, OceanWave,
    OceanWaves, Parent, ParticleBlend, ParticleEmitter, PresentationState, PresentedTransform,
    PresentedTransforms, ReflectionProbe, Skin, SplineChunks, SplineMesh, Spot, StreamSet,
    Streamed, StreamedId, StreamedItem, StreamingFocus, Terrain, TerrainSample, Terrains, Tint,
    Track, TrackDepth, Tracks, Transform, TransformEditError, TransformNodeLimit, Viewport,
    WorldCells, WorldChanges, edit_transform, remove_streamed, reset_all_interpolation,
    reset_interpolation, set_parent, set_parents,
};
// The bounds settings and components validate against.
pub use engine_core::{
    MAX_EMITTER_PARTICLES, MAX_KEPT_LEVELS, MAX_LIGHT_RANGE, MAX_OCEAN_WAVES, MAX_SHADOW_CASCADES,
    MAX_SPLINE_POINTS, MAX_SPOT_DEGREES, MAX_SPREAD_DEGREES, MAX_TEXTURE_DIMENSION,
    MAX_TRACK_DEPTH, MIN_SPOT_DEGREES,
};

/// Terrain authoring: layer stacks, masks, stamps, bakes, placement and
/// surfaces, which a [`Terrain`] component describes, and the tile data
/// they make.
pub mod terrain {
    /// Heightfield terrain data: tile pyramids, layer stacks, tiles,
    /// patches and the `ETR1` format.
    pub use engine_core::terrain_data as data;
    pub use engine_core::{
        AUTHORED_LAYER, AuthoredPlacement, Biome, BiomeError, BiomeLayer, BiomeRules, Biomes,
        Blend, Control, DECK_DEPTH, DECK_MARGIN, DECK_RISE, Erosion, ErosionLayer, ErosionStage,
        FarPlacement, HeightImage, HeightmapLayer, Hydrology, HydrologyLayer, HydrologySample,
        HydrologyStage, Layer, LayerError, Layers, MAX_COLLIDER_EXTENT, MAX_PLACEMENT_LODS,
        MIN_COLLIDER_EXTENT, Mask, MotionDetail, NoiseLayer, NoiseMask, Patch, PathLayer,
        PathProfile, PathStage, Placed, PlacedCollider, PlacedShape, PlacementAsset,
        PlacementChoice, PlacementLayer, PlacementLod, Pyramid, PyramidError, Remap, Route,
        RouteOptions, Stamp, StampOp, StreamingFocus, SurfaceMaterial, SurfaceProjection,
        Surroundings, SurroundingsStatus, Terrain, TerrainBakes, TerrainEdit, TerrainPrefetch,
        TerrainSample, TerrainSettings, TerrainSource, TerrainSurface, Terrains, Tile, TileKey,
        WaterGrid,
    };
}

/// The renderer's contract: what extraction hands it each frame, the GPU
/// presentation's inputs and the bounds its shaders share with the CPU.
pub mod render {
    pub use engine_core::{
        CLUSTER_TRIANGLES, CLUSTER_VERTICES, CLUSTER_WORDS, CellView, DEFORMATION_GRAIN,
        DeformationMap, GROUP_WORDS, GpuCurrent, GpuPresentation, MAX_NORMAL_INVERSE_NORM,
        MaterialBinding, NO_PARENT, NODE_ACTIVE, NODE_BLEND, NODE_PARENT_ACTIVE, NODE_VALID,
        NODE_WORDS, NodeInfo, NodePose, PAGE_WORDS, PoseRows, PresentationMode, RIM_REACH,
        RIM_START, RenderDecal, RenderDeformation, RenderEmitter, RenderFrame, RenderInstance,
        RenderJoint, RenderLight, RenderMaterial, RenderMesh, RenderPlacementAsset,
        RenderReflectionProbe, RenderSnapshot, RenderSurfaceMaps, RenderTerrain, RenderTerrains,
        RenderTexture, STREAMING_GRIDS, StreamingGrid, TransformIndex, TransformPropagation,
        VIRTUAL_MESH_MAGIC, carve_profile, presentation_bounds, propagate_transforms, rim_profile,
        valid_global_matrix,
    };
}

/// What a frame observed, by section: [`stats::FrameStats`] as hosts and
/// agents read it.
pub mod stats {
    pub use engine_core::{
        Applied, CellStats, CloudStats, CullingStats, DeformationStats, DetailStats, Drawn,
        ErosionStats, FrameStats, GroundCoverStats, HydrologyStats, JobStats, LightStats, NavStats,
        OccupancyStats, OceanStats, PathStats, ProbeStats, RenderPasses, RenderStats, ShadowStats,
        SkyStats, TerrainRenderStats, TerrainStats, VirtualGeometryStats, WaterStats,
    };
}

/// What hosts wire: job workers, terrain tile fetches and the terrain's
/// debug views.
pub mod host {
    pub use crate::app_host::{JobMessage, now_ms};
    pub use engine_core::{
        FileHeader, GAMEPLAY_JOBS, IDLE_JOBS, JobId, JobRequest, JobShare, Jobs, TerrainDebug,
        TerrainView, TileRequest, URGENT_JOBS, VIEW_JOBS, content_hash, job_class, job_priority,
    };
}

/// Optional, bounded development inspection. No inspection runs automatically.
#[cfg(feature = "dev")]
pub use engine_dev as dev;

mod app_host;
#[cfg(feature = "dev")]
mod dev_host;
mod game_host;
/// Where hosts present, and what renderers fail with.
pub use engine_render::{AdapterInfo, RenderError, RenderTarget};
/// A native window hosts present to, the GPU they ask for, and how tools
/// that must see the GPU interface's calls start its instances.
#[cfg(not(target_arch = "wasm32"))]
pub use engine_render::{GpuPreference, NativeWindow, start_native_instances_with};
pub use game_host::GameHost;

/// The page's side of a hosted game, owned by the main thread.
#[cfg(target_arch = "wasm32")]
pub mod browser;

/// What [`export_game!`] expands to refers to.
#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
pub mod __private {
    pub use js_sys;
    pub use wasm_bindgen;
    pub use wasm_bindgen::JsValue;
    pub use wasm_bindgen::prelude::wasm_bindgen as bindgen;
    pub use web_sys;
}

/// Its items in development builds of the engine, nothing otherwise.
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

/// Its items in profiling builds of the engine, nothing otherwise.
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

/// What games import: the root's gameplay names, collision, scenes, the
/// fly camera, the parts of Bevy the engine runs and glam's types.
pub mod prelude {
    pub use crate::camera::{CameraSystems, FlyCameraController, FlyCameraPlugin};
    pub use crate::{
        Aabb, AlphaMode, AnimatedProperty, AnimationChannel, AnimationClip, AssetBudget,
        AssetError, AssetLibrary, ClipHandle, Filter, Grounding, KeyInterpolation, Keyframes,
        Material, MaterialHandle, Mesh, MeshHandle, MeshVertex, Sampler, Surface, Texture,
        TextureHandle, TextureRef, TopLayer, Transmission, UvTransform, Wrap,
    };
    pub use crate::{
        ActionId, ActionInput, AnimationPlayer, AnimationSystems, App, Atmosphere, Camera, Cube,
        Engine, EngineConfig, FixedPostSimulation, FixedPreSimulation, FixedSimulation,
        FixedSimulationSystems, FixedTickEnd, FixedTickEndSystems, FixedTime, FrameReport,
        FrameTime, Game, InputState, Inspectors, Plugin, PluginGroup, Presentation, PresentedTime,
        ViewSync, ViewSyncSystems, WorldSync, WorldSyncSystems, math,
    };
    pub use crate::{
        AmbientOcclusion, Bloom, Clouds, ContactShadows, DecalSettings, Deformation, Environment,
        GroundCover, LevelOfDetail, LocalLightSettings, Navigation, OcclusionCulling, Occupancy,
        Ocean, ParticleSettings, ReflectionProbeSettings, Reflections, Refraction, Shadows, Sky,
        SubpixelCulling, SunShafts, TemporalAntialiasing, TerrainSettings, TimeOfDay,
        VirtualGeometry, Water, Wind, WorldStreaming,
    };
    pub use crate::{
        AnimationPlugin, DeformationPlugin, EnginePlugins, Feature, Ground, GroundSample,
        HeightHeld, NavigationPlugin, OceanPlugin, TerrainPlugin, TimeOfDayPlugin,
        WorldStreamingPlugin,
    };
    pub use crate::{
        Cover, Decal, DecalBlend, Deformer, EngineEntityCommands, Floating, GlobalTransform,
        GlobalTransforms, Grounded, HierarchyError, Interior, InterpolateTransform,
        InterpolationError, LocalLight, Mesh3d, NavAgent, NavAgents, NavStatus, NavSystems,
        NavWalk, OceanSystems, Parent, ParticleBlend, ParticleEmitter, PresentationState,
        PresentedTransform, PresentedTransforms, ReflectionProbe, Skin, SplineMesh, Spot,
        StreamSet, Streamed, StreamedId, StreamedItem, StreamingFocus, Terrain, TerrainSample,
        Terrains, Tint, Track, TrackDepth, Tracks, Transform, TransformEditError, WorldCells,
        WorldChanges, edit_transform, remove_streamed, reset_all_interpolation,
        reset_interpolation, set_parent, set_parents,
    };
    pub use engine_collision::{
        Collider, ColliderShape, CollisionLayers, CollisionPlugin, CollisionRay, CollisionSystems,
        CollisionWorld, QueryFilter, SensorFilter, ShapeCastOptions, refresh_collision_world,
    };
    pub use engine_core::glam::{Mat4, Quat, Vec2, Vec3, Vec4};
    // Bevy derives resolve this name in client scope when only the engine
    // facade is a direct dependency, so expose the crate beside the parts of
    // its prelude the engine runs: not its hierarchy (the engine's is
    // `Parent`) nor message queues (which need schedules it never runs).
    pub use engine_core::bevy_ecs::{
        self,
        bundle::Bundle,
        change_detection::{DetectChanges, DetectChangesMut, Mut, Ref},
        component::Component,
        entity::Entity,
        error::{BevyError, Result},
        event::{EntityEvent, Event},
        lifecycle::{Add, Despawn, Insert, Remove, RemovedComponents},
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
    pub use engine_scene::{SceneDocument, SceneEntity, SceneInstance};
}
