//! Browser-first 3D engine façade. Gameplay intentionally uses Bevy ECS.
//!
//! The root and [`prelude`] hold what games use; [`terrain`] holds terrain
//! authoring, [`render`] the renderer's contract, [`stats`] the frame
//! statistics and [`host`] what hosts wire.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

/// Opt-in collision geometry, filtered queries, and fixed-tick sensor events.
pub use engine_collision as collision;
/// Optional bounded profiling scopes; calls compile away without `profiling`.
pub use engine_profile as profiling;
/// Authoring, validation, and explicit scene ownership.
pub use engine_scene as scene;
/// Settings declared once with their rules (`settings!`), as the engine's own are.
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
/// What tooling generates from Rust, for apps' `schema` binaries.
#[cfg(not(target_arch = "wasm32"))]
pub mod tooling;

// The ECS and math the engine is built on.
/// Hash maps and sets with a fixed hasher: the same order every run.
pub use engine_core::collections;
pub use engine_core::{bevy_ecs, glam};
// Every environment section, for code listing them (`environment_sections!(callback)`).
pub use engine_core::environment_sections;

/// What games use, at the root and in the [`prelude`] alike: the engine and
/// its schedules, time, input, transforms, objects and assets, the look's
/// settings, the engine's features and their plugins, the ground, the
/// stores gameplay reads and the deterministic trigonometry (`math::sin`,
/// not `f32::sin`).
mod games {
    pub use engine_core::{
        Aabb, ActionId, ActionInput, AlphaMode, AmbientOcclusion, AnimatedProperty,
        AnimationChannel, AnimationClip, AnimationPlayer, AnimationPlugin, AnimationSystems, App,
        AssetBudget, AssetError, AssetLibrary, Atmosphere, Bloom, Camera, ClipHandle, Clouds,
        ContactShadows, Cover, Cube, Decal, DecalBlend, DecalSettings, Deformation,
        DeformationPlugin, Deformer, Engine, EngineConfig, EngineEntityCommands, Environment,
        Feature, Filter, FixedPostSimulation, FixedPreSimulation, FixedSimulation,
        FixedSimulationSystems, FixedTickEnd, FixedTickEndSystems, FixedTime, Floating,
        FrameReport, FrameTime, GlobalTransform, GlobalTransforms, Ground, GroundCover,
        GroundSample, Grounded, Grounding, HeightHeld, HierarchyError, InputState, Interior,
        InterpolateTransform, InterpolationError, KeyInterpolation, Keyframes, LevelOfDetail,
        LocalLight, LocalLightSettings, Material, MaterialHandle, Mesh, Mesh3d, MeshHandle,
        MeshVertex, NavAgent, NavAgents, NavStatus, NavSystems, NavWalk, Navigation,
        NavigationPlugin, OcclusionCulling, Occupancy, Ocean, OceanPlugin, OceanSystems, Parent,
        ParticleBlend, ParticleEmitter, ParticleSettings, Plugin, PluginGroup, Presentation,
        PresentationState, PresentedTime, PresentedTransform, PresentedTransforms, ReflectionProbe,
        ReflectionProbeSettings, Reflections, Refraction, Sampler, Shadows, Skin, Sky, SplineMesh,
        Spot, StreamSet, Streamed, StreamedId, StreamedItem, StreamingFocus, SubpixelCulling,
        SunShafts, Surface, TemporalAntialiasing, Terrain, TerrainPlugin, TerrainSample,
        TerrainSettings, Terrains, Texture, TextureHandle, TextureRef, TimeOfDay, TimeOfDayPlugin,
        Tint, TopLayer, Track, TrackDepth, Tracks, Transform, TransformEditError, Transmission,
        UvTransform, ViewSync, ViewSyncSystems, VirtualGeometry, Water, Wind, WorldCells,
        WorldChanges, WorldStreaming, WorldStreamingPlugin, WorldSync, WorldSyncSystems, Wrap,
        edit_transform, math, remove_streamed, reset_all_interpolation, reset_interpolation,
        set_parent, set_parents,
    };
}
pub use games::*;

// Lower-level names, at the root only: asset, animation and mesh details,
// the bounds settings and components validate against, the fallback mesh
// size `Mesh::from_virtual` takes by default, and what hosts and tools read.
pub use engine_core::{
    AnimationSample, DEFAULT_FALLBACK_TRIANGLES, EnvironmentSection, ErrorCode, Features,
    HierarchyReport, InvalidSetting, LodLevel, MAX_EMITTER_PARTICLES, MAX_HIERARCHY_DEPTH,
    MAX_KEPT_LEVELS, MAX_LIGHT_RANGE, MAX_OCEAN_WAVES, MAX_SHADOW_CASCADES, MAX_SPLINE_POINTS,
    MAX_SPOT_DEGREES, MAX_SPREAD_DEGREES, MAX_TEXTURE_DIMENSION, MAX_TRACK_DEPTH,
    MAX_TRANSFORM_NODES, MIN_SPOT_DEGREES, MeshLods, OceanPoint, OceanSurface, OceanWave,
    OceanWaves, PluginGroupBuilder, SplineChunks, TextureFormat, TransformNodeLimit, Viewport,
    VirtualMesh, has_feature, mesh_lods,
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
        RouteOptions, Stamp, StampOp, SurfaceMaterial, SurfaceProjection, Surroundings,
        SurroundingsStatus, TerrainBakes, TerrainEdit, TerrainPrefetch, TerrainSource,
        TerrainSurface, Tile, TileKey, WaterGrid,
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
        RIM_START, RenderDecal, RenderDecals, RenderDeformation, RenderEmitter, RenderFrame,
        RenderInstance, RenderJoint, RenderLight, RenderLights, RenderMaterial, RenderMesh,
        RenderParticles, RenderPlacementAsset, RenderReflectionProbe, RenderReflectionProbes,
        RenderSnapshot, RenderSurfaceMaps, RenderTerrain, RenderTerrains, RenderTexture,
        STREAMING_GRIDS, StreamingGrid, TransformIndex, TransformPropagation, VIRTUAL_MESH_MAGIC,
        carve_profile, presentation_bounds, propagate_transforms, rim_profile, valid_global_matrix,
    };
}

/// What a frame observed, by section: [`stats::FrameStats`] as hosts and
/// agents read it.
pub mod stats {
    pub use engine_core::{
        Applied, CellStats, CloudStats, CullingStats, DeformationStats, DepthFullRoute,
        DepthLightEntry, DepthOccupancyPage, DepthOccupancyPhaseStats, DepthOccupancyStats,
        DetailStats, Drawn, ErosionStats, FrameStats, GroundCoverStats, HydrologyStats, JobStats,
        LightStats, Names, NavStats, OccupancyStats, OceanStats, PathStats, ProbeStats,
        RenderStats, ShadowStats, SkyStats, TerrainRenderStats, TerrainStats, VirtualGeometryStats,
        WaterStats,
    };
}

pub mod host;

/// Optional, bounded development inspection. No inspection runs automatically.
#[cfg(feature = "dev")]
pub use engine_dev as dev;

mod app_host;
#[cfg(feature = "dev")]
mod dev_host;
mod game_host;
#[cfg(not(target_arch = "wasm32"))]
mod range_reads;
/// The renderer's shader modules by name, with their sources, as it
/// composes them: what tools compile and hash.
pub use engine_render::shaders::MODULES as SHADER_MODULES;
/// Where hosts present, and what renderers fail with.
pub use engine_render::{AdapterInfo, RenderError, RenderTarget};
/// A native window hosts present to, the GPU they ask for, and how tools
/// that must see the GPU interface's calls start its instances.
#[cfg(not(target_arch = "wasm32"))]
pub use engine_render::{GpuMark, GpuPreference, NativeWindow, start_native_instances_with};
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
    pub use crate::games::*;
    pub use crate::{EnginePlugins, Game, Inspectors};
    pub use engine_collision::{
        Collider, ColliderShape, CollisionLayers, CollisionPlugin, CollisionRay, CollisionSystems,
        CollisionWorld, QueryFilter, SensorFilter, ShapeCastOptions, refresh_collision_world,
    };
    pub use engine_core::glam::{Mat4, Quat, Vec2, Vec3, Vec4};
    // Bevy derives resolve this name in client scope when only the engine
    // facade is a direct dependency, so expose the crate beside the parts of
    // its prelude the engine runs: not its hierarchy (the engine's is
    // `Parent`), message queues (which need schedules it never runs) nor
    // `RemovedComponents` (removals between fixed ticks are lost; observers of
    // `Remove` see each).
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
    pub use engine_scene::{SceneDocument, SceneEntity, SceneInstance};
}
