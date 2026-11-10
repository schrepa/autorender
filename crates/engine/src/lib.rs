#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

pub use bevy_ecs;
pub use engine_profile as profiling;
pub use engine_scene as scene;
pub use engine_settings as settings;
pub use glam;
pub mod game;
pub use engine_core::{Detail, Error};
pub use engine_settings::{Allowed, Expected, InvalidSetting, JS_ERROR_NAME};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[doc(hidden)]
pub mod surface;
pub use game::{Arrival, DefaultScene, Game, GameEntry, GameFiles, LoadedScene, engine};
#[cfg(not(target_arch = "wasm32"))]
pub use surface::runtime_typescript;
#[cfg(not(target_arch = "wasm32"))]
pub mod tooling;

pub use engine_command_line as command_line;
pub use engine_core::collections;
pub use engine_core::smoke_eq;

mod games {
    pub use crate::frame_stats::ResolutionStats;
    pub use crate::pacing::Pacing;
    pub use crate::resolution::Resolution;
    pub use engine_core::{
        Aabb, AlphaMode, Asks, AssetBudget, AssetLibrary, Attribute, Attributes, Binding,
        BindingSource, Brick, BrickList, Bricks, Button, Camera, Clear, Clock, ColorSpace, Compare,
        ComponentKind, ContextAsk, Contexts, Cull, CursorAsk, CursorMode, CursorRequests,
        DepthBias, Devices, DrawKind, Drawn, DrawnProjection, DrawnState, DrawnTable, Engine,
        EngineConfig, Filter, FixedTime, FrameLimits, FrameTime, GeometryHandle, GlobalTransform,
        GlobalTransforms, HandOver, Hit, Input, InputSettings, InstanceValues, KindDecl, KindSkip,
        Layers, LightingModel, Listing, MakeTextures, Material, MaterialHandle, Mesh, MeshVertex,
        Mobility, MouseButton, Mover, MoverId, Moves, Node, NodeWord, Operation, Pace, PadButton,
        Parent, Pause, PauseReason, Picking, Player, Poses, Projection, ProvidedAssets, Ray, Rect,
        RenderState, Sampler, SceneComponents, Schedule, Section, Smoke, Source, Span, Spawner,
        Stick, SurfaceSlot, Target, Teleport, Texture, TextureHandle, TextureRef, TextureSet,
        TextureSource, TextureUsage, TimeScale, TouchPhase, Transform, Trigger, UvTransform, Verb,
        VerbId, VerbShape, Verbs, View, ViewCut, ViewFacts, ViewHistory, Views, Wrap, main_camera,
        make_fresh, math, place_camera, set_parent, set_parents,
    };
}
pub use games::*;

pub use engine_core::{
    ErrorCode, HierarchyReport, LodLevel, MAX_HIERARCHY_DEPTH, MAX_TRANSFORM_NODES, MeshLods,
    TextureFamily, TextureFormat, TextureSampling, TransformNodeLimit,
};

pub mod bricks {
    pub use engine_core::{BrickList, Listing};
    pub use engine_registry::{BrickCost, BrickCosts, BrickRow, BrickTable};
    pub use engine_settings::{parse_json, parse_value, parse_value_at};
}

pub mod stats {
    pub use crate::frame_stats::FrameStats;
    #[cfg(feature = "schema")]
    pub use crate::frame_stats::frame_stats_schema;
    pub use engine_core::{Cut, JobStats, RenderStats};
}

mod frame_stats;
pub mod host;
#[cfg(feature = "dev")]
pub mod smokes;

#[cfg(feature = "dev")]
pub use engine_dev as dev;

mod app_host;
#[cfg(feature = "dev")]
mod dev_host;
mod game_host;
mod pacing;
mod resolution;
#[cfg(feature = "profiling")]
pub use app_host::{PROFILE_DURATION_MS, PROFILE_EVENTS};
pub use engine_render::{AdapterInfo, RenderTarget};
#[cfg(not(target_arch = "wasm32"))]
pub use engine_render::{
    GpuPreference, NativeWindow, keep_pipeline_cache_in, start_native_instances_with,
};
pub use game_host::GameHost;

pub fn scene_kinds(
    bricks: impl FnOnce(BrickList) -> BrickList,
) -> Result<engine_scene::SceneKinds> {
    engine_scene::SceneKinds::listed(&bricks(BrickList::default()))
}

#[doc(hidden)]
pub use engine_worker as __worker;

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
    pub use crate::games::*;
    pub use crate::host::TypedSettings;
    pub use crate::{Arrival, Game, GameFiles, LoadedScene};
    pub use bevy_ecs::{
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
            ApplyDeferred, IntoScheduleConfigs, IntoSystemSet, Schedules, SystemCondition,
            SystemSet, common_conditions::*,
        },
        system::{
            Command, Commands, EntityCommand, EntityCommands, In, IntoSystem, Local, ParamSet,
            Populated, Query, Res, ResMut, Single, System, SystemParam,
        },
        world::{EntityMut, EntityRef, EntityWorldMut, FromWorld, World},
    };
    pub use engine_core::{
        AttachPoint, GpuStruct, Pass, PassName, PassViews, ScenePass, SlotSet, TargetName, Writes,
        gpu_struct, settings, stats,
    };
    pub use engine_render::{
        Allocation, Bind, Buffer, DepthUse, Draw, GpuPart, Install, Lifetime, Pipeline, Prepare,
        RenderBricks, RenderPipeline, SceneDraw, Shape, wgpu,
    };
    pub use engine_scene::{SceneDocument, SceneEntity, SceneInstance};
    pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
}
