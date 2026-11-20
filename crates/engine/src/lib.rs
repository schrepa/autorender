#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

pub use bevy_ecs;
pub use engine_scene as scene;
pub use engine_settings as settings;
pub use glam;
pub mod game;
pub use engine_core::Error;
pub use engine_settings::{Allowed, Expected, InvalidSetting, JS_ERROR_NAME};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[doc(hidden)]
pub mod surface;
pub use game::{Game, GameEntry, LoadedScene, Start};
#[cfg(not(target_arch = "wasm32"))]
pub use surface::runtime_typescript;
#[cfg(not(target_arch = "wasm32"))]
pub mod tooling;

pub use engine_command_line as command_line;
pub use engine_core::collections;

mod games {
    pub use engine_core::{
        AlphaMode, Asset, AssetId, AssetKind, AssetKinds, Assets, Brick, BrickList, Bricks, Camera,
        Clear, Clock, Compare, ComponentKind, Cull, CursorMode, DepthBias, DevSetting, Devices,
        DrawKind, Drawn, DrawnTable, Engine, Filter, FixedTime, FrameTime, GlobalTransforms,
        GroupReport, GroupStage, Host, InstanceValues, KindDecl, Layers, LightingModel, Listing,
        Material, MeshFile, Mobility, MouseButton, Mover, Moves, Node, Pace, Parent, Pause, Poses,
        Present, Projection, Rect, RenderState, Sampler, SceneComponents, Schedule, Show, Source,
        Spawner, Stats, Target, Teleport, Texture, TextureRef, TouchPhase, Transform, UvTransform,
        VertexStream, View, ViewFacts, ViewSkip, Views, math, set_parent,
    };
}
pub use games::*;

pub use engine_core::{ErrorCode, TextureFormat};

pub mod bricks {
    pub use engine_core::{BrickList, Listing};
    pub use engine_settings::{parse_json, parse_value, parse_value_at};
}

pub mod stats {
    #[cfg(feature = "dev")]
    pub use crate::frame_stats::frame_stats_schema;
    pub use crate::frame_stats::{FrameStats, GpuTime};
    pub use engine_core::RenderStats;
}

mod frame_stats;
pub mod host;

#[cfg(feature = "dev")]
pub use engine_dev as dev;

mod app_host;
#[cfg(feature = "dev")]
mod dev_host;
mod game_host;
pub use engine_render::{AdapterInfo, GpuAsk, GpuRequest, RenderTarget, SurfaceForm};
#[cfg(not(target_arch = "wasm32"))]
pub use engine_render::{NativeWindow, keep_pipeline_cache_in};
pub use game_host::GameHost;

#[cfg(target_arch = "wasm32")]
pub mod browser;

#[cfg(all(feature = "profiling", not(target_arch = "wasm32")))]
#[doc(hidden)]
pub mod __private {
    pub use engine_profile::TrackingAllocator;
}

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
pub mod __private {
    pub use js_sys;
    pub use wasm_bindgen;
    pub use wasm_bindgen::JsValue;
    pub use wasm_bindgen::prelude::wasm_bindgen as bindgen;
    pub use web_sys;

    #[cfg(feature = "profiling")]
    pub type PageAllocator = engine_profile::TrackingAllocator<engine_profile::GrowingHeap>;
    #[cfg(not(feature = "profiling"))]
    pub type PageAllocator = engine_profile::GrowingHeap;

    #[cfg(feature = "profiling")]
    #[must_use]
    pub const fn page_allocator() -> PageAllocator {
        engine_profile::TrackingAllocator::new(engine_profile::GrowingHeap::new())
    }

    #[cfg(not(feature = "profiling"))]
    #[must_use]
    pub const fn page_allocator() -> PageAllocator {
        engine_profile::GrowingHeap::new()
    }
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

pub mod prelude {
    pub use crate::games::*;
    pub use crate::host::TypedSettings;
    pub use crate::{Game, LoadedScene};
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
        GpuStruct, Pass, PassName, PassViews, ScenePass, TargetName, Writes, gpu_struct, settings,
        stats,
    };
    pub use engine_render::{
        Allocation, Bind, Buffer, DepthUse, Draw, GpuPart, Install, Lifetime, Pipeline, Prepare,
        RenderBricks, RenderPipeline, Shape, wgpu,
    };
    pub use engine_scene::{SceneDocument, SceneEntity, SceneInstance};
    pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
}
