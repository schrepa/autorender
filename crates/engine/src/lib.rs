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

pub const DYNAMIC_RESOLUTION: &str = "dynamic_resolution.enabled";
#[doc(hidden)]
pub mod surface;
pub use game::{Game, Inspectors, engine};
#[cfg(not(target_arch = "wasm32"))]
pub use surface::runtime_typescript;
#[cfg(not(target_arch = "wasm32"))]
pub mod tooling;

pub use engine_core::collections;
pub use engine_core::command_line;
pub use engine_core::smoke_eq;

mod games {
    pub use crate::host::MoveDirection;
    pub use crate::resolution::{DynamicResolution, Resolution};
    pub use engine_core::{
        Aabb, ActionId, ActionInput, AlphaImage, AlphaMode, AssetBudget, AssetLibrary, AxisId,
        AxisInput, BrickList, Camera, CursorMode, DynamicResolutionStats, Engine, EngineConfig,
        Filter, FixedTime, FrameReport, FrameTime, GlobalTransform, GlobalTransforms, Grounding,
        InterpolateTransform, Listing, MakeSources, MakeTextures, Material, MaterialHandle,
        MenuInput, Mesh, Mesh3d, MeshHandle, MeshVertex, MoveInput, Parent, PointerInput,
        PresentedTransforms, Projection, ProvidedAssets, ResolutionStats, Sampler, Surface,
        Texture, TextureHandle, TextureRef, TextureSet, TextureSource, TextureUsage, TopLayer,
        Transform, UvTransform, Wrap, edit_transform, math, reset_all_interpolation,
        reset_interpolation, set_parent, set_parents, trim_cards,
    };
}
pub use games::*;

pub use engine_core::{
    ErrorCode, HierarchyReport, LodLevel, MAX_HIERARCHY_DEPTH, MAX_TEXTURE_DIMENSION,
    MAX_TRANSFORM_NODES, MeshLods, TextureFamily, TextureFormat, TextureSampling,
    TransformNodeLimit, Viewport,
};

pub mod bricks {
    pub use engine_core::{BrickList, Listing};
    pub use engine_registry::{BrickCost, BrickCosts, BrickRow, BrickTable};
    pub use engine_settings::{parse_json, parse_value, parse_value_at};
}

pub mod stats {
    pub use engine_core::{Cut, FrameStats, JobStats, RenderStats};
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
mod resolution;
pub use engine_render::{AdapterInfo, RenderTarget};
#[cfg(not(target_arch = "wasm32"))]
pub use engine_render::{GpuMark, GpuPreference, NativeWindow, start_native_instances_with};
pub use game_host::GameHost;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderModule {
    pub path: &'static str,
    pub source: &'static str,
}

pub fn shader_modules(bricks: impl FnOnce(BrickList) -> BrickList) -> Result<Vec<ShaderModule>> {
    let list = bricks(BrickList::default());
    let mut modules: Vec<ShaderModule> = Vec::new();
    for (path, source) in engine_render::core_modules() {
        add_module(&mut modules, path, Box::leak(source.into_boxed_str()));
    }
    for listed in list.listed() {
        for module in listed.source().modules() {
            add_module(&mut modules, module.path(), module.text());
        }
    }
    let engine = game::listed(list)?;
    let Some(registry) = engine
        .app
        .world()
        .get_resource::<engine_registry::Registry>()
    else {
        return Ok(modules);
    };
    for place in registry.catalog() {
        if let Some(source) = place.wgsl() {
            let path = format!("place::{}", place.name());
            add_module(&mut modules, Box::leak(path.into_boxed_str()), source);
        }
    }
    Ok(modules)
}

pub fn scene_kinds(
    bricks: impl FnOnce(BrickList) -> BrickList,
) -> Result<engine_scene::SceneKinds> {
    engine_scene::SceneKinds::listed(&bricks(BrickList::default()))
}

fn add_module(modules: &mut Vec<ShaderModule>, path: &'static str, source: &'static str) {
    if !modules.iter().any(|known| known.path == path) {
        modules.push(ShaderModule { path, source });
    }
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
    pub use crate::{Game, Inspectors};
    pub use bevy_ecs::{
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
            Populated, Query, Res, ResMut, Single, System, SystemParam,
        },
        world::{EntityMut, EntityRef, EntityWorldMut, FromWorld, World},
    };
    pub use engine_scene::{SceneDocument, SceneEntity, SceneInstance};
    pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
}
