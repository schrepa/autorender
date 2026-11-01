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
pub use engine_settings::{Allowed, Expected, JS_ERROR_NAME};

pub type Result<T, E = Error> = std::result::Result<T, E>;
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
    pub use engine_core::{
        Aabb, ActionId, ActionInput, AlphaImage, AlphaMode, AssetBudget, AssetLibrary, BrickList,
        Camera, DynamicResolution, DynamicResolutionStats, Engine, EngineConfig,
        EngineEntityCommands, Filter, FixedTime, FrameReport, FrameTime, GlobalTransform,
        GlobalTransforms, Grounding, InputState, InterpolateTransform, Listing, MakeSources,
        MakeTextures, Material, MaterialHandle, Mesh, Mesh3d, MeshHandle, MeshVertex, Parent,
        PresentedTime, PresentedTransform, PresentedTransforms, ProvidedAssets, Resolution,
        ResolutionStats, Sampler, Surface, Texture, TextureHandle, TextureRef, TextureSet,
        TextureSource, TextureUsage, TopLayer, Transform, UvTransform, Wrap, edit_transform, math,
        reset_all_interpolation, reset_interpolation, set_parent, set_parents, trim_cards,
    };
}
pub use games::*;

pub use engine_core::{
    ErrorCode, HierarchyReport, InvalidSetting, LodLevel, MAX_HIERARCHY_DEPTH,
    MAX_TEXTURE_DIMENSION, MAX_TRANSFORM_NODES, MeshLods, TextureFamily, TextureFormat,
    TextureSampling, TransformNodeLimit, Viewport,
};

pub mod bricks {
    pub use engine_core::{BrickList, Listing};
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
pub use engine_render::{AdapterInfo, RenderTarget};
#[cfg(not(target_arch = "wasm32"))]
pub use engine_render::{GpuMark, GpuPreference, NativeWindow, start_native_instances_with};
pub use game_host::GameHost;

pub fn shader_modules(
    bricks: impl FnOnce(BrickList) -> BrickList,
) -> Result<Vec<engine_render::linker::ShaderModule>> {
    use engine_render::linker::ShaderModule;
    use engine_render::places::{PLACE_OWNER, place_module};
    let engine = engine(bricks)?;
    let mut modules = engine_render::shaders::MODULES.to_vec();
    let Some(registry) = engine
        .app
        .world()
        .get_resource::<engine_core::BrickRegistry>()
    else {
        return Ok(modules);
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
    Ok(modules)
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
            Populated, Query, Res, ResMut, Single, System,
        },
        world::{EntityMut, EntityRef, EntityWorldMut, FromWorld, World},
    };
    pub use engine_scene::{SceneDocument, SceneEntity, SceneInstance};
    pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
}
