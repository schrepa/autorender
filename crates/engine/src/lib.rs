//! Browser-first 3D engine façade. Gameplay intentionally uses Bevy ECS.
/// Opt-in collision geometry, filtered queries, and fixed-tick sensor events.
pub use engine_collision as collision;
pub use engine_core::*;
/// Optional bounded profiling scopes; calls compile away without `profiling`.
pub use engine_profile as profiling;
/// Versioned authoring, validation, and explicit scene ownership.
pub use engine_scene as scene;
pub mod camera;

/// Optional, bounded development inspection. No inspection runs automatically.
#[cfg(feature = "dev")]
pub use engine_dev as dev;

#[cfg(all(feature = "dev", target_arch = "wasm32"))]
pub mod dev_browser;

/// Browser lifecycle and rendering composition, owned by the main thread.
#[cfg(target_arch = "wasm32")]
pub mod browser;

/// Common gameplay types; raw GPU and JavaScript types stay out of gameplay.
pub mod prelude {
    pub use crate::camera::{CameraSystems, FlyCameraController, FlyCameraPlugin};
    pub use engine_collision::{
        Collider, ColliderShape, CollisionLayers, CollisionPlugin, CollisionRay, CollisionSystems,
        CollisionWorld, QueryFilter, SensorFilter, ShapeCastOptions, refresh_collision_world,
    };
    pub use engine_core::prelude::*;
    pub use engine_scene::{SceneDocument, SceneEntity, SceneInstance};
}
