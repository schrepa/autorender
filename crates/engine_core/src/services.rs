use bevy_ecs::component::{Component, ComponentId};
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use serde::{Serialize, de::DeserializeOwned};

use crate::{List, MeshHandle, settings};

#[derive(Clone, Copy)]
pub struct ComponentKind {
    key: &'static str,
    register: fn(&mut World) -> ComponentId,
    scene: engine_registry::SceneRead,
}

impl ComponentKind {
    pub fn of<C: Component + Clone + Serialize + DeserializeOwned>(key: &'static str) -> Self {
        todo!()
    }

    #[doc(hidden)]
    pub fn key(&self) -> &'static str {
        self.key
    }

    #[doc(hidden)]
    pub fn register(&self, world: &mut World) -> ComponentId {
        (self.register)(world)
    }

    #[doc(hidden)]
    pub fn scene(&self) -> engine_registry::SceneRead {
        self.scene
    }
}

pub struct SceneComponents;

impl List for SceneComponents {
    const NAME: &'static str = "scene_components";
    type Member = ComponentKind;
}

#[derive(Resource)]
#[component(immutable)]
pub struct View {
    cameras: engine_registry::ViewState,
}

impl View {
    pub(crate) fn empty() -> Self {
        Self {
            cameras: engine_registry::ViewState::default(),
        }
    }

    pub fn render_size(&self) -> glam::UVec2 {
        todo!()
    }

    pub fn view_projection(&self) -> glam::Mat4 {
        todo!()
    }

    pub fn detail_eye(&self) -> glam::Vec3 {
        todo!()
    }
}

settings! {
    #[section("detail")]
    pub struct Detail {
        pub enabled: bool = true => any,
        pub pixel_error: f32 = 1.0 => at_least(0.0),
        pub fade_seconds: f32 = 0.25 => at_least(0.0),
    }
}

#[derive(Resource)]
#[component(immutable)]
pub struct Poses {
    presented: engine_registry::PoseTable,
}

impl Poses {
    pub(crate) fn empty() -> Self {
        Self {
            presented: engine_registry::PoseTable::default(),
        }
    }

    pub fn of(&self, entity: Entity) -> Option<glam::Affine3A> {
        todo!()
    }
}

#[derive(Resource)]
#[component(immutable)]
pub struct AssetResidency {
    table: engine_registry::Residency,
}

impl AssetResidency {
    pub(crate) fn empty() -> Self {
        Self {
            table: engine_registry::Residency::default(),
        }
    }

    pub fn resident(&self, mesh: MeshHandle) -> bool {
        todo!()
    }

    pub fn generation(&self) -> u64 {
        todo!()
    }
}

#[derive(Resource, Clone, Copy, Debug)]
#[component(immutable)]
pub struct DeviceLimits {
    pub memory_budget: u64,
    pub cores: u32,
}
