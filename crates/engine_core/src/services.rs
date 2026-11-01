use bevy_ecs::component::{Component, ComponentId};
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Res, SystemParamItem};
use bevy_ecs::world::World;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{Camera, Error, ErrorCode, List, MeshHandle, Operation, settings};

#[derive(Clone, Copy)]
pub struct ComponentKind {
    key: &'static str,
    register: fn(&mut World) -> ComponentId,
    scene: engine_registry::SceneRead,
}

impl ComponentKind {
    pub fn of<C: Component + Clone + Serialize + DeserializeOwned>(key: &'static str) -> Self {
        Self {
            key,
            register: World::register_component::<C>,
            scene: engine_registry::SceneRead(read_component::<C>),
        }
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

    #[doc(hidden)]
    pub fn read(&self, world: &mut World, entity: Entity, value: &Value) -> Result<(), Error> {
        (self.scene.0)(world, entity, value)
    }
}

fn read_component<C: Component + DeserializeOwned>(
    world: &mut World,
    entity: Entity,
    value: &Value,
) -> Result<(), Error> {
    let component: C = engine_settings::parse_value(value.clone())?;
    let mut target = world
        .get_entity_mut(entity)
        .map_err(|_| Error::new(ErrorCode::UnknownEntity, "$").got(entity.to_string()))?;
    target.insert(component);
    Ok(())
}

pub struct SceneComponents;

impl List for SceneComponents {
    const NAME: &'static str = "scene_components";
    type Member = ComponentKind;
}

#[doc(hidden)]
pub fn scene_components(world: &World) -> impl Iterator<Item = &ComponentKind> + '_ {
    world
        .get_resource::<engine_registry::MembersOf<SceneComponents>>()
        .into_iter()
        .flat_map(|members| members.live())
}

#[doc(hidden)]
pub fn scene_component(world: &World, key: &str) -> Option<ComponentKind> {
    scene_components(world).find(|kind| kind.key == key).copied()
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
        self.cameras.render_size()
    }

    pub fn view_projection(&self) -> glam::Mat4 {
        self.cameras.view_projection()
    }

    pub fn detail_eye(&self) -> glam::Vec3 {
        self.cameras.detail_eye()
    }

    #[doc(hidden)]
    pub fn state(&self) -> &engine_registry::ViewState {
        &self.cameras
    }

    #[doc(hidden)]
    pub fn state_mut(&mut self) -> &mut engine_registry::ViewState {
        &mut self.cameras
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ViewControl {
    #[serde(default)]
    freeze: Option<bool>,
}

impl Operation for ViewControl {
    const NAME: &'static str = "view";
    type Params = (Res<'static, View>, Res<'static, Camera>, Commands<'static, 'static>);
    type Request = Self;

    fn run(params: SystemParamItem<'_, '_, Self::Params>, request: Self) -> Result<Value, Error> {
        let (view, camera, mut commands) = params;
        let frozen = match request.freeze {
            Some(true) => Some(view.cameras.frozen().unwrap_or_else(|| camera.eye())),
            Some(false) => None,
            None => view.cameras.frozen(),
        };
        if frozen != view.cameras.frozen() {
            commands.queue(move |world: &mut World| {
                crate::modify(world, |view: &mut View| view.cameras.freeze(frozen));
            });
        }
        Ok(json!({ "frozen_at": frozen.map(|eye| eye.to_array()) }))
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
        self.presented.of(entity)
    }

    pub(crate) fn take(&mut self) -> engine_registry::PoseTable {
        std::mem::take(&mut self.presented)
    }

    pub(crate) fn hold(&mut self, presented: engine_registry::PoseTable) {
        self.presented = presented;
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
        self.table.resident(mesh)
    }

    pub fn generation(&self) -> u64 {
        self.table.generation()
    }

    #[doc(hidden)]
    pub fn table_mut(&mut self) -> &mut engine_registry::Residency {
        &mut self.table
    }
}

#[derive(Resource, Clone, Copy, Debug)]
#[component(immutable)]
pub struct DeviceLimits {
    pub memory_budget: u64,
    pub cores: u32,
}
