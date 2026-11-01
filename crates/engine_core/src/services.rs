use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::component::{Component, ComponentId};
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Commands, Query, Res, SystemParam, SystemParamItem};
use bevy_ecs::world::World;
use engine_registry::{AuthoredId, OperationScene, SceneEntity};
use glam::{Mat4, Vec3};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

use crate::{Camera, Error, ErrorCode, List, Mesh, Operation, settings};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub root: bool,
    pub streamed: bool,
    pub world: Mat4,
}

pub struct Reach<'a> {
    pub world: Mat4,
    pub position: Vec3,
    pub from: f32,
    pub mesh_sizes: &'a dyn Fn(&str) -> f32,
}

impl Reach<'_> {
    pub fn mesh_size(&self, id: &str) -> f32 {
        (self.mesh_sizes)(id)
    }

    pub fn size_of(mesh: &Mesh) -> f32 {
        let bounds = mesh.bounds();
        (bounds.max() - bounds.min()).length()
    }

    pub fn of_box(&self, min: Vec3, max: Vec3) -> f32 {
        (0..8)
            .map(|corner| {
                let pick = |bit: u32, low: f32, high: f32| {
                    if corner & (1 << bit) == 0 { low } else { high }
                };
                let point = Vec3::new(
                    pick(0, min.x, max.x),
                    pick(1, min.y, max.y),
                    pick(2, min.z, max.z),
                );
                self.world.transform_point3(point).distance(self.position)
            })
            .fold(0.0, crate::math::max)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScenePath {
    pub id: String,
    pub origin: [f32; 3],
    pub points: Vec<[f32; 2]>,
    pub at: String,
}

type CheckHook = Arc<dyn Fn(&Value, &str) -> Result<(), Error> + Send + Sync>;
type PlacedHook = Arc<dyn Fn(&Value, Placed, &str) -> Result<(), Error> + Send + Sync>;
type FilesHook = Arc<dyn Fn(&Value) -> Vec<String> + Send + Sync>;
type ReachHook = Arc<dyn Fn(&Value, &Reach<'_>, &mut dyn FnMut(f32)) + Send + Sync>;
type PathsHook = Arc<dyn Fn(&Value, &str) -> Result<Vec<ScenePath>, Error> + Send + Sync>;

#[derive(Clone)]
pub struct ComponentKind {
    key: &'static str,
    register: fn(&mut World) -> ComponentId,
    scene: engine_registry::SceneRead,
    parse: fn(&Value, &str) -> Result<(), Error>,
    check: Option<CheckHook>,
    placed: Option<PlacedHook>,
    files: Option<FilesHook>,
    reach: Option<ReachHook>,
    paths: Option<PathsHook>,
    schema: Option<fn() -> Value>,
    example: Option<fn() -> Value>,
}

#[must_use]
pub struct ComponentKindOf<C> {
    kind: ComponentKind,
    component: PhantomData<fn() -> C>,
}

impl ComponentKind {
    pub fn of<C: Component + Clone + Serialize + DeserializeOwned>(
        key: &'static str,
    ) -> ComponentKindOf<C> {
        ComponentKindOf {
            kind: Self {
                key,
                register: World::register_component::<C>,
                scene: engine_registry::SceneRead(read_component::<C>),
                parse: parse_component::<C>,
                check: None,
                placed: None,
                files: None,
                reach: None,
                paths: None,
                schema: None,
                example: None,
            },
            component: PhantomData,
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

    #[doc(hidden)]
    pub fn validate(&self, value: &Value, path: &str) -> Result<(), Error> {
        match &self.check {
            Some(check) => check(value, path),
            None => (self.parse)(value, path),
        }
    }

    #[doc(hidden)]
    pub fn validate_placed(&self, value: &Value, placed: Placed, path: &str) -> Result<(), Error> {
        self.placed
            .as_ref()
            .map_or(Ok(()), |check| check(value, placed, path))
    }

    #[doc(hidden)]
    pub fn listed_files(&self, value: &Value) -> Vec<String> {
        self.files.as_ref().map_or_else(Vec::new, |files| files(value))
    }

    #[doc(hidden)]
    pub fn extend_reach(&self, value: &Value, reach: &Reach<'_>, extend: &mut dyn FnMut(f32)) {
        if let Some(of) = &self.reach {
            of(value, reach, extend);
        }
    }

    #[doc(hidden)]
    pub fn scene_paths(&self, value: &Value, path: &str) -> Result<Vec<ScenePath>, Error> {
        self.paths
            .as_ref()
            .map_or_else(|| Ok(Vec::new()), |paths| paths(value, path))
    }

    #[doc(hidden)]
    pub fn json_schema(&self) -> Value {
        self.schema.map_or(Value::Null, |schema| schema())
    }

    #[doc(hidden)]
    pub fn example_value(&self) -> Option<Value> {
        self.example.map(|example| example())
    }
}

impl<C: Component + Clone + Serialize + DeserializeOwned> ComponentKindOf<C> {
    pub fn check(mut self, check: fn(&C, &str) -> Result<(), Error>) -> Self {
        self.kind.check = Some(Arc::new(move |value: &Value, path: &str| {
            check(&parsed::<C>(value, path)?, path)
        }));
        self
    }

    pub fn check_placed(mut self, check: fn(&C, Placed, &str) -> Result<(), Error>) -> Self {
        self.kind.placed = Some(Arc::new(move |value: &Value, placed: Placed, path: &str| {
            check(&parsed::<C>(value, path)?, placed, path)
        }));
        self
    }

    pub fn files(mut self, files: fn(&C) -> Vec<String>) -> Self {
        self.kind.files = Some(Arc::new(move |value: &Value| {
            parsed::<C>(value, "$").map_or_else(|_| Vec::new(), |component| files(&component))
        }));
        self
    }

    pub fn reach(mut self, reach: fn(&C, &Reach<'_>, &mut dyn FnMut(f32))) -> Self {
        self.kind.reach = Some(Arc::new(
            move |value: &Value, at: &Reach<'_>, extend: &mut dyn FnMut(f32)| {
                if let Ok(component) = parsed::<C>(value, "$") {
                    reach(&component, at, extend);
                }
            },
        ));
        self
    }

    pub fn paths(mut self, paths: fn(&C, &str) -> Result<Vec<ScenePath>, Error>) -> Self {
        self.kind.paths = Some(Arc::new(move |value: &Value, path: &str| {
            paths(&parsed::<C>(value, path)?, path)
        }));
        self
    }

    pub fn schema(mut self, schema: fn() -> Value) -> Self {
        self.kind.schema = Some(schema);
        self
    }

    pub fn example(mut self, example: fn() -> Value) -> Self {
        self.kind.example = Some(example);
        self
    }
}

impl<C> From<ComponentKindOf<C>> for ComponentKind {
    fn from(of: ComponentKindOf<C>) -> Self {
        of.kind
    }
}

fn parsed<C: DeserializeOwned>(value: &Value, path: &str) -> Result<C, Error> {
    engine_settings::parse_value_at(value.clone(), path)
}

fn parse_component<C: DeserializeOwned>(value: &Value, path: &str) -> Result<(), Error> {
    parsed::<C>(value, path).map(drop)
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
        .flat_map(|members| members.all().iter())
}

#[doc(hidden)]
pub fn scene_component(world: &World, key: &str) -> Option<ComponentKind> {
    scene_components(world).find(|kind| kind.key == key).cloned()
}

#[derive(SystemParam)]
pub struct SceneIds<'w, 's> {
    scene: Option<Res<'w, OperationScene>>,
    entities: Query<'w, 's, (Entity, &'static SceneEntity)>,
}

impl SceneIds<'_, '_> {
    pub fn len(&self) -> usize {
        self.scene
            .as_ref()
            .map_or(0, |scene| scene.entity_count())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn entity(&self, id: &str) -> Option<Entity> {
        let scene = self.scene.as_ref()?;
        scene.entities.get(id).copied().or_else(|| {
            if !scene.streamed {
                return None;
            }
            self.entities
                .iter()
                .find(|(_, authored)| authored.authored_id == id && authored.scene_id == scene.id)
                .map(|(entity, _)| entity)
        })
    }

    pub fn id(&self, entity: Entity) -> Option<&str> {
        self.scene.as_ref()?;
        self.entities
            .get(entity)
            .ok()
            .map(|(_, authored)| authored.authored_id.as_str())
    }

    pub fn authored(&self, entity: Entity) -> Option<AuthoredId> {
        self.scene.as_ref()?;
        self.entities
            .get(entity)
            .ok()
            .map(|(_, authored)| authored.authored())
    }

    pub fn holds(&self, authored: &AuthoredId, entity: Entity) -> bool {
        self.scene
            .as_ref()
            .is_some_and(|scene| scene.holds(authored, entity))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Authored {
    pub id: String,
    pub name: String,
    pub tags: Vec<String>,
    pub parent: Option<String>,
    pub prefab_id: Option<String>,
    pub prefab_instance: Option<String>,
    pub prefab_local_id: Option<String>,
}

#[derive(Resource, Clone)]
#[component(immutable)]
pub struct SceneContext {
    pub id: String,
    pub assets: Arc<engine_assets::SceneAssets>,
    pub prefabs: Arc<BTreeMap<String, Value>>,
    pub paths: Arc<[ScenePath]>,
    pub entities: Arc<BTreeMap<String, Entity>>,
}

impl SceneContext {
    pub fn scope<R>(&self, run: impl FnOnce() -> R) -> R {
        self.assets.scope(run)
    }

    pub fn entity(&self, id: &str) -> Option<Entity> {
        self.entities.get(id).copied()
    }

    pub fn prefab(&self, id: &str) -> Option<&Value> {
        self.prefabs.get(id)
    }
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

    pub fn display_size(&self) -> glam::UVec2 {
        self.cameras.display_size()
    }

    pub fn render_share(&self) -> f32 {
        let (render, display) = (self.render_size(), self.display_size());
        render.y as f32 / display.y.max(1) as f32
    }

    pub fn view_projection(&self) -> glam::Mat4 {
        self.cameras.view_projection()
    }

    pub fn detail_eye(&self) -> glam::Vec3 {
        self.cameras.detail_eye()
    }

    pub fn display(&self) -> bool {
        self.cameras.facts().display
    }

    pub fn exact(&self) -> bool {
        self.cameras.facts().exact
    }

    pub fn temporal(&self) -> bool {
        self.cameras.facts().temporal
    }

    pub fn capture(&self) -> bool {
        self.cameras.facts().capture
    }

    pub fn debug(&self) -> bool {
        self.cameras.facts().debug
    }

    pub fn scene(&self) -> u64 {
        self.cameras.scene()
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
        pub pixel_error: f32 = 1.0 => range(0.25, 64.0),
        pub fade_seconds: f32 = 0.5 => range(0.0, 4.0),
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

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
#[component(immutable)]
pub struct DeviceLimits {
    pub memory_budget: u64,
    pub cores: u32,
    pub storage_room: u64,
    pub texture_layers: u32,
    pub max_texture_dimension_2d: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub indirect_first_instance: bool,
}

impl DeviceLimits {
    #[doc(hidden)]
    pub fn without_device(cores: u32) -> Self {
        Self {
            memory_budget: u64::MAX,
            cores,
            storage_room: 0,
            texture_layers: 0,
            max_texture_dimension_2d: 0,
            max_compute_workgroups_per_dimension: 0,
            indirect_first_instance: false,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default)]
#[component(immutable)]
pub struct WorkClock {
    pub reading: Option<fn() -> f64>,
}

impl WorkClock {
    pub fn now(&self) -> Option<f64> {
        self.reading.map(|clock| clock())
    }
}
