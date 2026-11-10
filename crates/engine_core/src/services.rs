use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::component::{Component, ComponentId};
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::{Discard, Insert};
use bevy_ecs::observer::On;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Query, Res, ResMut, SystemParam};
use bevy_ecs::world::World;
use engine_registry::{AuthoredId, JobList, OperationScene, SceneEntity};
use glam::Mat4;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Error, ErrorCode, Expected, List, settings};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub world: Mat4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Placement {
    #[default]
    Anywhere,
    Roots(ErrorCode),
}

pub type SceneBuffers = BTreeMap<String, Arc<[u8]>>;

pub struct SceneCx<'a> {
    list: Option<&'a JobList>,
    scene: Option<&'a SceneContext>,
    files: Option<&'a SceneBuffers>,
}

impl<'a> SceneCx<'a> {
    #[doc(hidden)]
    pub fn new(
        list: Option<&'a JobList>,
        scene: Option<&'a SceneContext>,
        files: Option<&'a SceneBuffers>,
    ) -> Self {
        Self { list, scene, files }
    }

    #[doc(hidden)]
    pub fn of(world: &'a World) -> Self {
        Self {
            list: world
                .get_resource::<crate::JobExchange>()
                .map(crate::JobExchange::list),
            scene: world.get_resource::<SceneContext>(),
            files: None,
        }
    }
}

impl SceneCx<'_> {
    pub fn members<L: List>(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.list.into_iter().flat_map(JobList::members::<L>)
    }

    pub fn scene(&self) -> Option<&SceneContext> {
        self.scene
    }

    pub fn buffer(&self, uri: &str, path: &str) -> Result<Option<Arc<[u8]>>, Error> {
        let files = self
            .files
            .or_else(|| self.scene.and_then(|scene| scene.files.as_deref()));
        files
            .map(|files| {
                files.get(uri).cloned().ok_or_else(|| {
                    Error::new(ErrorCode::MissingFile, path)
                        .expected(Expected::File)
                        .got(uri)
                })
            })
            .transpose()
    }
}

#[doc(hidden)]
pub struct Prepared(Box<dyn Any + Send + Sync>);

type CheckHook = Arc<dyn Fn(&Value, &str) -> Result<(), Error> + Send + Sync>;
type PlacedHook = Arc<dyn Fn(&Value, Placed, &str) -> Result<(), Error> + Send + Sync>;
type FilesHook = Arc<dyn Fn(&Value, &SceneCx<'_>) -> Vec<String> + Send + Sync>;
type SchemaHook = Arc<dyn Fn(&SceneCx<'_>) -> Value + Send + Sync>;
type PrepareHook = Arc<dyn Fn(&Value, &SceneCx<'_>, &str) -> Result<Prepared, Error> + Send + Sync>;
type ViewHook = Arc<dyn Fn(&World, Entity) -> Option<Value> + Send + Sync>;

#[derive(Clone)]
pub struct ComponentKind {
    key: &'static str,
    register: fn(&mut World) -> ComponentId,
    scene: engine_registry::SceneRead,
    parse: fn(&Value, &str) -> Result<(), Error>,
    value: fn(&World, Entity) -> Option<Value>,
    prepared: fn(&Value, &str) -> Result<Prepared, Error>,
    insert: fn(&mut World, Entity, Prepared) -> Result<(), Error>,
    remove: fn(&mut World, Entity),
    placement: Placement,
    check: Option<CheckHook>,
    placed: Option<PlacedHook>,
    files: Option<FilesHook>,
    schema: Option<SchemaHook>,
    prepare: Option<PrepareHook>,
    view: Option<ViewHook>,
    view_schema: Option<fn() -> Value>,
}

#[must_use]
pub struct ComponentKindOf<C> {
    kind: ComponentKind,
    component: PhantomData<fn() -> C>,
}

pub trait SceneForm: Clone + Serialize + DeserializeOwned + Send + Sync + 'static {
    type Component: Component;

    fn insert(self, world: &mut World, entity: Entity) -> Result<(), Error>;

    fn read(world: &World, entity: Entity) -> Option<Self>;

    fn remove(world: &mut World, entity: Entity) {
        remove_component::<Self::Component>(world, entity);
    }
}

impl ComponentKind {
    pub fn form<F: SceneForm>(key: &'static str) -> ComponentKindOf<F> {
        ComponentKindOf {
            kind: Self {
                key,
                register: World::register_component::<F::Component>,
                scene: engine_registry::SceneRead(read_form::<F>),
                parse: parse_component::<F>,
                value: form_value::<F>,
                prepared: prepared_component::<F>,
                insert: insert_form::<F>,
                remove: F::remove,
                placement: Placement::Anywhere,
                check: None,
                placed: None,
                files: None,
                schema: None,
                prepare: None,
                view: None,
                view_schema: None,
            },
            component: PhantomData,
        }
    }

    pub fn of<C: Component + Clone + Serialize + DeserializeOwned>(
        key: &'static str,
    ) -> ComponentKindOf<C> {
        ComponentKindOf {
            kind: Self {
                key,
                register: World::register_component::<C>,
                scene: engine_registry::SceneRead(read_component::<C>),
                parse: parse_component::<C>,
                value: component_value::<C>,
                prepared: prepared_component::<C>,
                insert: insert_component::<C>,
                remove: remove_component::<C>,
                placement: Placement::Anywhere,
                check: None,
                placed: None,
                files: None,
                schema: None,
                prepare: None,
                view: None,
                view_schema: None,
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
    pub fn registrar(&self) -> fn(&mut World) -> ComponentId {
        self.register
    }

    #[doc(hidden)]
    pub fn scene(&self) -> engine_registry::SceneRead {
        self.scene
    }

    #[doc(hidden)]
    pub fn placement(&self) -> Placement {
        self.placement
    }

    #[doc(hidden)]
    pub fn read(&self, world: &mut World, entity: Entity, value: &Value) -> Result<(), Error> {
        if self.prepare.is_none() {
            return (self.scene.0)(world, entity, value);
        }
        let list = world
            .get_resource::<crate::JobExchange>()
            .map(crate::JobExchange::shared_list);
        let scene = world.get_resource::<SceneContext>().cloned();
        let prepared = self.prepare_value(
            value,
            &SceneCx::new(list.as_deref(), scene.as_ref(), None),
            "$",
        )?;
        (self.insert)(world, entity, prepared)
    }

    #[doc(hidden)]
    pub fn prepare_value(
        &self,
        value: &Value,
        cx: &SceneCx<'_>,
        path: &str,
    ) -> Result<Prepared, Error> {
        match &self.prepare {
            Some(prepare) => prepare(value, cx, path),
            None => (self.prepared)(value, path),
        }
    }

    #[doc(hidden)]
    pub fn validate_prepared(
        &self,
        value: &Value,
        cx: &SceneCx<'_>,
        path: &str,
    ) -> Result<(), Error> {
        self.prepare
            .as_ref()
            .map_or(Ok(()), |prepare| prepare(value, cx, path).map(drop))
    }

    #[doc(hidden)]
    pub fn insert_prepared(
        &self,
        world: &mut World,
        entity: Entity,
        prepared: Prepared,
    ) -> Result<(), Error> {
        (self.insert)(world, entity, prepared)
    }

    #[doc(hidden)]
    pub fn remove(&self, world: &mut World, entity: Entity) {
        (self.remove)(world, entity);
    }

    #[doc(hidden)]
    pub fn value(&self, world: &World, entity: Entity) -> Option<Value> {
        match &self.view {
            Some(view) => view(world, entity),
            None => (self.value)(world, entity),
        }
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
    pub fn listed_files_in(&self, value: &Value, cx: &SceneCx<'_>) -> Vec<String> {
        self.files
            .as_ref()
            .map_or_else(Vec::new, |files| files(value, cx))
    }

    #[doc(hidden)]
    pub fn json_schema_in(&self, cx: &SceneCx<'_>) -> Value {
        self.schema
            .as_ref()
            .map_or(Value::Null, |schema| schema(cx))
    }

    #[doc(hidden)]
    pub fn view_schema(&self) -> Option<Value> {
        self.view_schema.map(|schema| schema())
    }
}

impl<C: Clone + Serialize + DeserializeOwned + Send + Sync + 'static> ComponentKindOf<C> {
    pub fn check(mut self, check: fn(&C, &str) -> Result<(), Error>) -> Self {
        self.kind.check = Some(Arc::new(move |value: &Value, path: &str| {
            check(&parsed::<C>(value, path)?, path)
        }));
        self
    }

    pub fn check_placed(mut self, check: fn(&C, Placed, &str) -> Result<(), Error>) -> Self {
        self.kind.placed = Some(Arc::new(
            move |value: &Value, placed: Placed, path: &str| {
                check(&parsed::<C>(value, path)?, placed, path)
            },
        ));
        self
    }

    pub fn placement(mut self, placement: Placement) -> Self {
        self.kind.placement = placement;
        self
    }

    pub fn files(mut self, files: fn(&C) -> Vec<String>) -> Self {
        self.kind.files = Some(Arc::new(move |value: &Value, _: &SceneCx<'_>| {
            parsed::<C>(value, "$").map_or_else(|_| Vec::new(), |component| files(&component))
        }));
        self
    }

    pub fn files_in(mut self, files: fn(&C, &SceneCx<'_>) -> Vec<String>) -> Self {
        self.kind.files = Some(Arc::new(move |value: &Value, cx: &SceneCx<'_>| {
            parsed::<C>(value, "$").map_or_else(|_| Vec::new(), |component| files(&component, cx))
        }));
        self
    }

    pub fn schema(mut self, schema: fn() -> Value) -> Self {
        self.kind.schema = Some(Arc::new(move |_: &SceneCx<'_>| schema()));
        self
    }

    pub fn schema_in(mut self, schema: fn(&SceneCx<'_>) -> Value) -> Self {
        self.kind.schema = Some(Arc::new(schema));
        self
    }

    pub fn prepare(mut self, prepare: fn(C, &SceneCx<'_>, &str) -> Result<C, Error>) -> Self {
        self.kind.prepare = Some(Arc::new(
            move |value: &Value, cx: &SceneCx<'_>, path: &str| {
                let component = prepare(parsed::<C>(value, path)?, cx, path)?;
                Ok(Prepared(Box::new(component)))
            },
        ));
        self
    }
}

impl<C: Component + Clone + Serialize + DeserializeOwned> ComponentKindOf<C> {
    pub fn view<A: Component, V: Serialize + schemars::JsonSchema + 'static>(
        mut self,
        view: fn(&C, Option<&A>) -> V,
    ) -> Self {
        self.kind.view_schema = Some(view_schema_of::<V>);
        self.kind.view = Some(Arc::new(move |world: &World, entity: Entity| {
            let component = world.get::<C>(entity)?;
            let write = || serde_json::to_value(view(component, world.get::<A>(entity))).ok();
            match world.get_resource::<SceneContext>() {
                Some(context) => context.scope(write),
                None => write(),
            }
        }));
        self
    }
}

impl<C> From<ComponentKindOf<C>> for ComponentKind {
    fn from(of: ComponentKindOf<C>) -> Self {
        of.kind
    }
}

fn view_schema_of<V: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(V)).unwrap_or(Value::Null)
}

fn parsed<C: DeserializeOwned>(value: &Value, path: &str) -> Result<C, Error> {
    engine_settings::parse_value_at(value.clone(), path)
}

fn parse_component<C: DeserializeOwned>(value: &Value, path: &str) -> Result<(), Error> {
    parsed::<C>(value, path).map(drop)
}

fn component_value<C: Component + Serialize>(world: &World, entity: Entity) -> Option<Value> {
    let component = world.get::<C>(entity)?;
    let write = || serde_json::to_value(component).ok();
    match world.get_resource::<SceneContext>() {
        Some(context) => context.scope(write),
        None => write(),
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

fn prepared_component<C: DeserializeOwned + Send + Sync + 'static>(
    value: &Value,
    path: &str,
) -> Result<Prepared, Error> {
    Ok(Prepared(Box::new(parsed::<C>(value, path)?)))
}

fn insert_component<C: Component>(
    world: &mut World,
    entity: Entity,
    prepared: Prepared,
) -> Result<(), Error> {
    let Ok(component) = prepared.0.downcast::<C>() else {
        return Err(Error::new(ErrorCode::InvalidType, "$").got(std::any::type_name::<C>()));
    };
    let mut target = world
        .get_entity_mut(entity)
        .map_err(|_| Error::new(ErrorCode::UnknownEntity, "$").got(entity.to_string()))?;
    target.insert(*component);
    Ok(())
}

fn read_form<F: SceneForm>(world: &mut World, entity: Entity, value: &Value) -> Result<(), Error> {
    let form: F = engine_settings::parse_value(value.clone())?;
    if world.get_entity(entity).is_err() {
        return Err(Error::new(ErrorCode::UnknownEntity, "$").got(entity.to_string()));
    }
    form.insert(world, entity)
}

fn form_value<F: SceneForm>(world: &World, entity: Entity) -> Option<Value> {
    let form = F::read(world, entity)?;
    let write = || serde_json::to_value(&form).ok();
    match world.get_resource::<SceneContext>() {
        Some(context) => context.scope(write),
        None => write(),
    }
}

fn insert_form<F: SceneForm>(
    world: &mut World,
    entity: Entity,
    prepared: Prepared,
) -> Result<(), Error> {
    let Ok(form) = prepared.0.downcast::<F>() else {
        return Err(Error::new(ErrorCode::InvalidType, "$").got(std::any::type_name::<F>()));
    };
    if world.get_entity(entity).is_err() {
        return Err(Error::new(ErrorCode::UnknownEntity, "$").got(entity.to_string()));
    }
    F::insert(*form, world, entity)
}

fn remove_component<C: Component>(world: &mut World, entity: Entity) {
    if let Ok(mut target) = world.get_entity_mut(entity) {
        target.remove::<C>();
    }
}

pub struct SceneComponents;

impl List for SceneComponents {
    const NAME: &'static str = "scene_components";
    type Member = ComponentKind;
}

#[doc(hidden)]
pub fn scene_components(world: &World) -> impl Iterator<Item = &ComponentKind> + '_ {
    crate::authoring::core_components().iter().chain(
        world
            .get_resource::<engine_registry::MembersOf<SceneComponents>>()
            .into_iter()
            .flat_map(|members| members.all().iter()),
    )
}

#[doc(hidden)]
pub fn scene_component(world: &World, key: &str) -> Option<ComponentKind> {
    scene_components(world)
        .find(|kind| kind.key == key)
        .cloned()
}

#[derive(SystemParam)]
pub struct SceneIds<'w, 's> {
    scene: Option<Res<'w, OperationScene>>,
    entities: Query<'w, 's, (Entity, &'static SceneEntity)>,
}

impl crate::EntityIds for SceneIds<'_, '_> {
    fn id(&self, entity: Entity) -> Option<String> {
        SceneIds::id(self, entity).map(str::to_owned)
    }
}

impl SceneIds<'_, '_> {
    pub fn loaded(&self) -> bool {
        self.scene.is_some()
    }

    pub fn len(&self) -> usize {
        self.scene.as_ref().map_or(0, |scene| scene.entity_count())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn entity(&self, id: &str) -> Option<Entity> {
        self.scene.as_ref()?.entity(id)
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

pub(crate) fn scene_entity_inserted(
    insert: On<Insert, SceneEntity>,
    authored: Query<&SceneEntity>,
    scene: Option<ResMut<OperationScene>>,
) {
    if let (Some(mut scene), Ok(authored)) = (scene, authored.get(insert.entity)) {
        scene.install(authored, insert.entity);
    }
}

pub(crate) fn scene_entity_discarded(
    discard: On<Discard, SceneEntity>,
    authored: Query<&SceneEntity>,
    scene: Option<ResMut<OperationScene>>,
) {
    if let (Some(mut scene), Ok(authored)) = (scene, authored.get(discard.entity)) {
        scene.uninstall(authored, discard.entity);
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

#[derive(Clone, Debug, PartialEq)]
pub struct SubSceneEntity {
    pub transform: crate::Transform,
    pub authored: Authored,
    pub parent: Option<usize>,
    pub components: Vec<(String, Value)>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SubScene {
    pub entities: Vec<SubSceneEntity>,
}

#[derive(Resource, Clone)]
#[component(immutable)]
pub struct SceneContext {
    pub id: String,
    pub assets: Arc<engine_assets::SceneAssets>,
    pub prefabs: Arc<BTreeMap<String, (usize, Value)>>,
    pub components: Arc<BTreeMap<String, BTreeMap<String, Value>>>,
    pub entities: Arc<BTreeMap<String, Entity>>,
    pub kinds: Arc<[ComponentKind]>,
    pub list: Option<Arc<JobList>>,
    pub files: Option<Arc<SceneBuffers>>,
}

impl SceneContext {
    pub fn scope<R>(&self, run: impl FnOnce() -> R) -> R {
        self.assets.scope(run)
    }

    pub fn entity(&self, id: &str) -> Option<Entity> {
        self.entities.get(id).copied()
    }

    pub fn prefab(&self, id: &str) -> Option<&(usize, Value)> {
        self.prefabs.get(id)
    }

    pub fn component(&self, entity: &str, key: &str) -> Option<&Value> {
        self.components.get(entity)?.get(key)
    }

    pub fn members<L: List>(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.list
            .as_deref()
            .into_iter()
            .flat_map(JobList::members::<L>)
    }
}

settings! {
    #[section("detail")]
    pub struct Detail {
        pub enabled: bool = true => any,
        pub pixel_error: f32 = 1.0 => range(0.25, 64.0),
        pub fade_seconds: f32 = 0.5 => range(0.0, 4.0),
        pub margin: f32 = 0.1 => range(0.0, 0.5),
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
#[component(immutable)]
pub struct DeviceLimits {
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

#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
#[component(immutable)]
pub struct PartDraws {
    draws: Vec<(TypeId, u32)>,
}

impl PartDraws {
    pub fn of(&self, part: TypeId) -> u32 {
        self.draws
            .iter()
            .find(|(of, _)| *of == part)
            .map_or(0, |&(_, draws)| draws)
    }

    #[doc(hidden)]
    pub fn add(&mut self, part: TypeId, draws: u32) {
        match self.draws.iter_mut().find(|(of, _)| *of == part) {
            Some((_, held)) => *held = held.saturating_add(draws),
            None => self.draws.push((part, draws)),
        }
    }
}
