use std::any::Any;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use bevy_ecs::component::{Component, ComponentId};
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::{Discard, Insert};
use bevy_ecs::observer::On;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{Query, ResMut};
use bevy_ecs::world::World;
use engine_place::{OperationScene, SceneEntity};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::{Assets, Error, ErrorCode, List};

pub struct SceneCx<'a> {
    scene: Option<&'a SceneContext>,
}

impl<'a> SceneCx<'a> {
    #[doc(hidden)]
    pub fn new(scene: Option<&'a SceneContext>) -> Self {
        Self { scene }
    }
}

impl SceneCx<'_> {
    pub fn scene(&self) -> Option<&SceneContext> {
        self.scene
    }
}

#[doc(hidden)]
pub struct Prepared(Box<dyn Any + Send + Sync>);

type Held = dyn Any + Send + Sync;
type CheckHook = Arc<dyn Fn(&Held, &str) -> Result<(), Error> + Send + Sync>;
type PrepareHook = Arc<dyn Fn(&Value, &SceneCx<'_>, &str) -> Result<Prepared, Error> + Send + Sync>;

#[derive(Clone)]
pub struct ComponentKind {
    key: &'static str,
    register: fn(&mut World) -> ComponentId,
    value: fn(&World, Entity) -> Option<Value>,
    prepared: fn(&Value, &str) -> Result<Prepared, Error>,
    insert: fn(&mut World, Entity, Prepared) -> Result<(), Error>,
    check: Option<CheckHook>,
    #[cfg(feature = "dev")]
    schema: Option<fn() -> Value>,
    prepare: Option<PrepareHook>,
}

#[must_use]
pub struct ComponentKindOf<C> {
    kind: ComponentKind,
    component: PhantomData<fn() -> C>,
}

pub trait SceneForm: Serialize + DeserializeOwned + Send + Sync + 'static {
    type Component: Component;

    fn insert(self, world: &mut World, entity: Entity) -> Result<(), Error>;

    fn read(world: &World, entity: Entity) -> Option<Self>;
}

impl ComponentKind {
    pub fn form<F: SceneForm>(key: &'static str) -> ComponentKindOf<F> {
        ComponentKindOf {
            kind: Self {
                key,
                register: World::register_component::<F::Component>,
                value: form_value::<F>,
                prepared: prepared_component::<F>,
                insert: insert_form::<F>,
                check: None,
                #[cfg(feature = "dev")]
                schema: None,
                prepare: None,
            },
            component: PhantomData,
        }
    }

    pub fn of<C: Component + Serialize + DeserializeOwned>(
        key: &'static str,
    ) -> ComponentKindOf<C> {
        ComponentKindOf {
            kind: Self {
                key,
                register: World::register_component::<C>,
                value: component_value::<C>,
                prepared: prepared_component::<C>,
                insert: insert_component::<C>,
                check: None,
                #[cfg(feature = "dev")]
                schema: None,
                prepare: None,
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
    pub fn read(&self, world: &mut World, entity: Entity, value: &Value) -> Result<(), Error> {
        let cx = SceneCx::new(world.get_resource::<SceneContext>());
        let prepared = self.prepare_value(value, &cx, "$")?;
        (self.insert)(world, entity, prepared)
    }

    #[doc(hidden)]
    pub fn prepare_value(
        &self,
        value: &Value,
        cx: &SceneCx<'_>,
        path: &str,
    ) -> Result<Prepared, Error> {
        let prepared = match &self.prepare {
            Some(prepare) => prepare(value, cx, path)?,
            None => (self.prepared)(value, path)?,
        };
        if let Some(check) = &self.check {
            check(prepared.0.as_ref(), path)?;
        }
        Ok(prepared)
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
    pub fn value(&self, world: &World, entity: Entity) -> Option<Value> {
        (self.value)(world, entity)
    }

    #[doc(hidden)]
    #[cfg(feature = "dev")]
    pub fn json_schema(&self) -> Value {
        self.schema.map_or(Value::Null, |schema| schema())
    }
}

impl<C: Serialize + DeserializeOwned + Send + Sync + 'static> ComponentKindOf<C> {
    pub fn check(mut self, check: fn(&C, &str) -> Result<(), Error>) -> Self {
        self.kind.check = Some(Arc::new(move |held: &Held, path: &str| {
            held.downcast_ref::<C>()
                .map_or(Ok(()), |component| check(component, path))
        }));
        self
    }

    #[cfg(feature = "dev")]
    pub fn schema(mut self, schema: fn() -> Value) -> Self {
        self.kind.schema = Some(schema);
        self
    }

    #[cfg(not(feature = "dev"))]
    pub fn schema(self, _: fn() -> Value) -> Self {
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

impl<C> From<ComponentKindOf<C>> for ComponentKind {
    fn from(of: ComponentKindOf<C>) -> Self {
        of.kind
    }
}

fn parsed<C: DeserializeOwned>(value: &Value, path: &str) -> Result<C, Error> {
    engine_settings::parse_value_at(value, path)
}

fn component_value<C: Component + Serialize>(world: &World, entity: Entity) -> Option<Value> {
    let component = world.get::<C>(entity)?;
    let write = || serde_json::to_value(component).ok();
    match world.get_resource::<Assets>() {
        Some(assets) => assets.scope(write),
        None => write(),
    }
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

fn form_value<F: SceneForm>(world: &World, entity: Entity) -> Option<Value> {
    let form = F::read(world, entity)?;
    let write = || serde_json::to_value(&form).ok();
    match world.get_resource::<Assets>() {
        Some(assets) => assets.scope(write),
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

#[derive(Resource, Clone)]
#[component(immutable)]
pub struct SceneContext {
    pub entities: Arc<BTreeMap<String, Entity>>,
}

impl SceneContext {
    pub fn entity(&self, id: &str) -> Option<Entity> {
        self.entities.get(id).copied()
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
#[component(immutable)]
pub struct DeviceLimits {
    pub storage_room: u64,
}
