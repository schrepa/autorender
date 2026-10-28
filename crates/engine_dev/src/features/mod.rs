use engine_core::bevy_ecs::prelude::{Resource, World};
use engine_core::{Engine, Feature, has_feature};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
use serde_json::Value;

use crate::{DevError, DevTools};

macro_rules! dev_features {
    ($then:ident) => {
        $then! {
            collision,
            deformation,
            navigation,
            ocean,
            physics,
            terrain,
            world_streaming,
        }
    };
}

pub trait EngineQuery {
    fn answer(&self, engine: &mut Engine, scene: Option<&SceneInstance>)
    -> Result<Value, DevError>;
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schema<T: schemars::JsonSchema>() -> Value {
    serde_json::json!(schemars::schema_for!(T))
}

pub trait EngineControl {
    fn changed(&self) -> &'static str;

    fn apply(&self, engine: &mut Engine, scene: Option<&SceneInstance>) -> Result<Value, DevError>;
}

pub(crate) fn missing(feature: Feature) -> DevError {
    DevError::new(
        ErrorCode::MissingPlugin,
        "$.op",
        format!(
            "{} is not added: add it, or EnginePlugins, to the engine",
            feature.plugin()
        ),
    )
}

pub(crate) fn feature<R: Resource>(world: &World, feature: Feature) -> Result<&R, DevError> {
    world
        .get_resource::<R>()
        .filter(|_| has_feature(world, feature))
        .ok_or_else(|| missing(feature))
}

pub(crate) fn feature_mut<
    R: Resource
        + engine_core::bevy_ecs::component::Component<
            Mutability = engine_core::bevy_ecs::component::Mutable,
        >,
>(
    world: &mut World,
    feature: Feature,
) -> Result<engine_core::bevy_ecs::prelude::Mut<'_, R>, DevError> {
    if !has_feature(world, feature) {
        return Err(missing(feature));
    }
    world
        .get_resource_mut::<R>()
        .ok_or_else(|| missing(feature))
}

impl DevTools {
    fn register_feature(&mut self, name: &str, register: impl FnOnce(&mut Self)) {
        let feature = Feature::ALL
            .into_iter()
            .find(|feature| feature.name() == name)
            .expect("dev_features! names features (the schema generator checks)");
        let before: Vec<String> = self.inspectors.keys().cloned().collect();
        register(self);
        for inspector in self.inspectors.keys() {
            if !before.contains(inspector) {
                self.features.insert(inspector.clone(), feature);
            }
        }
    }
}

macro_rules! files {
    ($($name:ident,)*) => {
        #[cfg(not(target_arch = "wasm32"))]
        pub(crate) const NAMES: &[&str] = &[$(stringify!($name)),*];

        pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
            $(tools.register_feature(stringify!($name), |tools| $name::register(tools, schemas));)*
        }

        #[cfg(not(target_arch = "wasm32"))]
        pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
            $($name::schemas(schemas);)*
        }
    };
}

pub mod collision;
pub mod deformation;
pub mod navigation;
pub mod ocean;
pub mod physics;
pub mod terrain;
pub mod world_streaming;

dev_features!(files);
