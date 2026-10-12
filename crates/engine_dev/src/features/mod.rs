//! The engine's features' development tools: each feature with tools of its
//! own has a file here that registers its inspectors, states their schemas
//! and answers its queries.

use engine_core::bevy_ecs::prelude::{Resource, World};
use engine_core::{Engine, Feature, has_feature};
use engine_scene::SceneInstance;
use engine_settings::ErrorCode;
use serde_json::Value;

use crate::{DevError, DevTools};

/// Every feature with a development file here, sorted: the one list of
/// them. Each entry is the feature's name as `engine_features!` names it,
/// which names its file. Code over these files passes a macro that receives
/// this list (`dev_features!(callback)`) instead of writing them out: their
/// modules, and registering their inspectors and schemas.
macro_rules! dev_features {
    ($then:ident) => {
        $then! {
            animation,
            collision,
            deformation,
            navigation,
            ocean,
            terrain,
            world_streaming,
        }
    };
}

/// A development query a feature's file answers: a
/// [`DevRequest`](crate::DevRequest) variant holds it, and the routing of
/// queries hands it to [`EngineQuery::answer`].
pub trait EngineQuery {
    /// The query's answer from `engine`'s world, beside the active `scene`.
    fn answer(&self, engine: &mut Engine, scene: Option<&SceneInstance>)
    -> Result<Value, DevError>;
}

/// The JSON schema of `T`, an inspector's view.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn schema<T: schemars::JsonSchema>() -> Value {
    serde_json::json!(schemars::schema_for!(T))
}

/// A development control a feature's file applies: a
/// [`BrowserControl`](crate::BrowserControl) variant holds it, and the host
/// applies it to its engine, then renders.
pub trait EngineControl {
    /// What it changes, which a render failing after it names.
    fn changed(&self) -> &'static str;

    /// Applies it to `engine`'s world, beside the active `scene`: its result.
    fn apply(&self, engine: &mut Engine, scene: Option<&SceneInstance>) -> Result<Value, DevError>;
}

/// The `missing_plugin` error of requests about `feature`, which the engine
/// lacks.
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

/// `world`'s resource `R` of `feature`, or [`missing`].
pub(crate) fn feature<R: Resource>(world: &World, feature: Feature) -> Result<&R, DevError> {
    world
        .get_resource::<R>()
        .filter(|_| has_feature(world, feature))
        .ok_or_else(|| missing(feature))
}

/// [`feature`]'s resource, mutably.
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
    /// Runs `register`, the development file of feature `name` registering
    /// its inspectors, which become the feature's: inspecting them without
    /// its plugin answers `missing_plugin`.
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

/// This module's part of `dev_features!`: the files, their names, and
/// registering their inspectors and schemas.
macro_rules! files {
    ($($name:ident,)*) => {
        $(pub mod $name;)*

        /// The features with a development file, by name, as listed.
        #[cfg(not(target_arch = "wasm32"))]
        pub(crate) const NAMES: &[&str] = &[$(stringify!($name)),*];

        /// Registers each feature's inspectors, as its file does, as the
        /// feature's own.
        pub(crate) fn register(tools: &mut DevTools, schemas: &Value) {
            $(tools.register_feature(stringify!($name), |tools| $name::register(tools, schemas));)*
        }

        /// Adds the schemas of each feature's inspectors, by inspector.
        #[cfg(not(target_arch = "wasm32"))]
        pub(crate) fn schemas(schemas: &mut serde_json::Map<String, Value>) {
            $($name::schemas(schemas);)*
        }
    };
}

dev_features!(files);
