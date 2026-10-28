use engine_collision::CollisionPlugin;
use engine_core::{
    Feature, NavigationPlugin, PluginGroup, PluginGroupBuilder, TerrainPlugin, TimeOfDayPlugin,
    WorldStreamingPlugin,
};
use engine_physics::PhysicsPlugin;

pub struct EnginePlugins;

macro_rules! engine_plugins {
    ($($(#[$doc:meta])* $name:ident: $variant:ident => $plugin:ident, needs [$($need:ident),*];)*) => {
        impl EnginePlugins {
            pub fn without(features: impl IntoIterator<Item = Feature>) -> PluginGroupBuilder {
                features
                    .into_iter()
                    .fold(Self.build(), |plugins, feature| match feature {
                        $(Feature::$variant => plugins.disable::<$plugin>(),)*
                    })
            }
        }

        impl PluginGroup for EnginePlugins {
            fn build(self) -> PluginGroupBuilder {
                PluginGroupBuilder::start::<Self>()$(.add($plugin))*
            }
        }
    };
}

engine_core::engine_features!(engine_plugins);
