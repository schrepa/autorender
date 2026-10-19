//! The engine's features as one plugin group, from the one list of them
//! (`engine_features!`).

use engine_collision::CollisionPlugin;
use engine_core::{
    AnimationPlugin, DeformationPlugin, Feature, NavigationPlugin, OceanPlugin, PluginGroup,
    PluginGroupBuilder, TerrainPlugin, TimeOfDayPlugin, WorldStreamingPlugin,
};
use engine_physics::PhysicsPlugin;

/// Every feature of the engine ([`Feature::ALL`]) as one plugin group, in
/// that order. Add it to an engine for all of them, or leave some out:
/// `EnginePlugins.build().disable::<OceanPlugin>()`. Scenes and development
/// requests naming a feature left out fail with `missing_plugin`, and an
/// engine with a feature but not one it needs ([`Feature::needs`]) does not
/// start.
pub struct EnginePlugins;

/// The plugin group: every feature's plugin in the list's order, and
/// `without` leaving out those named.
macro_rules! engine_plugins {
    ($($(#[$doc:meta])* $name:ident: $variant:ident => $plugin:ident, needs [$($need:ident),*];)*) => {
        impl EnginePlugins {
            /// Every feature's plugin but those of `features`: what hosts
            /// and tools choosing features by name build engines with.
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
