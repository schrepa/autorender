//! The engine's features as one plugin group.

use engine_collision::CollisionPlugin;
use engine_core::{
    AnimationPlugin, DeformationPlugin, Feature, NavigationPlugin, OceanPlugin, PluginGroup,
    PluginGroupBuilder, TerrainPlugin, TimeOfDayPlugin, WorldStreamingPlugin,
};

/// Every feature of the engine as one plugin group: terrain, world
/// streaming, navigation, the ocean, animation, the time of day,
/// deformation and collision. Add it to an engine for all of them, or
/// leave some out: `EnginePlugins.build().disable::<OceanPlugin>()`.
/// Scenes and development requests naming a feature left out fail with
/// `missing_plugin`; navigation needs terrain.
pub struct EnginePlugins;

impl EnginePlugins {
    /// Every feature's plugin but those of `features`: what hosts and tools
    /// choosing features by name build engines with.
    pub fn without(features: impl IntoIterator<Item = Feature>) -> PluginGroupBuilder {
        features
            .into_iter()
            .fold(Self.build(), |plugins, feature| match feature {
                Feature::Terrain => plugins.disable::<TerrainPlugin>(),
                Feature::WorldStreaming => plugins.disable::<WorldStreamingPlugin>(),
                Feature::Navigation => plugins.disable::<NavigationPlugin>(),
                Feature::Ocean => plugins.disable::<OceanPlugin>(),
                Feature::Animation => plugins.disable::<AnimationPlugin>(),
                Feature::TimeOfDay => plugins.disable::<TimeOfDayPlugin>(),
                Feature::Deformation => plugins.disable::<DeformationPlugin>(),
                Feature::Collision => plugins.disable::<CollisionPlugin>(),
            })
    }
}

impl PluginGroup for EnginePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(TerrainPlugin)
            .add(WorldStreamingPlugin)
            .add(NavigationPlugin)
            .add(OceanPlugin)
            .add(AnimationPlugin)
            .add(TimeOfDayPlugin)
            .add(DeformationPlugin)
            .add(CollisionPlugin)
    }
}
