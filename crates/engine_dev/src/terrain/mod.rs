//! Terrain for agents: read-only queries ([`queries`]: heights at points,
//! a sweep comparing what terrain queries, collision rays and generated
//! terrains' layers say the ground is, routes for paths, walks and tracks)
//! and controls ([`controls`]: temporary edits of generated terrains, with
//! undo and redo, tracks pressed, agents sent and streamed items removed).

mod controls;
mod queries;

use engine_core::{
    Feature,
    bevy_ecs::prelude::{Resource, World},
    has_feature,
};
use engine_settings::ErrorCode;

use crate::{DevError, coordinate};

pub use controls::{Press, TerrainEditing, change_world, edit_terrain, nav_go, press_track};
pub(crate) use queries::execute;

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
fn feature<R: Resource>(world: &World, feature: Feature) -> Result<&R, DevError> {
    world
        .get_resource::<R>()
        .filter(|_| has_feature(world, feature))
        .ok_or_else(|| missing(feature))
}

/// [`feature`]'s resource, mutably.
fn feature_mut<
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
