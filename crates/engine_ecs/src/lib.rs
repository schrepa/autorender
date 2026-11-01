pub mod prelude {
    pub use bevy_ecs::change_detection::{DetectChanges, DetectChangesMut, Mut, Ref, Res, ResMut};
    pub use bevy_ecs::component::Component;
    pub use bevy_ecs::entity::Entity;
    pub use bevy_ecs::query::{Added, Changed, Has, Or, With, Without};
    pub use bevy_ecs::resource::Resource;
    pub use bevy_ecs::system::{Local, Query, SystemParamItem};
}

pub use prelude::*;

#[doc(hidden)]
pub mod component {
    pub use bevy_ecs::component::{
        Component, ComponentCloneBehavior, ComponentId, DefaultCloneBehaviorBase,
        DefaultCloneBehaviorSpecialization, DefaultCloneBehaviorViaClone, Immutable, Mutable,
        RequiredComponentsRegistrator, StorageType,
    };
}

#[doc(hidden)]
pub mod resource {
    pub use bevy_ecs::resource::{IsResource, Resource};
}

#[doc(hidden)]
pub mod relationship {
    pub use bevy_ecs::relationship::ComponentRelationshipAccessor;
}
