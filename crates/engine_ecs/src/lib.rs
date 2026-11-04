pub mod prelude {
    pub use bevy_ecs::change_detection::{DetectChanges, DetectChangesMut, Mut, Ref, Res, ResMut};
    pub use bevy_ecs::component::Component;
    pub use bevy_ecs::entity::Entity;
    pub use bevy_ecs::query::{Added, Changed, Has, Or, With, Without};
    pub use bevy_ecs::resource::Resource;
    pub use bevy_ecs::system::{In, Local, Query, SystemParam, SystemParamItem};
}

pub use prelude::*;

pub mod system {
    pub use bevy_ecs::system::{In, Local, Query, SystemChangeTick, SystemParam, SystemParamItem};
    #[doc(hidden)]
    pub use bevy_ecs::system::{ReadOnlySystemParam, SystemMeta, SystemParamValidationError};
}

#[doc(hidden)]
pub mod world {
    pub use bevy_ecs::world::{DeferredWorld, World};

    pub mod unsafe_world_cell {
        pub use bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell;
    }
}

#[doc(hidden)]
pub mod query {
    pub use bevy_ecs::query::FilteredAccessSet;
}

#[doc(hidden)]
pub mod change_detection {
    pub use bevy_ecs::change_detection::Tick;
}

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
