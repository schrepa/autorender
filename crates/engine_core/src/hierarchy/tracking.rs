use bevy_ecs::{
    change_detection::CheckChangeTicks, entity_disabling::Disabled, lifecycle::HookContext,
    prelude::*, world::DeferredWorld,
};

use super::DisabledByParent;

#[derive(Resource)]
pub(super) struct Invalidation {
    pub(super) full_scan: bool,
    pub(super) hidden: bool,
    pub(super) disabled: bool,
}

impl Default for Invalidation {
    fn default() -> Self {
        Self {
            full_scan: true,
            hidden: false,
            disabled: false,
        }
    }
}

pub(super) fn init(world: &mut World) {
    if world.contains_resource::<Invalidation>() {
        return;
    }
    world.init_resource::<Invalidation>();
    world.add_observer(
        |_: On<CheckChangeTicks>, mut invalidation: ResMut<Invalidation>| {
            invalidation.full_scan = true;
        },
    );
}

pub(crate) fn track_switches(world: &mut World) {
    world.register_disabling_component::<DisabledByParent>();
    world
        .register_component_hooks::<Disabled>()
        .on_add(disabled_switched)
        .on_remove(disabled_switched);
}

pub(crate) fn topology_changed(mut world: DeferredWorld, _: HookContext) {
    if let Some(mut invalidation) = world.get_resource_mut::<Invalidation>() {
        invalidation.full_scan = true;
    }
}

pub(crate) fn hidden_switched(mut world: DeferredWorld, _: HookContext) {
    if let Some(mut invalidation) = world.get_resource_mut::<Invalidation>() {
        invalidation.hidden = true;
    }
}

pub(crate) fn disabled_switched(mut world: DeferredWorld, _: HookContext) {
    if let Some(mut invalidation) = world.get_resource_mut::<Invalidation>() {
        invalidation.disabled = true;
    }
}
