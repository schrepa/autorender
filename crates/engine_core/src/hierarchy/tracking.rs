//! Structural invalidation complements the hierarchy's private change-tick
//! watermark. Ordinary mutable writes are detected from component metadata;
//! deliberately bypassing change detection or rewriting ticks opts out.
use bevy_ecs::{
    change_detection::CheckChangeTicks, lifecycle::HookContext, prelude::*, world::DeferredWorld,
};

#[derive(Resource)]
pub(super) struct Invalidation {
    pub(super) full_scan: bool,
}

impl Default for Invalidation {
    fn default() -> Self {
        Self { full_scan: true }
    }
}

/// Install once for the lifetime of the engine-owned resource. Tick maintenance
/// may clamp component timestamps, so the next boundary must adopt authoritative
/// values rather than mistake an aged, unobserved edit for an unchanged value.
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

pub(crate) fn topology_changed(mut world: DeferredWorld, _: HookContext) {
    if let Some(mut invalidation) = world.get_resource_mut::<Invalidation>() {
        invalidation.full_scan = true;
    }
}
