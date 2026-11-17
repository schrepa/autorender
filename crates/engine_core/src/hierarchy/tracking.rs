use bevy_ecs::{
    change_detection::CheckChangeTicks, lifecycle::HookContext, prelude::*, world::DeferredWorld,
};
use engine_place::{Mover, Node};

#[derive(Resource)]
pub(crate) struct Invalidation {
    pub(crate) full_scan: bool,
    pub(crate) touched: Vec<Entity>,
    pub(crate) placed: Vec<Entity>,
}

impl Default for Invalidation {
    fn default() -> Self {
        Self {
            full_scan: true,
            touched: Vec::new(),
            placed: Vec::new(),
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

pub(crate) fn track_nodes(world: &mut World) {
    init(world);
    world
        .register_component_hooks::<Node>()
        .on_insert(node_touched)
        .on_remove(node_touched);
    world
        .register_component_hooks::<Mover>()
        .on_insert(node_touched)
        .on_remove(node_touched);
}

pub(crate) fn topology_changed(mut world: DeferredWorld, context: HookContext) {
    if let Some(mut invalidation) = world.get_resource_mut::<Invalidation>() {
        invalidation.placed.push(context.entity);
    }
}

pub(crate) fn node_touched(mut world: DeferredWorld, context: HookContext) {
    if let Some(mut invalidation) = world.get_resource_mut::<Invalidation>() {
        invalidation.touched.push(context.entity);
    }
}
