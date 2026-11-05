use bevy_ecs::entity_disabling::Disabled;
use bevy_ecs::prelude::*;

use super::{GlobalTransforms, tracking};

#[derive(Component, Debug, Clone, Copy, Default)]
#[component(
    on_add = crate::hierarchy::tracking::hidden_switched,
    on_remove = crate::hierarchy::tracking::hidden_switched)]
pub struct Hidden;

#[derive(Component, Debug, Clone, Copy, Default)]
#[component(
    on_add = crate::hierarchy::tracking::hidden_switched,
    on_remove = crate::hierarchy::tracking::hidden_switched)]
pub struct HiddenByParent;

#[derive(Component, Debug, Clone, Copy, Default)]
#[component(
    on_add = crate::hierarchy::tracking::disabled_switched,
    on_remove = crate::hierarchy::tracking::disabled_switched)]
pub struct DisabledByParent;

pub type Shown = (Without<Hidden>, Without<HiddenByParent>);
pub type EvenDisabled = (Allow<Disabled>, Allow<DisabledByParent>);
pub(crate) type SeesDisabled = (Has<Disabled>, Has<DisabledByParent>);

pub(super) type SwitchQuery = QueryState<
    (
        Entity,
        Has<Hidden>,
        Has<HiddenByParent>,
        Has<Disabled>,
        Has<DisabledByParent>,
    ),
    Or<(
        With<Hidden>,
        With<HiddenByParent>,
        With<Disabled>,
        With<DisabledByParent>,
    )>,
>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Switches {
    pub hidden: u64,
    pub disabled: u64,
}

#[derive(Clone, Copy, Default)]
struct NodeSwitch {
    hidden: bool,
    hidden_above: bool,
    disabled: bool,
    disabled_above: bool,
}

fn set_marker<M: Component + Default>(world: &mut World, entity: Entity, on: bool) {
    if let Ok(mut node) = world.get_entity_mut(entity) {
        if on {
            node.insert(M::default());
        } else {
            node.remove::<M>();
        }
    }
}

impl GlobalTransforms {
    pub fn switches(&self) -> Switches {
        self.switches
    }

    pub(super) fn settle_switches(&mut self, world: &mut World, topology_changed: bool) {
        let held = world.resource::<tracking::Invalidation>();
        if !(topology_changed || held.hidden || held.disabled) {
            return;
        }
        self.reach_children(world);
        let mut held = world.resource_mut::<tracking::Invalidation>();
        self.switches.hidden += u64::from(std::mem::take(&mut held.hidden));
        self.switches.disabled += u64::from(std::mem::take(&mut held.disabled));
    }

    fn reach_children(&mut self, world: &mut World) {
        let _profile = engine_profile::span("hierarchy", "reach_children");
        let mut nodes = vec![NodeSwitch::default(); self.links.len()];
        let mut any = false;
        for (entity, hidden, hidden_above, disabled, disabled_above) in
            self.switch_query.iter(world)
        {
            if let Some(&index) = self.indices.get(&entity) {
                nodes[index] = NodeSwitch {
                    hidden,
                    hidden_above,
                    disabled,
                    disabled_above,
                };
                any = true;
            }
        }
        if !any {
            return;
        }
        for &index in &self.order {
            let above = self.links[index]
                .parent()
                .map_or(NodeSwitch::default(), |parent| nodes[parent]);
            let node = &mut nodes[index];
            if node.hidden_above != above.hidden {
                set_marker::<HiddenByParent>(world, self.entities[index], above.hidden);
            }
            if node.disabled_above != above.disabled {
                set_marker::<DisabledByParent>(world, self.entities[index], above.disabled);
            }
            node.hidden |= above.hidden;
            node.disabled |= above.disabled;
        }
    }
}
