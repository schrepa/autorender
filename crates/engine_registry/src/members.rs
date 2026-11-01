use std::any::Any;

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_place::List;

use crate::gates::{GateId, Gates};

#[derive(Resource)]
pub struct MembersOf<L: List> {
    members: Vec<L::Member>,
    live: Vec<bool>,
    gates: Vec<GateId>,
}

impl<L: List> Default for MembersOf<L> {
    fn default() -> Self {
        Self {
            members: Vec::new(),
            live: Vec::new(),
            gates: Vec::new(),
        }
    }
}

impl<L: List> MembersOf<L> {
    pub fn live(&self) -> impl Iterator<Item = &L::Member> + '_ {
        self.members
            .iter()
            .zip(&self.live)
            .filter(|(_, live)| **live)
            .map(|(member, _)| member)
    }

    pub fn all(&self) -> &[L::Member] {
        &self.members
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn push(&mut self, member: L::Member, gate: GateId) {
        self.members.push(member);
        self.live.push(true);
        self.gates.push(gate);
    }

    pub fn refresh(&mut self, gates: &Gates) {
        for (live, gate) in self.live.iter_mut().zip(&self.gates) {
            *live = gates.on(*gate);
        }
    }
}

pub(crate) fn add_member<L: List>(world: &mut World, member: Box<dyn Any + Send + Sync>, gate: GateId) {
    if let Ok(member) = member.downcast::<L::Member>() {
        world.get_resource_or_init::<MembersOf<L>>().push(*member, gate);
    }
}

pub(crate) fn init_members<L: List>(world: &mut World) {
    world.init_resource::<MembersOf<L>>();
}

pub(crate) fn refresh_members<L: List>(world: &mut World) {
    world.try_resource_scope::<Gates, _>(|world, gates| {
        if let Some(mut members) = world.get_resource_mut::<MembersOf<L>>() {
            members.refresh(&gates);
        }
    });
}
