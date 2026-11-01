use std::any::Any;

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_place::{Error, ErrorCode, List};

use crate::gates::{GateId, Gates};

#[derive(Resource)]
pub struct MembersOf<L: List> {
    members: Vec<L::Member>,
    live: Vec<bool>,
    gates: Vec<GateId>,
    owners: Vec<&'static str>,
}

impl<L: List> Default for MembersOf<L> {
    fn default() -> Self {
        Self {
            members: Vec::new(),
            live: Vec::new(),
            gates: Vec::new(),
            owners: Vec::new(),
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

    pub fn owners(&self) -> &[&'static str] {
        &self.owners
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn push(&mut self, member: L::Member, gate: GateId, owner: &'static str) {
        self.members.push(member);
        self.live.push(true);
        self.gates.push(gate);
        self.owners.push(owner);
    }

    pub fn refresh(&mut self, gates: &Gates) {
        for (live, gate) in self.live.iter_mut().zip(&self.gates) {
            *live = gates.on(*gate);
        }
    }

    pub fn names_twice(&self, name: fn(&L::Member) -> &str) -> Vec<Error> {
        let mut problems = Vec::new();
        for (index, member) in self.members.iter().enumerate() {
            let named = name(member);
            let first = self.members[..index]
                .iter()
                .position(|known| name(known) == named);
            if let Some(first) = first {
                let path = format!("$.places.{}.{named}", L::NAME);
                problems.push(
                    Error::new(ErrorCode::NameTwice, path)
                        .got(self.owners[first])
                        .nearest(self.owners[index]),
                );
            }
        }
        problems
    }
}

pub(crate) fn add_member<L: List>(
    world: &mut World,
    member: Box<dyn Any + Send + Sync>,
    gate: GateId,
    owner: &'static str,
) {
    if let Ok(member) = member.downcast::<L::Member>() {
        world
            .get_resource_or_init::<MembersOf<L>>()
            .push(*member, gate, owner);
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
