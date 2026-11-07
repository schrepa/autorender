use std::any::{Any, TypeId};

use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_place::{Error, ErrorCode, List};

use crate::state::{BrickStates, GateId};

#[derive(Resource)]
pub struct MembersOf<L: List> {
    members: Vec<L::Member>,
    gates: Vec<GateId>,
    owners: Vec<&'static str>,
    bricks: Vec<TypeId>,
    locals: Vec<u32>,
}

impl<L: List> Default for MembersOf<L> {
    fn default() -> Self {
        Self {
            members: Vec::new(),
            gates: Vec::new(),
            owners: Vec::new(),
            bricks: Vec::new(),
            locals: Vec::new(),
        }
    }
}

impl<L: List> MembersOf<L> {
    pub fn live<'a>(&'a self, states: &'a BrickStates) -> impl Iterator<Item = &'a L::Member> + 'a {
        self.members
            .iter()
            .zip(&self.gates)
            .filter(|(_, gate)| states.on(**gate))
            .map(|(member, _)| member)
    }

    pub fn owned<'a>(
        &'a self,
        states: &'a BrickStates,
    ) -> impl Iterator<Item = (TypeId, u32, &'a L::Member)> + 'a {
        self.members
            .iter()
            .zip(&self.gates)
            .zip(self.bricks.iter().zip(&self.locals))
            .filter(|((_, gate), _)| states.on(**gate))
            .map(|((member, _), (&brick, &local))| (brick, local, member))
    }

    pub fn all(&self) -> &[L::Member] {
        &self.members
    }

    pub fn every(&self) -> impl Iterator<Item = (TypeId, u32, &L::Member)> + '_ {
        self.members
            .iter()
            .zip(self.bricks.iter().zip(&self.locals))
            .map(|(member, (&brick, &local))| (brick, local, member))
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

    pub fn push(&mut self, member: L::Member, gate: GateId, owner: &'static str, brick: TypeId) {
        let local = self.bricks.iter().filter(|&&known| known == brick).count() as u32;
        self.members.push(member);
        self.gates.push(gate);
        self.owners.push(owner);
        self.bricks.push(brick);
        self.locals.push(local);
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
    (owner, brick): (&'static str, TypeId),
) {
    if let Ok(member) = member.downcast::<L::Member>() {
        world
            .get_resource_or_init::<MembersOf<L>>()
            .push(*member, gate, owner, brick);
    }
}

pub(crate) fn each_member<L: List>(world: &World, each: &mut dyn FnMut(&dyn Any)) {
    if let (Some(members), Some(states)) = (
        world.get_resource::<MembersOf<L>>(),
        world.get_resource::<BrickStates>(),
    ) {
        for member in members.live(states) {
            each(member);
        }
    }
}

pub(crate) fn init_members<L: List>(world: &mut World) {
    world.init_resource::<MembersOf<L>>();
}
