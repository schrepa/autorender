use std::any::Any;

use bevy_ecs::resource::Resource;
use bevy_ecs::world::{Mut, World};
use engine_place::{Error, ErrorCode, List};

use crate::BrickId;

#[derive(Resource)]
pub struct MembersOf<L: List> {
    members: Vec<L::Member>,
    ids: Vec<BrickId>,
    owners: Vec<&'static str>,
}

impl<L: List> Default for MembersOf<L> {
    fn default() -> Self {
        Self {
            members: Vec::new(),
            ids: Vec::new(),
            owners: Vec::new(),
        }
    }
}

impl<L: List> MembersOf<L> {
    pub fn all(&self) -> &[L::Member] {
        &self.members
    }

    pub fn owners(&self) -> &[&'static str] {
        &self.owners
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
    id: BrickId,
    owner: &'static str,
) {
    if let Ok(member) = member.downcast::<L::Member>() {
        let mut members = world.get_resource_or_init::<MembersOf<L>>();
        members.members.push(*member);
        members.ids.push(id);
        members.owners.push(owner);
    }
}

pub(crate) fn init_members<L: List>(world: &mut World) {
    world.init_resource::<MembersOf<L>>();
}

pub(crate) fn drop_members<L: List>(world: &mut World, brick: BrickId) {
    world.try_resource_scope(|world, members: Mut<MembersOf<L>>| {
        for (member, _) in members
            .members
            .iter()
            .zip(&members.ids)
            .filter(|(_, held)| **held == brick)
        {
            L::dropped(world, member);
        }
    });
}
