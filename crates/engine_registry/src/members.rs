use std::any::TypeId;

use bevy_ecs::resource::Resource;
use bevy_ecs::world::{Mut, World};
use engine_place::{Error, ErrorCode, List};

use crate::{BrickId, BrickRecord};

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
    pub fn taken(records: &mut [BrickRecord]) -> Self {
        let mut taken = Self::default();
        let list = TypeId::of::<L>();
        for (index, record) in records.iter_mut().enumerate() {
            for joined in record.joins.iter_mut().filter(|joined| joined.list == list) {
                if let Some(Ok(member)) = joined
                    .member
                    .take()
                    .map(|member| member.downcast::<L::Member>())
                {
                    taken.members.push(*member);
                    taken.ids.push(BrickId(index as u32));
                    taken.owners.push(record.package);
                }
            }
        }
        taken
    }

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

pub(crate) fn fill_members<L: List>(world: &mut World, records: &mut [BrickRecord]) {
    if !world.contains_resource::<MembersOf<L>>() {
        world.insert_resource(MembersOf::<L>::taken(records));
    }
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
