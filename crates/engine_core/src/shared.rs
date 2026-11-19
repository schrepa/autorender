use bevy_ecs::world::World;

use crate::{Board, Readiness, settings};

settings! {
    #[section("feeds")]
    pub struct FeedSettings {
        pub mesh_residency: u32 = 4096 => at_least(1),
    }
}

pub(crate) fn settle_readiness(world: &mut World, board: &Board) {
    let compiling = world
        .get_resource::<Readiness>()
        .map_or(0, |readiness| readiness.compiling);
    let mut readiness = Readiness {
        compiling,
        ..Readiness::default()
    };
    board.add_to(&mut readiness);
    if let Some(refusals) = world.get_resource::<crate::Refusals>() {
        for (_, refused) in refusals.iter() {
            readiness.fail(refused.part, refused.reason.clone());
        }
    }
    if world.get_resource::<Readiness>() != Some(&readiness) {
        world.insert_resource(readiness);
    }
}
