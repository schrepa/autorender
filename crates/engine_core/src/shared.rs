use bevy_ecs::world::World;
use engine_registry::CORE_ROW;

use crate::{Board, Readiness, Views, settings};

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
    let viewed = world
        .get_resource::<Views>()
        .is_some_and(|views| !views.listed().is_empty());
    let mut readiness = Readiness {
        compiling,
        ..Readiness::default()
    };
    board.add_to(&mut readiness, viewed);
    if let Some(cooking) = world.get_resource::<crate::cook::Cooking>() {
        readiness.settle(CORE_ROW, u32::try_from(cooking.left()).unwrap_or(u32::MAX));
    }
    if let Some(refusals) = world.get_resource::<crate::Refusals>() {
        for (_, refused) in refusals.iter() {
            readiness.fail(refused.part, refused.reason.clone());
        }
    }
    if world.get_resource::<Readiness>() != Some(&readiness) {
        world.insert_resource(readiness);
    }
}
