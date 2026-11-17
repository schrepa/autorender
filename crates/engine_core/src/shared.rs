use bevy_ecs::world::World;
use engine_registry::CORE_ROW;

use crate::{Board, GlobalTransforms, Readiness, Views, settings};

settings! {
    #[section("feeds")]
    pub struct FeedSettings {
        pub transforms: u32 = 16384 => at_least(1),
        pub mesh_residency: u32 = 4096 => at_least(1),
    }
}

pub(crate) fn start_frame(world: &mut World) {
    let transforms = world
        .get_resource::<FeedSettings>()
        .map_or(FeedSettings::default().transforms, |feeds| feeds.transforms);
    if let Some(mut globals) = world.get_resource_mut::<GlobalTransforms>() {
        globals.resize_changes(transforms as usize);
    }
    if let Some(mut poses) = world.get_resource_mut::<crate::DrawnPoses>() {
        poses.resize_changes(transforms as usize);
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
    if let Some(reports) = world.get_resource::<crate::DrawReports>() {
        reports.add_to(&mut readiness);
    }
    world.insert_resource(readiness);
}
