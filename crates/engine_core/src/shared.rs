use bevy_ecs::world::World;
use engine_registry::CORE_ROW;

use crate::{
    Board, FrameBudget, GlobalTransforms, Pause, PauseReason, Readiness, Views, WorkClock,
    settings,
};

settings! {
    #[section("feeds")]
    pub struct FeedSettings {
        pub transforms: u32 = 16384 => at_least(1),
        pub mesh_residency: u32 = 4096 => at_least(1),
    }
}

settings! {
    #[section("frame_budget")]
    pub struct FrameBudgetSettings {
        pub work_ms: f32 = 4.0 => range(0.1, 1000.0),
        pub upload_bytes: u32 = 16 << 20 => at_least(1),
    }
}

pub(crate) fn start_frame(world: &mut World, spare_ms: f64) {
    let budget = world
        .get_resource::<FrameBudgetSettings>()
        .cloned()
        .unwrap_or_default();
    let clock = world
        .get_resource::<WorkClock>()
        .and_then(|clock| clock.reading);
    let work_ms = f64::from(budget.work_ms);
    let preparing = world
        .get_resource::<Pause>()
        .is_some_and(|pause| pause.holds(PauseReason::Preparing));
    let upload_bytes = if preparing {
        (f64::from(budget.upload_bytes) * (work_ms + spare_ms) / work_ms) as u64
    } else {
        u64::from(budget.upload_bytes)
    };
    let clock = clock.filter(|_| !preparing);
    world.insert_resource(FrameBudget::start(clock, (work_ms, spare_ms), upload_bytes));
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
