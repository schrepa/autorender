use bevy_ecs::world::World;
use engine_registry::Registry;

use crate::{
    Error, ErrorCode, FrameBudget, GlobalTransforms, Level, MeshLevels, Readiness,
    RecordExchange, Records, WorkClock, settings,
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
    world.insert_resource(FrameBudget::start(
        clock,
        (f64::from(budget.work_ms), spare_ms),
        u64::from(budget.upload_bytes),
    ));
    let transforms = world
        .get_resource::<FeedSettings>()
        .map_or(FeedSettings::default().transforms, |feeds| feeds.transforms);
    if let Some(mut globals) = world.get_resource_mut::<GlobalTransforms>() {
        globals.resize_changes(transforms as usize);
    }
    if let Some(mut poses) = world.get_resource_mut::<crate::DrawnPoses>() {
        poses.resize_changes(transforms as usize);
    }
    let now = world.resource::<crate::FrameTime>().real_elapsed;
    if let Some(mut level) = world.get_resource_mut::<Level>() {
        level.settle(now);
    }
}

pub(crate) fn settle_readiness(world: &mut World) {
    let compiling = world
        .get_resource::<Readiness>()
        .map_or(0, |readiness| readiness.compiling);
    let mut readiness = Readiness {
        compiling,
        ..Readiness::default()
    };
    if let Some(registry) = world.get_resource::<Registry>() {
        readiness.settle(registry.pending(world));
    }
    if let Some(records) = world.get_resource::<Records>() {
        readiness.load(&records.census());
        for (key, error) in records.failures() {
            readiness.fail(&format!("records.{}", key.hex()), error.clone());
        }
    }
    if let Some(exchange) = world.get_resource::<RecordExchange>() {
        readiness.load(&exchange.census());
        for (key, error) in exchange.failures() {
            readiness.fail(&format!("textures.{}", key.hex()), error.clone());
        }
    }
    if let Some(levels) = world.get_resource::<MeshLevels>() {
        readiness.load(&levels.census());
        for (key, request) in levels.unsettled() {
            if let crate::Request::Failed(error) = request {
                readiness.fail(&format!("mesh_lods.{}", key.hex()), error.clone());
            }
        }
    }
    if let Some(cooking) = world.get_resource::<crate::cook::Cooking>() {
        readiness.settle(u32::try_from(cooking.left()).unwrap_or(u32::MAX));
    }
    if let Some(level) = world.get_resource::<Level>() {
        readiness.load(&level.census());
        if let Some((path, error)) = level.failure() {
            readiness.fail(&format!("level.{path}"), error.clone());
        }
    }
    world.insert_resource(readiness);
}
