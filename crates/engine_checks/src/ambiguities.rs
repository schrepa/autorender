use engine::Engine;
use engine::bevy_ecs::schedule::{LogLevel, ScheduleBuildSettings, Schedules};

pub fn check(mut engine: Engine) -> Result<(), String> {
    engine.app.finish();
    engine.app.cleanup();
    let world = engine.app.world_mut();
    let labels: Vec<_> = world
        .resource::<Schedules>()
        .iter()
        .map(|(_, schedule)| schedule.label())
        .collect();
    let mut problems = Vec::new();
    for label in labels {
        let Some(mut schedule) = world.resource_mut::<Schedules>().remove(label) else {
            continue;
        };
        schedule.set_build_settings(ScheduleBuildSettings {
            ambiguity_detection: LogLevel::Error,
            ..ScheduleBuildSettings::default()
        });
        if let Err(error) = schedule.initialize(world) {
            problems.push(
                engine_core::Error::new(
                    engine_core::ErrorCode::ScheduleAmbiguity,
                    format!("{label:?}"),
                )
                .got(error.to_string(schedule.graph(), world)),
            );
        }
        world.resource_mut::<Schedules>().insert(schedule);
    }
    if problems.is_empty() {
        println!("{}", serde_json::json!({"event": "schedules_ordered"}));
        return Ok(());
    }
    crate::report(&problems)
}
