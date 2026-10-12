//! The engine's schedules order every pair of systems that conflict: an
//! engine with every feature is built the way hosts build one, and each
//! schedule it runs is checked for systems that touch the same data with
//! no order between them, which would run in whichever order a later
//! change of the schedule happened to give them.

use std::path::Path;

use engine::bevy_ecs::schedule::{LogLevel, ScheduleBuildSettings, Schedules};

/// Fails, naming each pair of systems of the engine's schedules that
/// conflict without an order, and what they both touch.
pub fn check(_root: &Path) -> Result<(), String> {
    let mut engine = engine::Engine::new(engine::EngineConfig::default());
    engine.app.add_plugins(engine::EnginePlugins);
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
        // Conflicts fail the build, which names them while the schedule's
        // graph still holds its systems.
        schedule.set_build_settings(ScheduleBuildSettings {
            ambiguity_detection: LogLevel::Error,
            ..ScheduleBuildSettings::default()
        });
        if let Err(error) = schedule.initialize(world) {
            problems.push(format!(
                "{label:?}: {}",
                error.to_string(schedule.graph(), world)
            ));
        }
        world.resource_mut::<Schedules>().insert(schedule);
    }
    if problems.is_empty() {
        println!("The engine's schedules order every pair of conflicting systems.");
        return Ok(());
    }
    Err(problems.join("\n"))
}
