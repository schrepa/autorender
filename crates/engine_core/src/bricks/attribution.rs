#[cfg(any(feature = "dev", feature = "profiling"))]
pub fn under<T>(brick: &'static str, run: impl FnOnce() -> T) -> T {
    let _span = engine_profile::span("brick", brick);
    let _scope = engine_profile::enter(engine_profile::brick_index(brick));
    run()
}

#[cfg(not(any(feature = "dev", feature = "profiling")))]
#[inline(always)]
pub fn under<T>(_brick: &'static str, run: impl FnOnce() -> T) -> T {
    run()
}

#[cfg(any(feature = "dev", feature = "profiling"))]
pub(crate) use measured::{measure_systems, system_keys};

#[cfg(any(feature = "dev", feature = "profiling"))]
mod measured {
    use std::any::TypeId;

    use bevy_app::App;
    use bevy_ecs::schedule::{ApplyDeferred, InternedScheduleLabel, SystemKey, SystemWithAccess};
    use bevy_ecs::system::{
        Adapt, AdapterSystem, IntoSystem, RunSystemError, System, SystemIn, SystemInput,
    };

    struct Measured {
        brick: &'static str,
        index: u16,
    }

    impl Adapt<SystemWithAccess> for Measured {
        type In = ();
        type Out = ();

        fn adapt(
            &mut self,
            input: <Self::In as SystemInput>::Inner<'_>,
            run_system: impl FnOnce(
                SystemIn<'_, SystemWithAccess>,
            )
                -> Result<<SystemWithAccess as System>::Out, RunSystemError>,
        ) -> Result<Self::Out, RunSystemError> {
            let _span = engine_profile::span("brick", self.brick);
            let _scope = engine_profile::enter(self.index);
            run_system(input)
        }
    }

    fn idle() {}

    pub(crate) fn system_keys(
        app: &App,
        schedule: InternedScheduleLabel,
    ) -> Vec<SystemKey> {
        app.get_schedule(schedule)
            .map(|schedule| {
                schedule
                    .graph()
                    .systems
                    .iter()
                    .map(|(key, _, _)| key)
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(crate) fn measure_systems(
        app: &mut App,
        schedule: InternedScheduleLabel,
        before: &[SystemKey],
        brick: &'static str,
    ) {
        let index = engine_profile::brick_index(brick);
        let Some(schedule) = app.get_schedule_mut(schedule) else {
            return;
        };
        let systems = &mut schedule.graph_mut().systems;
        let added: Vec<SystemKey> = systems
            .iter()
            .filter(|(key, system, _)| {
                !before.contains(key) && system.system_type() != TypeId::of::<ApplyDeferred>()
            })
            .map(|(key, _, _)| key)
            .collect();
        for key in added {
            let Some(slot) = systems.get_mut(key) else {
                continue;
            };
            let inner = std::mem::replace(
                slot,
                SystemWithAccess::new(Box::new(IntoSystem::<(), (), _>::into_system(idle))),
            );
            let name = inner.name();
            *slot = SystemWithAccess::new(Box::new(AdapterSystem::new(
                Measured { brick, index },
                inner,
                name,
            )));
        }
    }
}
