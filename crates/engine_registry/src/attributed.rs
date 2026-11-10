use std::any::TypeId;

use bevy_ecs::schedule::{ApplyDeferred, Schedule, SystemKey, SystemWithAccess};
use bevy_ecs::system::{
    Adapt, AdapterSystem, IntoSystem, RunSystemError, System, SystemIn, SystemInput,
};
use engine_place::Board;

struct Attributed {
    brick: &'static str,
    #[cfg(feature = "profiling")]
    index: u16,
    #[cfg(feature = "profiling")]
    row: usize,
}

impl Adapt<SystemWithAccess> for Attributed {
    type In = ();
    type Out = ();

    fn adapt(
        &mut self,
        input: <Self::In as SystemInput>::Inner<'_>,
        run_system: impl FnOnce(
            SystemIn<'_, SystemWithAccess>,
        ) -> Result<<SystemWithAccess as System>::Out, RunSystemError>,
    ) -> Result<Self::Out, RunSystemError> {
        #[cfg(feature = "profiling")]
        let _span = engine_profile::span("brick", self.brick);
        #[cfg(feature = "profiling")]
        let _scope = engine_profile::enter(self.index);
        #[cfg(feature = "profiling")]
        let _row = engine_profile::timed(self.row);
        Board::owned_by(self.brick, || run_system(input))
    }
}

fn idle() {}

pub(crate) fn keys(schedule: &Schedule) -> Vec<SystemKey> {
    schedule
        .graph()
        .systems
        .iter()
        .map(|(key, _, _)| key)
        .collect()
}

pub(crate) fn wrap(
    schedule: &mut Schedule,
    before: &[SystemKey],
    (brick, index): (&'static str, u16),
) {
    #[cfg(not(feature = "profiling"))]
    let _ = index;
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
        let attributed = Attributed {
            brick,
            #[cfg(feature = "profiling")]
            index,
            #[cfg(feature = "profiling")]
            row: engine_profile::row(Box::leak(name.to_string().into_boxed_str())),
        };
        *slot = SystemWithAccess::new(Box::new(AdapterSystem::new(attributed, inner, name)));
    }
}
