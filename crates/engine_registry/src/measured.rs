use std::any::TypeId;

use bevy_ecs::schedule::{ApplyDeferred, Schedule, SystemKey, SystemWithAccess};
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
        ) -> Result<<SystemWithAccess as System>::Out, RunSystemError>,
    ) -> Result<Self::Out, RunSystemError> {
        let _span = engine_profile::span("brick", self.brick);
        let _scope = engine_profile::enter(self.index);
        run_system(input)
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
