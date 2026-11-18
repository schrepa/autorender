use std::any::TypeId;
use std::sync::Arc;

use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::{
    ApplyDeferred, InternedScheduleLabel, Schedule, SystemKey, SystemWithAccess,
};
use bevy_ecs::system::{
    Adapt, AdapterSystem, IntoSystem, RunSystemError, System, SystemIn, SystemInput,
};
use engine_place::Board;
use engine_profile::Runs;

#[cfg(feature = "profiling")]
pub(crate) use read::*;

struct Attributed {
    brick: &'static str,
    name: &'static str,
    runs: Arc<Runs>,
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
        engine_profile::caused(engine_profile::cause(), || {
            engine_profile::run(&self.runs, self.brick, self.name, || {
                Board::owned_by(self.brick, || run_system(input))
            })
        })
    }
}

#[cfg(feature = "profiling")]
struct Wrapped {
    name: &'static str,
    brick: &'static str,
    schedule: InternedScheduleLabel,
    key: SystemKey,
    runs: Arc<Runs>,
}

#[derive(Resource)]
pub struct Systems {
    #[cfg(feature = "profiling")]
    schedules: Vec<InternedScheduleLabel>,
    #[cfg(feature = "profiling")]
    wrapped: Vec<Wrapped>,
}

#[cfg(feature = "profiling")]
impl Systems {
    pub fn new(schedules: &[InternedScheduleLabel]) -> Self {
        Self {
            schedules: schedules.to_vec(),
            wrapped: Vec::new(),
        }
    }

    fn adopt(
        &mut self,
        (schedule, key): (InternedScheduleLabel, SystemKey),
        brick: &'static str,
        system: &str,
    ) -> (&'static str, Arc<Runs>) {
        let named = format!("{schedule:?}:{system}");
        let repeats = self
            .wrapped
            .iter()
            .filter(|wrapped| wrapped.name.split('#').next() == Some(named.as_str()))
            .count();
        let name: &'static str = Box::leak(
            match repeats {
                0 => named,
                _ => format!("{named}#{}", repeats + 1),
            }
            .into_boxed_str(),
        );
        let runs: Arc<Runs> = Arc::default();
        self.wrapped.push(Wrapped {
            name,
            brick,
            schedule,
            key,
            runs: Arc::clone(&runs),
        });
        (name, runs)
    }
}

#[cfg(not(feature = "profiling"))]
impl Systems {
    pub fn new(_: &[InternedScheduleLabel]) -> Self {
        Self {}
    }

    fn adopt(
        &mut self,
        _: (InternedScheduleLabel, SystemKey),
        _: &'static str,
        _: &str,
    ) -> (&'static str, Arc<Runs>) {
        ("", Arc::default())
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
    brick: &'static str,
    systems: &mut Systems,
) {
    let label = schedule.label();
    let graph = &mut schedule.graph_mut().systems;
    let added: Vec<SystemKey> = graph
        .iter()
        .filter(|(key, system, _)| {
            !before.contains(key) && system.system_type() != TypeId::of::<ApplyDeferred>()
        })
        .map(|(key, _, _)| key)
        .collect();
    for key in added {
        let Some(slot) = graph.get_mut(key) else {
            continue;
        };
        let inner = std::mem::replace(
            slot,
            SystemWithAccess::new(Box::new(IntoSystem::<(), (), _>::into_system(idle))),
        );
        let debug = inner.name();
        let (name, runs) = systems.adopt((label, key), brick, &debug.to_string());
        let attributed = Attributed { brick, name, runs };
        *slot = SystemWithAccess::new(Box::new(AdapterSystem::new(attributed, inner, debug)));
    }
}

#[cfg(feature = "profiling")]
mod read {
    use std::any::TypeId;

    use bevy_ecs::schedule::{ApplyDeferred, Schedules};
    use bevy_ecs::world::World;
    use engine_profile::Runs;
    use serde_json::{Map, Value, json};

    use super::Systems;

    fn unavailable(reason: &str) -> Value {
        json!({ "unavailable": reason })
    }

    fn per_system<T: Into<Value>>(
        world: &World,
        read: impl Fn(&Runs) -> Result<T, &'static str>,
    ) -> Value {
        let Some(systems) = world.get_resource::<Systems>() else {
            return unavailable("systems_not_installed");
        };
        let mut out = Map::new();
        for wrapped in &systems.wrapped {
            match read(&wrapped.runs) {
                Ok(value) => out.insert(wrapped.name.to_owned(), value.into()),
                Err(reason) => return unavailable(reason),
            };
        }
        Value::Object(out)
    }

    fn per_brick(world: &World, read: impl Fn(&Runs) -> Result<u64, &'static str>) -> Value {
        let Some(systems) = world.get_resource::<Systems>() else {
            return unavailable("systems_not_installed");
        };
        let mut sums: Vec<(&'static str, u64)> = Vec::new();
        for wrapped in &systems.wrapped {
            let value = match read(&wrapped.runs) {
                Ok(value) => value,
                Err(reason) => return unavailable(reason),
            };
            match sums.iter_mut().find(|(brick, _)| *brick == wrapped.brick) {
                Some((_, sum)) => *sum = sum.wrapping_add(value),
                None => sums.push((wrapped.brick, value)),
            }
        }
        Value::Object(
            sums.into_iter()
                .map(|(brick, sum)| (brick.to_owned(), Value::from(sum)))
                .collect(),
        )
    }

    pub(crate) fn systems_runs(world: &World) -> Value {
        per_system(world, Runs::runs)
    }

    pub(crate) fn systems_time(world: &World) -> Value {
        per_system(world, Runs::ns)
    }

    pub(crate) fn systems_allocations(world: &World) -> Value {
        per_system(world, Runs::allocations)
    }

    pub(crate) fn systems_allocated_bytes(world: &World) -> Value {
        per_system(world, Runs::allocated_bytes)
    }

    pub(crate) fn systems_brick(world: &World) -> Value {
        let Some(systems) = world.get_resource::<Systems>() else {
            return unavailable("systems_not_installed");
        };
        Value::Object(
            systems
                .wrapped
                .iter()
                .map(|wrapped| (wrapped.name.to_owned(), Value::from(wrapped.brick)))
                .collect(),
        )
    }

    pub(crate) fn bricks_runs(world: &World) -> Value {
        per_brick(world, Runs::runs)
    }

    pub(crate) fn bricks_time(world: &World) -> Value {
        per_brick(world, Runs::ns)
    }

    pub(crate) fn bricks_allocations(world: &World) -> Value {
        per_brick(world, Runs::allocations)
    }

    pub(crate) fn bricks_allocated_bytes(world: &World) -> Value {
        per_brick(world, Runs::allocated_bytes)
    }

    pub(crate) fn systems_outside(world: &World) -> Value {
        let Some(systems) = world.get_resource::<Systems>() else {
            return unavailable("systems_not_installed");
        };
        let Some(schedules) = world.get_resource::<Schedules>() else {
            return unavailable("schedules_missing");
        };
        let outside = systems
            .schedules
            .iter()
            .filter_map(|&label| schedules.get(label).map(|schedule| (label, schedule)))
            .flat_map(|(label, schedule)| {
                schedule
                    .graph()
                    .systems
                    .iter()
                    .filter(move |(key, system, _)| {
                        system.system_type() != TypeId::of::<ApplyDeferred>()
                            && !systems
                                .wrapped
                                .iter()
                                .any(|wrapped| wrapped.schedule == label && wrapped.key == *key)
                    })
            })
            .count();
        Value::from(outside)
    }
}
