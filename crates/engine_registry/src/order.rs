use bevy_ecs::component::ComponentId;
use bevy_ecs::query::FilteredAccessSet;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::graph::{DiGraph, Direction};
use bevy_ecs::schedule::passes::AutoInsertApplyDeferredPass;
use bevy_ecs::schedule::{
    FlattenedDependencies, InternedScheduleLabel, InternedSystemSet, LogLevel, NodeId, Schedule,
    ScheduleBuildError, ScheduleBuildPass, ScheduleBuildSettings, ScheduleBuildWarning,
    ScheduleGraph, SystemKey, SystemSetKey,
};
use bevy_ecs::system::System;
use bevy_ecs::world::World;
use bevy_platform::hash::FixedHasher;
use engine_place::{Error, ErrorCode};
use indexmap::IndexSet;

use crate::BrickId;
use crate::access::AccessRule;

#[derive(Resource, Default)]
pub(crate) struct PlaceProblems(pub(crate) Vec<Error>);

#[derive(Clone, Debug)]
pub(crate) struct PlaceRules {
    pub(crate) schedule: InternedScheduleLabel,
    pub(crate) bricks: Box<[(InternedSystemSet, BrickId)]>,
    pub(crate) slots: Box<[(ComponentId, Option<BrickId>)]>,
    pub(crate) firsts: Box<[ComponentId]>,
    pub(crate) rule: AccessRule,
}

impl ScheduleBuildPass for PlaceRules {
    type EdgeOptions = ();

    fn add_dependency(&mut self, _: NodeId, _: NodeId, _: Option<&Self::EdgeOptions>) {}

    fn collapse_set(
        &mut self,
        _: SystemSetKey,
        _: &IndexSet<SystemKey, FixedHasher>,
        _: &DiGraph<NodeId>,
    ) -> impl Iterator<Item = (NodeId, NodeId)> {
        std::iter::empty()
    }

    fn build(
        &mut self,
        world: &mut World,
        graph: &mut ScheduleGraph,
        mut flat: FlattenedDependencies<'_>,
    ) -> Result<(), ScheduleBuildError> {
        let mut problems: Vec<Error> = Vec::new();
        let mut found: Vec<(SystemKey, BrickId, FilteredAccessSet)> = Vec::new();
        for &(set, brick) in &*self.bricks {
            let Some(key) = graph.system_sets.get_key(set) else {
                continue;
            };
            for system in systems_under(graph, key) {
                if found.iter().any(|(known, _, _)| *known == system) {
                    continue;
                }
                let Some(node) = graph.systems.get_mut(system) else {
                    continue;
                };
                let last = node.get_last_run();
                let mut access = node.initialize(world);
                node.set_last_run(last);
                let name = node.name();
                for condition in graph.systems.get_conditions(system).into_iter().flatten() {
                    access.extend(condition.access.clone());
                }
                self.rule
                    .check_into(brick, &name, &access, world, &mut problems);
                found.push((system, brick, access));
            }
        }
        for (key, system, _) in graph.systems.iter() {
            if graph.ambiguous_with_all.contains(&NodeId::System(key))
                || found.iter().any(|(known, _, _)| *known == key)
            {
                continue;
            }
            let name = system.name();
            let path = format!("$.schedules.{:?}.{}", self.schedule, &*name);
            problems.push(Error::new(ErrorCode::UnownedSystem, path).got(&*name));
        }
        for &(slot, filler) in &*self.slots {
            let Some(filler) = filler else {
                continue;
            };
            let writers = found.iter().filter(|(_, brick, access)| {
                *brick == filler && access.combined_access().has_write(slot)
            });
            for (writer, _, _) in writers {
                let readers = found.iter().filter(|(_, brick, access)| {
                    *brick != filler && access.combined_access().has_read(slot)
                });
                for (reader, _, _) in readers {
                    flat.add_edge(*writer, *reader);
                }
            }
        }
        for &first in &*self.firsts {
            let writes = |access: &FilteredAccessSet| access.combined_access().has_write(first);
            let reads = |access: &FilteredAccessSet| access.combined_access().has_read(first);
            for (writer, writing, _) in found.iter().filter(|(_, _, access)| writes(access)) {
                let readers = found.iter().filter(|(_, reading, access)| {
                    reading != writing && !writes(access) && reads(access)
                });
                for (reader, _, _) in readers {
                    flat.add_edge(*writer, *reader);
                }
            }
        }
        world
            .get_resource_or_init::<PlaceProblems>()
            .0
            .append(&mut problems);
        Ok(())
    }
}

fn systems_under(graph: &ScheduleGraph, set: SystemSetKey) -> Vec<SystemKey> {
    let hierarchy = graph.hierarchy();
    let mut stack = vec![NodeId::Set(set)];
    let mut seen: Vec<NodeId> = Vec::new();
    let mut systems: Vec<SystemKey> = Vec::new();
    while let Some(node) = stack.pop() {
        for child in hierarchy.neighbors_directed(node, Direction::Outgoing) {
            if seen.contains(&child) {
                continue;
            }
            seen.push(child);
            match child {
                NodeId::System(system) => systems.push(system),
                NodeId::Set(_) => stack.push(child),
            }
        }
    }
    systems
}

pub(crate) fn order(schedule: &mut Schedule, rules: PlaceRules) {
    schedule.set_build_settings(ScheduleBuildSettings {
        ambiguity_detection: LogLevel::Error,
        ..ScheduleBuildSettings::new()
    });
    schedule.remove_build_pass::<AutoInsertApplyDeferredPass>();
    schedule.add_build_pass(rules);
    schedule.add_build_pass(AutoInsertApplyDeferredPass::default());
}

pub(crate) fn build(world: &mut World, labels: &[InternedScheduleLabel]) -> Vec<Error> {
    let mut refused: Vec<Error> = Vec::new();
    for &label in labels {
        let built =
            world.try_schedule_scope(label, |world, schedule| match schedule.initialize(world) {
                Ok(_) => None,
                Err(error) => Some(refusal(label, &error, schedule.graph(), world)),
            });
        if let Ok(Some(error)) = built {
            refused.push(error);
        }
    }
    let mut problems = world
        .remove_resource::<PlaceProblems>()
        .map(|place| place.0)
        .unwrap_or_default();
    problems.append(&mut refused);
    problems
}

fn refusal(
    label: InternedScheduleLabel,
    error: &ScheduleBuildError,
    graph: &ScheduleGraph,
    world: &World,
) -> Error {
    let code = match error {
        ScheduleBuildError::DependencySort(_) | ScheduleBuildError::FlatDependencySort(_) => {
            ErrorCode::PlaceCycle
        }
        ScheduleBuildError::Elevated(ScheduleBuildWarning::Ambiguity(_)) => {
            ErrorCode::ScheduleAmbiguity
        }
        _ => ErrorCode::InvalidValue,
    };
    Error::new(code, format!("$.schedules.{label:?}")).got(error.to_string(graph, world))
}
