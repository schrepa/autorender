use bevy_ecs::component::ComponentId;
use bevy_ecs::query::FilteredAccessSet;
use bevy_ecs::resource::Resource;
use bevy_ecs::schedule::graph::{Dag, DiGraph, DiGraphToposortError, Direction};
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
use serde_json::{Map, Value, json};

use crate::BrickId;
use crate::access::AccessRule;

type Found = (SystemKey, BrickId, FilteredAccessSet);

type Forced = (SystemKey, SystemKey, ComponentId);

#[derive(Resource, Default)]
pub(crate) struct PlaceProblems(pub(crate) Vec<Error>);

#[derive(Resource, Default)]
struct PlaceCycles(Vec<(InternedScheduleLabel, Value)>);

#[derive(Clone, Debug)]
pub(crate) struct PlaceRules {
    pub(crate) schedule: InternedScheduleLabel,
    pub(crate) bricks: Box<[(InternedSystemSet, BrickId)]>,
    pub(crate) slots: Box<[(ComponentId, Option<BrickId>)]>,
    pub(crate) firsts: Box<[ComponentId]>,
    pub(crate) orders: Box<[(BrickId, BrickId)]>,
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
        let mut found: Vec<Found> = Vec::new();
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
                let exclusive = node.is_exclusive();
                let last = node.get_last_run();
                let mut access = node.initialize(world);
                node.set_last_run(last);
                let name = node.name();
                for condition in graph.systems.get_conditions(system).into_iter().flatten() {
                    access.extend(condition.access.clone());
                }
                self.rule
                    .check_into(brick, &name, &access, world, &mut problems);
                if exclusive {
                    self.rule.check_exclusive(brick, &name, &mut problems);
                }
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
        let forced = self.forced(&found);
        for &(writer, reader, _) in &forced {
            flat.add_edge(writer, reader);
        }
        for &(first, then) in &*self.orders {
            for (early, _, early_access) in found.iter().filter(|(_, brick, _)| *brick == first) {
                let lates = found.iter().filter(|(_, brick, access)| {
                    *brick == then && !early_access.is_compatible(access)
                });
                for (late, _, _) in lates {
                    if !reaches(flat.graph(), *late, *early) {
                        flat.add_edge(*early, *late);
                    }
                }
            }
        }
        if let Some(DiGraphToposortError::Cycle(cycles)) = flat.toposort().err() {
            let facts = self.cycle_facts(&cycles, flat.graph(), &found, &forced, graph, world);
            world
                .get_resource_or_init::<PlaceCycles>()
                .0
                .push((self.schedule, facts));
        }
        world
            .get_resource_or_init::<PlaceProblems>()
            .0
            .append(&mut problems);
        Ok(())
    }
}

impl PlaceRules {
    fn forced(&self, found: &[Found]) -> Vec<Forced> {
        let mut forced = Vec::new();
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
                    forced.push((*writer, *reader, slot));
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
                    forced.push((*writer, *reader, first));
                }
            }
        }
        forced
    }

    fn cycle_facts(
        &self,
        cycles: &[Vec<SystemKey>],
        flat: &DiGraph<SystemKey>,
        found: &[Found],
        forced: &[Forced],
        graph: &ScheduleGraph,
        world: &World,
    ) -> Value {
        let name = |system: SystemKey| {
            graph
                .systems
                .get(system)
                .map(|node| node.name().to_string())
        };
        let mut steps: Vec<Value> = Vec::new();
        for cycle in cycles {
            for (&first, &then) in cycle.iter().zip(cycle.iter().cycle().skip(1)) {
                let place = forced
                    .iter()
                    .find(|&&(writer, reader, _)| (writer, reader) == (first, then))
                    .and_then(|&(_, _, place)| world.components().get_info(place))
                    .map(|info| info.name().to_string());
                let step = match place {
                    Some(place) => json!({
                        "first": name(first),
                        "then": name(then),
                        "writer_before_reader": place,
                    }),
                    None => json!({
                        "first": name(first),
                        "then": name(then),
                        "listed_in": owner(found, first).and_then(|brick| self.rule.package(brick)),
                    }),
                };
                if !steps.contains(&step) {
                    steps.push(step);
                }
            }
        }
        let system_order = match listing(flat, found, forced) {
            Some((order, moved)) => {
                let mut listed = Map::new();
                for brick in moved {
                    let systems: Vec<Option<String>> = order
                        .iter()
                        .filter(|system| owner(found, **system) == Some(brick))
                        .map(|system| name(*system))
                        .collect();
                    let package = self.rule.package(brick).unwrap_or_default();
                    listed.insert(package.to_owned(), json!(systems));
                }
                Value::Object(listed)
            }
            None => json!("no_order_fixes_this"),
        };
        json!({ "steps": steps, "brick_order": "no_effect", "system_order": system_order })
    }
}

fn owner(found: &[Found], system: SystemKey) -> Option<BrickId> {
    found
        .iter()
        .find(|(known, _, _)| *known == system)
        .map(|(_, brick, _)| *brick)
}

fn listing(
    flat: &DiGraph<SystemKey>,
    found: &[Found],
    forced: &[Forced],
) -> Option<(Vec<SystemKey>, Vec<BrickId>)> {
    let mut kept: Dag<SystemKey> = Dag::new();
    for (system, _, _) in found {
        kept.add_node(*system);
    }
    for &(writer, reader, _) in forced {
        kept.add_edge(writer, reader);
    }
    kept.toposort().ok()?;
    let mut moved: Vec<BrickId> = Vec::new();
    for (before, after) in flat.all_edges() {
        let Some(brick) = owner(found, before).filter(|&brick| owner(found, after) == Some(brick))
        else {
            continue;
        };
        if !reaches(kept.graph(), after, before) {
            kept.add_edge(before, after);
        } else if !moved.contains(&brick) {
            moved.push(brick);
        }
    }
    Some((kept.toposort().ok()?.to_vec(), moved))
}

fn reaches(graph: &DiGraph<SystemKey>, from: SystemKey, to: SystemKey) -> bool {
    let mut stack = vec![from];
    let mut seen = vec![from];
    while let Some(node) = stack.pop() {
        if node == to {
            return true;
        }
        for next in graph.neighbors_directed(node, Direction::Outgoing) {
            if !seen.contains(&next) {
                seen.push(next);
                stack.push(next);
            }
        }
    }
    false
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
    world.remove_resource::<PlaceCycles>();
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
    let text = error.to_string(graph, world);
    let lines = text.lines().collect::<Vec<_>>();
    let refused = Error::new(code, format!("$.schedules.{label:?}"));
    let facts = world
        .get_resource::<PlaceCycles>()
        .and_then(|cycles| cycles.0.iter().find(|(schedule, _)| *schedule == label));
    match facts {
        Some((_, facts)) => {
            let mut got = facts.clone();
            got["cycles"] = json!(lines);
            refused.got(got)
        }
        None => refused.got(lines),
    }
}
