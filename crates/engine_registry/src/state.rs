use std::any::TypeId;

use bevy_ecs::change_detection::{CheckChangeTicks, Res};
use bevy_ecs::component::ComponentId;
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{ReadOnlySystem, RunSystemError};
use bevy_ecs::world::World;
use engine_place::{Error, ErrorCode, EventPage, Feed, page_of};
use serde::Serialize;
use serde_json::Value;

use crate::BrickId;
use crate::registry::Registry;

pub type GateFn = Box<dyn ReadOnlySystem<In = (), Out = bool>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Scope(pub(crate) u32);

impl Scope {
    pub const BRICK: Self = Self(0);

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GateId(pub(crate) u32);

impl GateId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "because", rename_all = "snake_case")]
pub enum Off {
    Unstarted,
    Condition,
    Parent,
    Needs { brick: &'static str },
    Cleared,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BrickState {
    Off(Off),
    On,
    Faulted(Error),
}

impl BrickState {
    pub fn on(&self) -> bool {
        matches!(self, Self::On)
    }
}

pub const BRICK_STATE_EVENT: &str = "brick_state";

const KEPT_CHANGES: usize = 256;

#[derive(Clone, Debug)]
pub struct BrickChange {
    pub brick: BrickId,
    pub gate: GateId,
    pub state: BrickState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BrickStateWord {
    Off,
    On,
    Faulted,
}

#[derive(Clone, Debug, PartialEq, Serialize, schemars::JsonSchema)]
pub struct BrickChangeRow {
    pub brick: String,
    pub whole: bool,
    pub state: BrickStateWord,
    pub off: Option<Value>,
    pub error: Option<Value>,
}

#[derive(Resource)]
pub struct BrickStates {
    states: Vec<BrickState>,
    parents: Vec<Option<GateId>>,
    bricks: Vec<BrickId>,
    roots: Vec<GateId>,
    types: Vec<TypeId>,
    needs: Vec<Vec<BrickId>>,
    installs: Vec<u64>,
    generation: u64,
    changes: Feed<BrickChange>,
}

impl Default for BrickStates {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new(), Vec::new(), Vec::new())
    }
}

pub fn brick_changes(world: &World, after: u64) -> Option<EventPage> {
    let states = world.get_resource::<BrickStates>()?;
    let registry = world.get_resource::<Registry>();
    Some(page_of(states.changes(), after, |change| BrickChangeRow {
        brick: registry
            .and_then(|registry| registry.record(change.brick))
            .map_or_else(String::new, |record| record.package.to_owned()),
        whole: states.root(change.brick) == Some(change.gate),
        state: match &change.state {
            BrickState::Off(_) => BrickStateWord::Off,
            BrickState::On => BrickStateWord::On,
            BrickState::Faulted(_) => BrickStateWord::Faulted,
        },
        off: match &change.state {
            BrickState::Off(off) => serde_json::to_value(off).ok(),
            BrickState::On | BrickState::Faulted(_) => None,
        },
        error: match &change.state {
            BrickState::Faulted(error) => serde_json::to_value(error).ok(),
            BrickState::Off(_) | BrickState::On => None,
        },
    }))
}

impl BrickStates {
    pub(crate) fn new(
        parents: Vec<Option<GateId>>,
        bricks: Vec<BrickId>,
        roots: Vec<GateId>,
        types: Vec<TypeId>,
    ) -> Self {
        let gates = parents.len();
        let count = roots.len();
        Self {
            states: vec![BrickState::Off(Off::Unstarted); gates],
            parents,
            bricks,
            roots,
            types,
            needs: vec![Vec::new(); count],
            installs: vec![0; gates],
            generation: 0,
            changes: Feed::new(KEPT_CHANGES),
        }
    }

    pub(crate) fn set_needs(&mut self, needs: Vec<Vec<BrickId>>) {
        self.needs = needs;
    }

    pub fn state(&self, brick: BrickId) -> Option<&BrickState> {
        self.gate_state(self.root(brick)?)
    }

    pub fn gate_state(&self, gate: GateId) -> Option<&BrickState> {
        self.states.get(gate.index())
    }

    pub fn on(&self, gate: GateId) -> bool {
        self.gate_state(gate).is_some_and(BrickState::on)
    }

    pub fn brick_on(&self, brick: BrickId) -> bool {
        self.state(brick).is_some_and(BrickState::on)
    }

    pub fn installed(&self, gate: GateId) -> Option<u64> {
        self.on(gate)
            .then(|| self.installs.get(gate.index()).copied())
            .flatten()
    }

    pub fn brick(&self, gate: GateId) -> Option<BrickId> {
        self.bricks.get(gate.index()).copied()
    }

    pub fn root(&self, brick: BrickId) -> Option<GateId> {
        self.roots.get(brick.index()).copied()
    }

    pub fn parent(&self, gate: GateId) -> Option<GateId> {
        self.parents.get(gate.index()).copied().flatten()
    }

    pub fn needs(&self, brick: BrickId) -> &[BrickId] {
        self.needs.get(brick.index()).map_or(&[], Vec::as_slice)
    }

    pub fn id_of(&self, brick: TypeId) -> Option<BrickId> {
        self.types
            .iter()
            .position(|known| *known == brick)
            .map(|index| BrickId(index as u32))
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn changes(&self) -> &Feed<BrickChange> {
        &self.changes
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (BrickId, &BrickState)> + '_ {
        self.roots
            .iter()
            .enumerate()
            .filter_map(|(index, root)| Some((BrickId(index as u32), self.gate_state(*root)?)))
    }

    pub fn faults(&self) -> impl Iterator<Item = (BrickId, &Error)> + '_ {
        self.iter().filter_map(|(brick, state)| match state {
            BrickState::Faulted(error) => Some((brick, error)),
            BrickState::Off(_) | BrickState::On => None,
        })
    }

    fn children(&self, gate: GateId) -> Vec<GateId> {
        self.parents
            .iter()
            .enumerate()
            .filter(|(_, parent)| **parent == Some(gate))
            .map(|(index, _)| GateId(index as u32))
            .collect()
    }

    fn dependents(&self, brick: BrickId) -> Vec<BrickId> {
        self.needs
            .iter()
            .enumerate()
            .filter(|(_, needs)| needs.contains(&brick))
            .map(|(index, _)| BrickId(index as u32))
            .collect()
    }

    fn faulted(&self, gate: GateId) -> bool {
        self.brick(gate)
            .and_then(|brick| self.state(brick))
            .is_some_and(|state| matches!(state, BrickState::Faulted(_)))
    }

    fn set(&mut self, gate: GateId, state: BrickState) {
        let Some(held) = self.states.get_mut(gate.index()) else {
            return;
        };
        *held = state.clone();
        self.generation = self.generation.wrapping_add(1);
        if let Some(brick) = self.brick(gate) {
            self.changes.push(BrickChange { brick, gate, state });
        }
    }

    fn explain(&mut self, gate: GateId, off: Off) {
        if let Some(BrickState::Off(held)) = self.states.get_mut(gate.index()) {
            *held = off;
        }
    }

    fn started(&mut self, gate: GateId) {
        self.set(gate, BrickState::On);
        if let Some(installs) = self.installs.get_mut(gate.index()) {
            *installs = installs.wrapping_add(1);
        }
    }
}

pub(crate) fn state_on(
    gate: GateId,
) -> impl FnMut(Res<BrickStates>) -> bool + Clone + Send + Sync + 'static {
    move |states: Res<BrickStates>| states.on(gate)
}

pub fn fault(world: &mut World, brick: BrickId, error: Error) {
    let root = world
        .get_resource::<BrickStates>()
        .and_then(|states| states.root(brick));
    if let Some(root) = root {
        turn(world, root, BrickState::Faulted(error));
    }
}

pub fn fault_of<B: 'static>(world: &mut World, error: Error) {
    let brick = world
        .get_resource::<BrickStates>()
        .and_then(|states| states.id_of(TypeId::of::<B>()));
    if let Some(brick) = brick {
        fault(world, brick, error);
    }
}

pub fn ask_on(world: &mut World, brick: BrickId) {
    let root = world
        .get_resource::<BrickStates>()
        .and_then(|states| states.root(brick).filter(|root| states.faulted(*root)));
    if let Some(root) = root {
        turn(world, root, BrickState::Off(Off::Cleared));
    }
}

pub(crate) fn turn(world: &mut World, gate: GateId, to: BrickState) {
    let Some(from) = world
        .get_resource::<BrickStates>()
        .and_then(|states| states.gate_state(gate))
        .cloned()
    else {
        return;
    };
    match (from, to) {
        (BrickState::Off(_), BrickState::On) => turn_on(world, gate),
        (BrickState::On, to @ (BrickState::Off(_) | BrickState::Faulted(_))) => {
            turn_off(world, gate, to);
        }
        (BrickState::Off(_), to @ BrickState::Faulted(_))
        | (BrickState::Faulted(_), to @ BrickState::Off(_)) => {
            if let Some(mut states) = world.get_resource_mut::<BrickStates>() {
                states.set(gate, to);
            }
        }
        (BrickState::Off(_), BrickState::Off(off)) => {
            if let Some(mut states) = world.get_resource_mut::<BrickStates>() {
                states.explain(gate, off);
            }
        }
        (BrickState::On, BrickState::On)
        | (BrickState::Faulted(_), BrickState::On | BrickState::Faulted(_)) => {}
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Reset {
    Remove(ComponentId),
    Typed(fn(&mut World)),
}

impl Reset {
    fn apply(self, world: &mut World) {
        match self {
            Self::Remove(id) => {
                for entity in holders(world, id) {
                    if let Ok(mut held) = world.get_entity_mut(entity) {
                        held.remove_by_id(id);
                    }
                }
            }
            Self::Typed(reset) => reset(world),
        }
    }
}

pub type Despawn = fn(&mut World, &[Entity]);

type Turn = fn(&mut World);

#[derive(Default)]
pub(crate) struct Turning {
    pub(crate) resources: Vec<(Turn, Turn)>,
    pub(crate) settings: Vec<fn(&mut World)>,
    pub(crate) slots: Vec<fn(&mut World)>,
    pub(crate) joins: Vec<fn(&mut World, GateId)>,
    pub(crate) private: Vec<Reset>,
    pub(crate) spawned: Option<fn(&mut World) -> ComponentId>,
    pub(crate) despawn: Option<Despawn>,
}

fn despawn_all(world: &mut World, entities: &[Entity]) {
    for &entity in entities {
        world.despawn(entity);
    }
}

fn turning(world: &World, brick: BrickId, gate: GateId) -> Turning {
    world
        .get_resource::<Registry>()
        .map(|registry| registry.turning(brick, gate))
        .unwrap_or_default()
}

fn turn_on(world: &mut World, gate: GateId) {
    let Some(brick) = world
        .get_resource::<BrickStates>()
        .and_then(|states| states.brick(gate))
    else {
        return;
    };
    let work = turning(world, brick, gate);
    for &(insert, _) in &work.resources {
        insert(world);
    }
    for reset in &work.private {
        reset.apply(world);
    }
    for touch in &work.settings {
        touch(world);
    }
    if let Some(mut states) = world.get_resource_mut::<BrickStates>() {
        states.started(gate);
    }
}

fn turn_off(world: &mut World, gate: GateId, to: BrickState) {
    let Some((brick, children, dependents)) = world.get_resource::<BrickStates>().and_then(|states| {
        let brick = states.brick(gate)?;
        let dependents = if states.root(brick) == Some(gate) {
            states.dependents(brick)
        } else {
            Vec::new()
        };
        Some((brick, states.children(gate), dependents))
    }) else {
        return;
    };
    if let Some(mut states) = world.get_resource_mut::<BrickStates>() {
        states.set(gate, to);
    }
    for child in children {
        turn(world, child, BrickState::Off(Off::Parent));
    }
    let needed = Off::Needs {
        brick: package(world, brick),
    };
    for dependent in dependents {
        let root = world
            .get_resource::<BrickStates>()
            .and_then(|states| states.root(dependent));
        if let Some(root) = root {
            turn(world, root, BrickState::Off(needed));
        }
    }
    let work = turning(world, brick, gate);
    if let Some(spawned) = work.spawned {
        let id = spawned(world);
        let owned = holders(world, id);
        work.despawn.unwrap_or(despawn_all)(world, &owned);
    }
    for &(_, remove) in &work.resources {
        remove(world);
    }
    for reset in &work.slots {
        reset(world);
    }
    for left in &work.joins {
        left(world, gate);
    }
}

pub(crate) fn holders(world: &World, id: ComponentId) -> Vec<Entity> {
    world
        .archetypes()
        .iter()
        .filter(|archetype| archetype.contains(id))
        .flat_map(|archetype| archetype.entities().iter().map(|held| held.id()))
        .collect()
}

pub struct GateDriver {
    conditions: Box<[Option<GateFn>]>,
}

impl GateDriver {
    pub(crate) fn new(conditions: Vec<Option<GateFn>>) -> Self {
        Self {
            conditions: conditions.into_boxed_slice(),
        }
    }

    pub fn run(&mut self, world: &mut World) {
        for _ in 0..=self.conditions.len() {
            let before = generation(world);
            let Some(want) = self.wanted(world) else {
                return;
            };
            for index in (0..want.len()).rev() {
                let gate = GateId(index as u32);
                let turnable = state_of(world, gate)
                    .is_some_and(|state| matches!(state, BrickState::On | BrickState::Off(_)));
                if let (Err(off), true) = (want[index], turnable) {
                    turn(world, gate, BrickState::Off(off));
                }
            }
            for (index, wanted) in want.iter().enumerate() {
                let gate = GateId(index as u32);
                if wanted.is_ok() && matches!(state_of(world, gate), Some(BrickState::Off(_))) {
                    turn(world, gate, BrickState::On);
                }
            }
            if generation(world) == before {
                return;
            }
        }
    }

    fn wanted(&mut self, world: &mut World) -> Option<Vec<Result<(), Off>>> {
        let count = self.conditions.len();
        let mut want: Vec<Result<(), Off>> = vec![Err(Off::Unstarted); count];
        for index in 0..count {
            let gate = GateId(index as u32);
            let (brick, parent, faulted, lacking) = {
                let states = world.get_resource::<BrickStates>()?;
                let brick = states.brick(gate)?;
                let parent = states.parent(gate);
                let lacking = parent
                    .is_none()
                    .then(|| {
                        states
                            .needs(brick)
                            .iter()
                            .copied()
                            .find(|need| !states.brick_on(*need))
                    })
                    .flatten();
                (brick, parent, states.faulted(gate), lacking)
            };
            if faulted {
                continue;
            }
            if parent.is_some_and(|parent| want[parent.index()].is_err()) {
                want[index] = Err(Off::Parent);
                continue;
            }
            want[index] = match self.conditions[index].as_mut() {
                None => Ok(()),
                Some(condition) => match condition.run_readonly((), world) {
                    Ok(true) => Ok(()),
                    Ok(false) => Err(Off::Condition),
                    Err(error) => match lacking {
                        Some(need) => Err(Off::Needs {
                            brick: package(world, need),
                        }),
                        None => {
                            let refused = refused(world, brick, &error);
                            fault(world, brick, refused);
                            Err(Off::Condition)
                        }
                    },
                },
            };
        }
        let states = world.get_resource::<BrickStates>()?;
        loop {
            let mut dropped = false;
            for (brick, root) in states.roots.iter().enumerate() {
                let needs = states.needs.get(brick).map_or(&[][..], Vec::as_slice);
                let lacking = needs.iter().copied().find(|need| {
                    states
                        .root(*need)
                        .is_none_or(|need| want[need.index()].is_err())
                });
                if let (Ok(()), Some(need)) = (want[root.index()], lacking) {
                    want[root.index()] = Err(Off::Needs {
                        brick: package(world, need),
                    });
                    dropped = true;
                }
            }
            if !dropped {
                break;
            }
        }
        for index in 0..count {
            if let Some(parent) = states.parent(GateId(index as u32))
                && want[parent.index()].is_err()
                && want[index].is_ok()
            {
                want[index] = Err(Off::Parent);
            }
        }
        Some(want)
    }

    pub fn check_change_ticks(&mut self, check: CheckChangeTicks) {
        for condition in self.conditions.iter_mut().flatten() {
            condition.check_change_tick(check);
        }
    }
}

fn generation(world: &World) -> u64 {
    world
        .get_resource::<BrickStates>()
        .map_or(0, BrickStates::generation)
}

fn state_of(world: &World, gate: GateId) -> Option<&BrickState> {
    world.get_resource::<BrickStates>()?.gate_state(gate)
}

fn package(world: &World, brick: BrickId) -> &'static str {
    world
        .get_resource::<Registry>()
        .and_then(|registry| registry.record(brick))
        .map_or("", |record| record.package)
}

fn refused(world: &World, brick: BrickId, error: &RunSystemError) -> Error {
    let package = world
        .get_resource::<Registry>()
        .and_then(|registry| registry.record(brick))
        .map_or("", |record| record.package);
    Error::new(
        ErrorCode::MissingResource,
        format!("$.bricks.{package}.when"),
    )
    .got(error.to_string())
}
