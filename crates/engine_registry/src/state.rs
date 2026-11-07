use std::any::TypeId;

use bevy_ecs::change_detection::{CheckChangeTicks, Res};
use bevy_ecs::component::ComponentId;
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{ReadOnlySystem, RunSystemError};
use bevy_ecs::world::World;
use engine_place::{Error, ErrorCode};
use serde::Serialize;

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

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BrickState {
    Off,
    On,
    Faulted(Error),
}

impl BrickState {
    pub fn on(&self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Resource, Default)]
pub struct BrickStates {
    states: Vec<BrickState>,
    parents: Vec<Option<GateId>>,
    bricks: Vec<BrickId>,
    roots: Vec<GateId>,
    types: Vec<TypeId>,
    needs: Vec<Vec<BrickId>>,
    installs: Vec<u64>,
    generation: u64,
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
            states: vec![BrickState::Off; gates],
            parents,
            bricks,
            roots,
            types,
            needs: vec![Vec::new(); count],
            installs: vec![0; gates],
            generation: 0,
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
            BrickState::Off | BrickState::On => None,
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
        if let Some(held) = self.states.get_mut(gate.index()) {
            *held = state;
            self.generation = self.generation.wrapping_add(1);
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
        turn(world, root, BrickState::Off);
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
        (BrickState::Off, BrickState::On) => turn_on(world, gate),
        (BrickState::On, to @ (BrickState::Off | BrickState::Faulted(_))) => {
            turn_off(world, gate, to);
        }
        (BrickState::Off, to @ BrickState::Faulted(_))
        | (BrickState::Faulted(_), to @ BrickState::Off) => {
            if let Some(mut states) = world.get_resource_mut::<BrickStates>() {
                states.set(gate, to);
            }
        }
        (BrickState::Off, BrickState::Off)
        | (BrickState::On, BrickState::On)
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
        turn(world, child, BrickState::Off);
    }
    for dependent in dependents {
        let root = world
            .get_resource::<BrickStates>()
            .and_then(|states| states.root(dependent));
        if let Some(root) = root {
            turn(world, root, BrickState::Off);
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
                if !want[index] && state_of(world, gate).is_some_and(|state| state.on()) {
                    turn(world, gate, BrickState::Off);
                }
            }
            for (index, wanted) in want.iter().enumerate() {
                let gate = GateId(index as u32);
                if *wanted && matches!(state_of(world, gate), Some(BrickState::Off)) {
                    turn(world, gate, BrickState::On);
                }
            }
            if generation(world) == before {
                return;
            }
        }
    }

    fn wanted(&mut self, world: &mut World) -> Option<Vec<bool>> {
        let count = self.conditions.len();
        let mut want = vec![false; count];
        for index in 0..count {
            let gate = GateId(index as u32);
            let (brick, parent, faulted, lacking) = {
                let states = world.get_resource::<BrickStates>()?;
                let brick = states.brick(gate)?;
                let parent = states.parent(gate);
                let lacking = parent.is_none()
                    && states.needs(brick).iter().any(|need| !states.brick_on(*need));
                (brick, parent, states.faulted(gate), lacking)
            };
            if faulted || parent.is_some_and(|parent| !want[parent.index()]) {
                continue;
            }
            want[index] = match self.conditions[index].as_mut() {
                None => true,
                Some(condition) => match condition.run_readonly((), world) {
                    Ok(on) => on,
                    Err(_) if lacking => false,
                    Err(error) => {
                        let refused = refused(world, brick, &error);
                        fault(world, brick, refused);
                        false
                    }
                },
            };
        }
        let states = world.get_resource::<BrickStates>()?;
        loop {
            let mut dropped = false;
            for (brick, root) in states.roots.iter().enumerate() {
                let needs = states.needs.get(brick).map_or(&[][..], Vec::as_slice);
                let lacking = needs.iter().any(|need| {
                    states
                        .root(*need)
                        .is_none_or(|need| !want[need.index()])
                });
                if want[root.index()] && lacking {
                    want[root.index()] = false;
                    dropped = true;
                }
            }
            if !dropped {
                break;
            }
        }
        for index in 0..count {
            if let Some(parent) = states.parent(GateId(index as u32))
                && !want[parent.index()]
            {
                want[index] = false;
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
