use bevy_ecs::change_detection::{CheckChangeTicks, Res};
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{BoxedSystem, ReadOnlySystem, RunSystemError, System};
use bevy_ecs::world::World;

use crate::BrickId;

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

#[derive(Resource, Default)]
pub struct Gates {
    own: Vec<bool>,
    on: Vec<bool>,
    parents: Vec<Option<GateId>>,
    bricks: Vec<BrickId>,
    flipped: Vec<(GateId, bool)>,
    generation: u64,
}

impl Gates {
    pub(crate) fn new(parents: Vec<Option<GateId>>, bricks: Vec<BrickId>) -> Self {
        Self {
            own: vec![true; parents.len()],
            on: vec![true; parents.len()],
            parents,
            bricks,
            flipped: Vec::new(),
            generation: 0,
        }
    }

    pub fn on(&self, gate: GateId) -> bool {
        self.on.get(gate.index()).copied().unwrap_or(false)
    }

    pub fn brick(&self, gate: GateId) -> Option<BrickId> {
        self.bricks.get(gate.index()).copied()
    }

    pub fn parent(&self, gate: GateId) -> Option<GateId> {
        self.parents.get(gate.index()).copied().flatten()
    }

    pub fn flipped(&self) -> &[(GateId, bool)] {
        &self.flipped
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn len(&self) -> usize {
        self.on.len()
    }

    pub fn is_empty(&self) -> bool {
        self.on.is_empty()
    }

    pub fn flip(&mut self, flips: &[(GateId, bool)]) {
        self.flipped.clear();
        for &(gate, own) in flips {
            if let Some(slot) = self.own.get_mut(gate.index()) {
                *slot = own;
            }
        }
        for index in 0..self.on.len() {
            let parent = self.parents[index].is_none_or(|parent| self.on[parent.index()]);
            let on = self.own[index] && parent;
            if on != self.on[index] {
                self.on[index] = on;
                self.flipped.push((GateId(index as u32), on));
            }
        }
        if !self.flipped.is_empty() {
            self.generation = self.generation.wrapping_add(1);
        }
    }

    pub fn settle(&mut self) {
        self.flipped.clear();
    }
}

pub fn gate_on(gate: GateId) -> impl FnMut(Res<Gates>) -> bool + Clone + Send + Sync + 'static {
    move |gates: Res<Gates>| gates.on(gate)
}

pub(crate) struct Gate {
    pub(crate) id: GateId,
    pub(crate) brick: &'static str,
    pub(crate) condition: GateFn,
    pub(crate) on: bool,
}

pub(crate) struct Off {
    pub(crate) root: GateId,
    pub(crate) brick: &'static str,
    pub(crate) system: BoxedSystem,
}

pub struct GateDriver {
    gates: Box<[Gate]>,
    offs: Box<[Off]>,
    resets: Box<[(GateId, fn(&mut World))]>,
    refresh: Box<[fn(&mut World)]>,
    scratch: Vec<(GateId, bool)>,
    changed: Vec<(GateId, bool)>,
    faults: Vec<(&'static str, RunSystemError)>,
}

impl GateDriver {
    pub(crate) fn new(
        gates: Vec<Gate>,
        offs: Vec<Off>,
        resets: Vec<(GateId, fn(&mut World))>,
        refresh: Vec<fn(&mut World)>,
    ) -> Self {
        Self {
            scratch: Vec::with_capacity(gates.len()),
            changed: Vec::new(),
            gates: gates.into_boxed_slice(),
            offs: offs.into_boxed_slice(),
            resets: resets.into_boxed_slice(),
            refresh: refresh.into_boxed_slice(),
            faults: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.gates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    pub fn run(&mut self, world: &mut World) {
        self.scratch.clear();
        for gate in self.gates.iter_mut() {
            let on = match gate.condition.run_readonly((), world) {
                Ok(on) => on,
                Err(error) => {
                    self.faults.push((gate.brick, error));
                    false
                }
            };
            if on != gate.on {
                gate.on = on;
                self.scratch.push((gate.id, on));
            }
        }
        if self.scratch.is_empty() {
            if world
                .get_resource::<Gates>()
                .is_some_and(|gates| !gates.flipped().is_empty())
            {
                world.resource_mut::<Gates>().settle();
            }
            return;
        }
        {
            let mut gates = world.resource_mut::<Gates>();
            gates.flip(&self.scratch);
            self.changed.clear();
            self.changed.extend_from_slice(gates.flipped());
        }
        for refresh in self.refresh.iter() {
            refresh(world);
        }
        for &(gate, on) in &self.changed {
            if on {
                continue;
            }
            for &(owner, reset) in self.resets.iter() {
                if owner == gate {
                    reset(world);
                }
            }
            for off in self.offs.iter_mut() {
                if off.root == gate
                    && let Err(error) = off.system.run((), world)
                {
                    self.faults.push((off.brick, error));
                }
            }
        }
    }

    pub fn take_faults(&mut self) -> Vec<(&'static str, RunSystemError)> {
        std::mem::take(&mut self.faults)
    }

    pub fn check_change_ticks(&mut self, check: CheckChangeTicks) {
        for gate in self.gates.iter_mut() {
            gate.condition.check_change_tick(check);
        }
        for off in self.offs.iter_mut() {
            off.system.check_change_tick(check);
        }
    }
}
