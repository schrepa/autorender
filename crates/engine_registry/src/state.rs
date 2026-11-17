use bevy_ecs::component::ComponentId;
use bevy_ecs::entity::Entity;
use bevy_ecs::resource::Resource;
use bevy_ecs::world::World;
use engine_place::Board;

use crate::BrickId;
use crate::registry::Registry;

pub type SwitchFn = fn(&World) -> bool;

pub type Despawn = fn(&mut World, &[Entity]);

type Left = fn(&mut World, BrickId);

#[derive(Resource, Default)]
pub struct BrickStates {
    on: Vec<bool>,
    installs: Vec<u64>,
    generation: u64,
}

impl BrickStates {
    pub(crate) fn new(count: usize) -> Self {
        Self {
            on: vec![false; count],
            installs: vec![0; count],
            generation: 0,
        }
    }

    pub fn on(&self, brick: BrickId) -> bool {
        self.on.get(brick.index()).copied().unwrap_or(false)
    }

    pub fn installed(&self, brick: BrickId) -> Option<u64> {
        self.on(brick)
            .then(|| self.installs.get(brick.index()).copied())
            .flatten()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn iter(&self) -> impl Iterator<Item = (BrickId, bool)> + '_ {
        self.on
            .iter()
            .enumerate()
            .map(|(index, &on)| (BrickId(index as u32), on))
    }

    fn set(&mut self, brick: BrickId, on: bool) {
        let Some(held) = self.on.get_mut(brick.index()) else {
            return;
        };
        *held = on;
        self.generation = self.generation.wrapping_add(1);
        if on && let Some(installs) = self.installs.get_mut(brick.index()) {
            *installs = installs.wrapping_add(1);
        }
    }
}

pub struct Switches {
    switches: Box<[Option<SwitchFn>]>,
}

impl Switches {
    pub(crate) fn new(switches: Vec<Option<SwitchFn>>) -> Self {
        Self {
            switches: switches.into_boxed_slice(),
        }
    }

    pub fn run(&mut self, world: &mut World) {
        for (index, switch) in self.switches.iter().enumerate() {
            let brick = BrickId(index as u32);
            let want = switch.is_none_or(|switch| switch(world));
            let on = world
                .get_resource::<BrickStates>()
                .is_some_and(|states| states.on(brick));
            match (on, want) {
                (false, true) => turn_on(world, brick),
                (true, false) => turn_off(world, brick),
                (true, true) | (false, false) => {}
            }
        }
    }
}

fn turn_on(world: &mut World, brick: BrickId) {
    let Some(record) = world
        .get_resource::<Registry>()
        .and_then(|registry| registry.record(brick))
    else {
        return;
    };
    let (package, resources, settings) = (record.package, record.dropped(), record.kept());
    Board::owned_by(package, || {
        for (insert, _) in &resources {
            insert(world);
        }
    });
    for touch in &settings {
        touch(world);
    }
    if let Some(mut states) = world.get_resource_mut::<BrickStates>() {
        states.set(brick, true);
    }
}

fn turn_off(world: &mut World, brick: BrickId) {
    let Some(registry) = world.get_resource::<Registry>() else {
        return;
    };
    let Some(record) = registry.record(brick) else {
        return;
    };
    let (spawned, despawn, resources) = (record.spawned, registry.despawn, record.dropped());
    let joins: Vec<Left> = record.joins().iter().map(|joined| joined.left).collect();
    if let Some(mut states) = world.get_resource_mut::<BrickStates>() {
        states.set(brick, false);
    }
    let id = spawned(world);
    let owned = holders(world, id);
    despawn.unwrap_or(despawn_all)(world, &owned);
    for (_, remove) in &resources {
        remove(world);
    }
    for left in &joins {
        left(world, brick);
    }
}

fn despawn_all(world: &mut World, entities: &[Entity]) {
    for &entity in entities {
        world.despawn(entity);
    }
}

fn holders(world: &World, id: ComponentId) -> Vec<Entity> {
    world
        .archetypes()
        .iter()
        .filter(|archetype| archetype.contains(id))
        .flat_map(|archetype| archetype.entities().iter().map(|held| held.id()))
        .collect()
}
