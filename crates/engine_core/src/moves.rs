use std::marker::PhantomData;

use bevy_ecs::change_detection::Mut;
use bevy_ecs::component::Component;
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::{Add, Remove};
use bevy_ecs::observer::On;
use bevy_ecs::prelude::{Commands, Query, Res, Without};
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::World;
use engine_registry::Registry;

use crate::{Asked, Brick, GlobalTransforms, Mover, MoverId, Pace, Transform, edit_transform};

pub const fn mover_of<B: Brick>() -> MoverId {
    if B::SOURCE.game() {
        MoverId::Game
    } else {
        MoverId::Brick(B::SOURCE.package())
    }
}

#[derive(SystemParam)]
pub struct Moves<'w, 's, B: Brick> {
    nodes: Query<'w, 's, (Entity, &'static mut Transform, &'static Mover)>,
    still: Query<'w, 's, &'static Transform, Without<Mover>>,
    globals: Res<'w, GlobalTransforms>,
    brick: PhantomData<fn() -> B>,
}

impl<B: Brick> Moves<'_, '_, B> {
    pub const fn id() -> MoverId {
        mover_of::<B>()
    }

    pub fn contains(&self, node: Entity) -> bool {
        self.nodes
            .get(node)
            .is_ok_and(|(_, _, mover)| mover.by == Self::id())
    }

    pub fn pace(&self, node: Entity) -> Option<Pace> {
        let (_, _, mover) = self.nodes.get(node).ok()?;
        (mover.by == Self::id()).then_some(mover.pace)
    }

    pub fn get(&self, node: Entity) -> Option<&Transform> {
        let (_, transform, mover) = self.nodes.get(node).ok()?;
        (mover.by == Self::id()).then_some(transform)
    }

    pub fn local(&self, node: Entity) -> Option<&Transform> {
        match self.nodes.get(node) {
            Ok((_, transform, _)) => Some(transform),
            Err(_) => self.still.get(node).ok(),
        }
    }

    pub fn get_mut(&mut self, node: Entity) -> Option<Mut<'_, Transform>> {
        if !self.globals.active(node) {
            return None;
        }
        let (_, transform, mover) = self.nodes.get_mut(node).ok()?;
        (mover.by == Self::id()).then_some(transform)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Entity, &Transform)> + '_ {
        let globals = &self.globals;
        self.nodes
            .iter()
            .filter(move |(node, _, mover)| mover.by == Self::id() && globals.active(*node))
            .map(|(node, transform, _)| (node, transform))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Entity, Mut<'_, Transform>)> + '_ {
        let Self { nodes, globals, .. } = self;
        let globals = &**globals;
        nodes
            .iter_mut()
            .filter(move |(node, _, mover)| mover.by == Self::id() && globals.active(*node))
            .map(|(node, transform, _)| (node, transform))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Teleport {
    pub node: Entity,
    pub to: Transform,
}

pub fn send_ask<A: Send + Sync + 'static>(world: &mut World, ask: A) {
    if let Some(mut asked) = world.get_resource_mut::<Asked<A>>() {
        asked.ask(ask);
    }
}

pub fn take_asks<A: Send + Sync + 'static>(world: &mut World) -> Vec<A> {
    if world.get_resource::<Asked<A>>().is_none_or(Asked::is_empty) {
        return Vec::new();
    }
    world.resource_mut::<Asked<A>>().take()
}

pub(crate) fn end_tick(world: &mut World) -> bool {
    let teleports = take_asks::<Teleport>(world);
    if teleports.is_empty() {
        return false;
    }
    let fresh: Vec<Entity> = teleports
        .into_iter()
        .filter_map(|Teleport { node, to }| {
            edit_transform(world, node, |transform| *transform = to)
                .ok()
                .map(|()| node)
        })
        .collect();
    if !fresh.is_empty() {
        crate::presentation::make_fresh(world, &fresh);
    }
    true
}

fn tick_paced(mover: Option<&Mover>) -> bool {
    mover.is_some_and(|mover| mover.pace == Pace::Tick)
}

fn frame_teleports(world: &mut World) -> Vec<Teleport> {
    let framed: Vec<Entity> = world
        .get_resource::<Asked<Teleport>>()
        .into_iter()
        .flat_map(Asked::pending)
        .filter(|teleport| !tick_paced(world.get::<Mover>(teleport.node)))
        .map(|teleport| teleport.node)
        .collect();
    if framed.is_empty() {
        return Vec::new();
    }
    world
        .resource_mut::<Asked<Teleport>>()
        .take_where(|teleport| framed.contains(&teleport.node))
}

pub(crate) fn start_frame(world: &mut World) -> bool {
    let teleports = frame_teleports(world);
    if teleports.is_empty() {
        return false;
    }
    let mut fresh = Vec::with_capacity(teleports.len());
    for Teleport { node, to } in teleports {
        if edit_transform(world, node, |transform| *transform = to).is_ok() {
            fresh.push(node);
        }
    }
    crate::presentation::make_fresh(world, &fresh);
    true
}

#[derive(Clone, Copy)]
pub(crate) enum NodeKind {
    Claimed {
        claim: fn(&mut World, Mover),
        mover: Mover,
    },
}

pub(crate) fn claim<C: Component>(world: &mut World, mover: Mover) {
    world.add_observer(move |add: On<Add, C>, mut commands: Commands| {
        let node = add.entity;
        commands.queue(move |world: &mut World| claimed(world, node, mover));
    });
    world.add_observer(move |remove: On<Remove, C>, mut commands: Commands| {
        let node = remove.entity;
        commands.queue(move |world: &mut World| released(world, node, mover));
    });
    world.register_component::<C>();
}

fn claimed(world: &mut World, node: Entity, mover: Mover) {
    if let Ok(mut held) = world.get_entity_mut(node)
        && !held.contains::<Mover>()
    {
        held.insert(mover);
    }
}

fn released(world: &mut World, node: Entity, mover: Mover) {
    if let Ok(mut held) = world.get_entity_mut(node)
        && held.get::<Mover>().is_some_and(|holder| holder.by == mover.by)
    {
        held.remove::<Mover>();
    }
}

pub(crate) fn install_kinds(world: &mut World) {
    let kinds: Vec<NodeKind> = world
        .get_resource::<Registry>()
        .map_or_else(Vec::new, |registry| {
            registry
                .bricks()
                .iter()
                .flat_map(|record| record.entries::<NodeKind>().copied())
                .collect()
        });
    for kind in kinds {
        match kind {
            NodeKind::Claimed { claim, mover } => claim(world, mover),
        }
    }
}
