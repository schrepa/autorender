use std::any::TypeId;
use std::marker::PhantomData;

use bevy_ecs::change_detection::Mut;
use bevy_ecs::component::{Component, ComponentId};
use bevy_ecs::entity::Entity;
use bevy_ecs::lifecycle::{Add, Remove};
use bevy_ecs::observer::On;
use bevy_ecs::prelude::{Commands, Query, Res, ResMut, With, Without};
use bevy_ecs::resource::Resource;
use bevy_ecs::system::{SystemParam, SystemParamItem};
use bevy_ecs::world::World;
use engine_registry::{MoverRule, Registry, SceneEntity};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::hierarchy::tracking::Invalidation;
use crate::{
    ApplyAt, Ask, Asked, Brick, Error, ErrorCode, Expected, FixedTime, FrameTime,
    GlobalTransforms, Moment, Mover, MoverId, Operation, Pace, SceneIds, Transform,
    edit_transform,
};

mod key {
    #[derive(bevy_ecs::resource::Resource, Default)]
    pub struct Key;
}

pub(crate) use key::Key;

pub const fn mover_of<B: Brick>() -> MoverId {
    if B::SOURCE.game() {
        MoverId::Game
    } else {
        MoverId::Brick(B::SOURCE.package())
    }
}

pub(crate) fn rule() -> MoverRule {
    MoverRule {
        transform: TypeId::of::<Transform>(),
        mover: TypeId::of::<Mover>(),
        key: TypeId::of::<Key>(),
    }
}

#[derive(SystemParam)]
pub struct Moves<'w, 's, B: Brick> {
    nodes: Query<'w, 's, (Entity, &'static mut Transform, &'static Mover)>,
    still: Query<'w, 's, &'static Transform, Without<Mover>>,
    globals: Res<'w, GlobalTransforms>,
    _key: Res<'w, Key>,
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

impl Ask for Teleport {
    const NAME: &'static str = "teleport";
    const AT: ApplyAt = ApplyAt::TickEnd;

    fn node(&self) -> Option<Entity> {
        Some(self.node)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandOver {
    pub node: Entity,
    pub to: Mover,
}

impl Ask for HandOver {
    const NAME: &'static str = "hand_over";
    const AT: ApplyAt = ApplyAt::TickEnd;

    fn node(&self) -> Option<Entity> {
        Some(self.node)
    }
}

#[derive(SystemParam)]
pub struct Asks<'w, 's, A: Ask> {
    commands: Commands<'w, 's>,
    ask: PhantomData<fn() -> A>,
}

impl<A: Ask> Asks<'_, '_, A> {
    pub fn send(&mut self, ask: A) {
        self.commands
            .queue(move |world: &mut World| send_ask(world, ask));
    }
}

#[derive(SystemParam)]
pub struct Applies<'w, A: Ask> {
    asked: ResMut<'w, Asked<A>>,
    fixed: Res<'w, FixedTime>,
    frame: Res<'w, FrameTime>,
}

impl<A: Ask> Applies<'_, A> {
    pub fn take(&mut self) -> Vec<A> {
        if self.asked.is_empty() {
            return Vec::new();
        }
        let now = Moment {
            tick: self.fixed.tick,
            frame: self.frame.frame,
        };
        self.asked.take(now)
    }
}

pub fn send_ask<A: Ask>(world: &mut World, ask: A) {
    let at = moment(world);
    if let Some(mut asked) = world.get_resource_mut::<Asked<A>>() {
        asked.ask(ask, at);
    }
}

pub fn take_asks<A: Ask>(world: &mut World) -> Vec<A> {
    if world.get_resource::<Asked<A>>().is_none_or(Asked::is_empty) {
        return Vec::new();
    }
    let now = moment(world);
    world.resource_mut::<Asked<A>>().take(now)
}

pub(crate) fn moment(world: &World) -> Moment {
    Moment {
        tick: world.get_resource::<FixedTime>().map_or(0, |time| time.tick),
        frame: world.get_resource::<FrameTime>().map_or(0, |time| time.frame),
    }
}

fn paced_handovers(world: &mut World, pace: Pace) -> Vec<HandOver> {
    if world.get_resource::<Asked<HandOver>>().is_none_or(Asked::is_empty) {
        return Vec::new();
    }
    let now = moment(world);
    world
        .resource_mut::<Asked<HandOver>>()
        .take_where(now, |handover| handover.to.pace == pace)
}

pub(crate) fn end_tick(world: &mut World) -> bool {
    let handovers = paced_handovers(world, Pace::Tick);
    let teleports = take_asks::<Teleport>(world);
    if handovers.is_empty() && teleports.is_empty() {
        return false;
    }
    for HandOver { node, to } in handovers {
        if let Ok(mut held) = world.get_entity_mut(node) {
            held.insert(to);
        }
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
    let now = moment(world);
    world
        .resource_mut::<Asked<Teleport>>()
        .take_where(now, |teleport| framed.contains(&teleport.node))
}

pub(crate) fn start_frame(world: &mut World) -> bool {
    let handovers = paced_handovers(world, Pace::Frame);
    let teleports = frame_teleports(world);
    if handovers.is_empty() && teleports.is_empty() {
        return false;
    }
    let mut fresh = Vec::with_capacity(handovers.len() + teleports.len());
    for HandOver { node, to } in handovers {
        if let Ok(mut held) = world.get_entity_mut(node) {
            held.insert(to);
            fresh.push(node);
        }
    }
    for Teleport { node, to } in teleports {
        if edit_transform(world, node, |transform| *transform = to).is_ok() {
            fresh.push(node);
        }
    }
    crate::presentation::make_fresh(world, &fresh);
    true
}

#[derive(Resource, Default)]
pub(crate) struct MovingKinds(pub(crate) Vec<ComponentId>);

#[derive(Clone, Copy)]
pub(crate) enum NodeKind {
    Claimed { claim: fn(&mut World, Mover), mover: Mover },
    Moving { track: fn(&mut World) -> ComponentId },
}

pub(crate) fn claim<C: Component>(world: &mut World, mover: Mover) {
    world.add_observer(
        move |add: On<Add, C>, held: Query<(), With<Mover>>, mut commands: Commands| {
            if !held.contains(add.entity) {
                commands.entity(add.entity).try_insert_if_new(mover);
            }
        },
    );
}

pub(crate) fn moving<C: Component>(world: &mut World) -> ComponentId {
    world.add_observer(|add: On<Add, C>, mut held: ResMut<Invalidation>| {
        held.touched.push(add.entity);
    });
    world.add_observer(|remove: On<Remove, C>, mut held: ResMut<Invalidation>| {
        held.touched.push(remove.entity);
    });
    world.register_component::<C>()
}

pub(crate) fn install_kinds(world: &mut World) {
    let kinds: Vec<NodeKind> = world.get_resource::<Registry>().map_or_else(Vec::new, |registry| {
        registry
            .bricks()
            .iter()
            .flat_map(|record| record.entries::<NodeKind>().copied())
            .collect()
    });
    let mut moving = Vec::new();
    for kind in kinds {
        match kind {
            NodeKind::Claimed { claim, mover } => claim(world, mover),
            NodeKind::Moving { track } => moving.push(track(world)),
        }
    }
    world.insert_resource(MovingKinds(moving));
}

fn named(world: &World, node: Entity) -> Value {
    match world.get::<SceneEntity>(node) {
        Some(authored) => json!({ "node": node.to_bits(), "id": authored.authored_id, "name": authored.name }),
        None => json!({ "node": node.to_bits() }),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MovesRequest {}

pub(crate) struct MovesState;

impl Operation for MovesState {
    const NAME: &'static str = "moves";
    const CONTROL: bool = false;
    type Params = &'static World;
    type Request = MovesRequest;

    fn run(world: SystemParamItem<'_, '_, Self::Params>, _: MovesRequest) -> Result<Value, Error> {
        let asks: Vec<Value> = world
            .get_resource::<Registry>()
            .map_or_else(Vec::new, |registry| {
                registry
                    .catalog()
                    .iter()
                    .filter_map(|place| place.view(world))
                    .collect()
            });
        let moved_static: Vec<Value> = world
            .get_resource::<GlobalTransforms>()
            .map_or_else(Vec::new, |globals| {
                globals
                    .moved_static()
                    .map(|node| named(world, node))
                    .collect()
            });
        Ok(json!({ "asks": asks, "moved_static": moved_static }))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TeleportControl {
    entity: String,
    translation: [f32; 3],
}

impl Operation for TeleportControl {
    const NAME: &'static str = Teleport::NAME;
    type Params = (
        SceneIds<'static, 'static>,
        Query<'static, 'static, (&'static Transform, Option<&'static Mover>)>,
        Asks<'static, 'static, Teleport>,
    );
    type Request = Self;

    fn run(
        (ids, nodes, mut asks): SystemParamItem<'_, '_, Self::Params>,
        request: Self,
    ) -> Result<Value, Error> {
        let node = ids
            .entity(&request.entity)
            .ok_or_else(|| Error::new(ErrorCode::UnknownEntity, "$.entity").got(&request.entity))?;
        let (held, mover) = nodes.get(node).map_err(|_| {
            Error::new(ErrorCode::MissingComponent, "$.entity.transform")
                .expected(Expected::Present)
                .got(node.to_bits())
        })?;
        asks.send(Teleport {
            node,
            to: Transform {
                translation: glam::Vec3::from_array(request.translation),
                ..*held
            },
        });
        let at = if tick_paced(mover) {
            ApplyAt::TickEnd
        } else {
            ApplyAt::FrameStart
        };
        Ok(json!({ "at": at }))
    }
}
