engine::brick!(body);

mod arena;
mod input;
mod steps;
mod watch;

use engine::math;
use engine::prelude::*;
use engine_core::Asked;
use glam::{Quat, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::actors::{Actor, Alive, Body, Gait, Health, Human, Motion, SpeedFactor, Wish};
use crate::play::events::body::BodyEvents;
use crate::play::numbers::{Numbers, PlayerNumbers};
use crate::play::powers::Push;
use crate::play::round::Round;
use crate::play::view::{Controls, ViewOffsets, Watched};
use crate::play::world::{Arena, Zone};
use crate::rules::Side;
use crate::solids::{self, Moved};

use arena::ArenaFile;
use input::{Latched, NumbersFile, Swallow};

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
struct Tally {
    apex: f32,
    placed: Vec3,
    sweeps: u32,
    triangles: u32,
    travel: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HumanStats {
    pub slot: u32,
    pub position: [f32; 3],
    pub eye: [f32; 3],
    pub velocity: [f32; 3],
    pub speed: f32,
    pub grounded: bool,
    pub crouch: f32,
    pub motion: String,
    pub surface: Option<String>,
    pub yaw: f32,
    pub pitch: f32,
    pub apex: f32,
    pub sweeps: u32,
    pub triangles: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ActorStats {
    pub slot: u32,
    pub character: String,
    pub position: [f32; 3],
    pub grounded: bool,
    pub motion: String,
    pub alive: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ArenaStats {
    pub ready: bool,
    pub triangles: u32,
    pub dropped: u32,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EventStats {
    pub footsteps: u32,
    pub loud_footsteps: u32,
    pub jumps: u32,
    pub landings: u32,
    pub hard_landings: u32,
    pub last_fall_speed: f32,
    pub last_tick: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WatchStats {
    pub slot: Option<u32>,
    pub watch: String,
}

stats! {
    #[section("body")]
    pub struct BodyStats {
        pub events: EventStats,
        pub watched: WatchStats,
        pub human: Option<HumanStats>,
        pub actors: Vec<ActorStats>,
        pub ticks: u64,
        pub arena: ArenaStats,
    }
}

pub struct BodyBrick;

impl Brick for BodyBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<NumbersFile>("numbers")
                    .schema(engine_core::schema_of::<NumbersFile>)
                    .files(NumbersFile::files)
                    .prepare(NumbersFile::prepared),
            )
            .join::<SceneComponents>(
                ComponentKind::of::<Actor>("actor").schema(engine_core::schema_of::<Actor>),
            )
            .join::<SceneComponents>(
                ComponentKind::of::<Human>("human").schema(engine_core::schema_of::<Human>),
            )
            .join::<SceneComponents>(
                ComponentKind::of::<ArenaFile>("arena")
                    .schema(engine_core::schema_of::<ArenaFile>)
                    .check(ArenaFile::checked)
                    .files(ArenaFile::files)
                    .prepare(ArenaFile::prepared),
            )
            .moves::<Actor>(Pace::Tick)
            .resource::<Numbers>()
            .resource::<Latched>()
            .resource::<Arena>()
            .resource::<Swallow>()
            .resource::<BodyEvents>()
            .resource::<EventStats>()
            .resource::<Watched>()
            .settings::<Controls>()
            .stats::<BodyStats>()
            .systems(Schedule::FrameStart, input::adopt_numbers)
            .systems(Schedule::FrameStart, adopt)
            .systems(Schedule::FrameStart, input::capture)
            .systems(Schedule::FrameStart, input::aim)
            .systems(Schedule::FrameStart, input::latch)
            .systems(Schedule::FrameStart, embody)
            .systems(Schedule::FixedSimulation, input::wish)
            .systems(Schedule::FixedSimulation, advance)
            .systems(Schedule::FixedSimulation, watch::watch)
            .systems(Schedule::Presentation, report);
    }
}

fn adopt(files: Query<Ref<ArenaFile>>, mut arena: ResMut<Arena>) {
    for file in &files {
        if file.is_changed()
            && let Some(built) = &file.built
        {
            let (solids, markers) = &**built;
            arena.adopt(solids.clone(), markers.clone());
        }
    }
}

fn eye_height(player: &PlayerNumbers, crouch: f32) -> f32 {
    player.eye_height_m + (player.crouch_eye_height_m - player.eye_height_m) * crouch
}

fn embody(
    actors: Query<(Entity, &Actor), Without<Body>>,
    numbers: Res<Numbers>,
    moves: Moves<BodyBrick>,
    mut spawner: Spawner<BodyBrick>,
) {
    let Some(numbers) = numbers.get() else {
        return;
    };
    let player = &numbers.player;
    for (entity, actor) in &actors {
        let Some(placed) = moves.local(entity) else {
            continue;
        };
        let feet = placed.translation;
        let eye = feet + Vec3::Y * player.eye_height_m;
        spawner.insert(
            entity,
            (
                Body {
                    feet,
                    velocity: Vec3::ZERO,
                    grounded: false,
                    crouched: false,
                    crouch: 0.0,
                    eye,
                    eye_before: eye,
                    motion: Motion::Air,
                    ground: None,
                    landed: None,
                },
                Wish {
                    yaw_radians: actor.yaw_radians,
                    ..Wish::default()
                },
                Health {
                    health: player.health,
                    armour: 0.0,
                },
                Alive,
                ViewOffsets::NONE,
                Tally {
                    apex: feet.y,
                    placed: feet,
                    sweeps: 0,
                    triangles: 0,
                    travel: 0.0,
                },
            ),
        );
    }
}

fn wished_speed(player: &PlayerNumbers, wish: &Wish, crouched: bool, factor: f32) -> f32 {
    let base = player.run_speed_mps * factor;
    if crouched || wish.crouch {
        base * player.crouch_factor
    } else {
        match wish.gait {
            Gait::Walk => base * player.walk_factor,
            Gait::Run => base,
        }
    }
}

fn motion_of(moved: &Moved, wish: &Wish) -> Motion {
    if !moved.body.grounded {
        Motion::Air
    } else if moved.body.crouched {
        Motion::Crouching
    } else if wish.direction == Vec3::ZERO {
        Motion::Standing
    } else {
        match wish.gait {
            Gait::Walk => Motion::Walking,
            Gait::Run => Motion::Running,
        }
    }
}

fn moved_body(player: &PlayerNumbers, body: &Body, wish: &Wish, moved: &Moved, dt: f32) -> Body {
    let target = f32::from(u8::from(moved.body.crouched));
    let ease = dt / player.crouch_seconds;
    let crouch = math::clamp(
        body.crouch + math::clamp(target - body.crouch, -ease, ease),
        0.0,
        1.0,
    );
    let feet = moved.body.position;
    Body {
        feet,
        velocity: moved.body.velocity,
        grounded: moved.body.grounded,
        crouched: moved.body.crouched,
        crouch,
        eye: feet + Vec3::Y * eye_height(player, crouch),
        eye_before: body.eye,
        motion: motion_of(moved, wish),
        ground: moved.ground,
        landed: moved.landed,
    }
}

type Moving<'a> = (
    Entity,
    &'a Actor,
    &'a Wish,
    &'a mut Body,
    &'a mut Tally,
    Option<&'a SpeedFactor>,
    Option<&'a mut Push>,
);

fn held_in_buy_zone(arena: &Arena, round: Option<&Round>, slot: u32, from: Vec3, to: Vec3) -> Vec3 {
    let Some(side) = round
        .filter(|round| round.buy_lock)
        .and_then(|round| round.side_of_slot(slot as usize))
    else {
        return to;
    };
    let zone = match side {
        Side::Attack => Zone::BuyAttack,
        Side::Defend => Zone::BuyDefend,
    };
    if arena.zone(zone, from) && !arena.zone(zone, to) {
        Vec3::new(from.x, to.y, from.z)
    } else {
        to
    }
}

fn pushed(arena: &Arena, player: &PlayerNumbers, body: &solids::Body, shove: Vec3) -> Vec3 {
    let capsule = solids::capsule_of(body, &player.moving);
    match arena.solids().capsule_sweep(capsule, shove) {
        Some(hit) => body.position + shove.normalize_or_zero() * hit.distance,
        None => body.position + shove,
    }
}

fn step_once(
    arena: &Arena,
    player: &PlayerNumbers,
    (wish, body, factor): (&Wish, &Body, f32),
    dt: f32,
) -> Moved {
    solids::step(
        arena.solids(),
        solids::Body {
            position: body.feet,
            velocity: body.velocity,
            grounded: body.grounded,
            crouched: body.crouched,
        },
        solids::Wish {
            direction: wish.direction,
            speed: wished_speed(player, wish, body.crouched, factor),
            jump: wish.jump,
            crouch: wish.crouch,
        },
        &player.moving,
        dt,
    )
}

#[derive(SystemParam)]
struct Setting<'w> {
    time: Res<'w, FixedTime>,
    numbers: Res<'w, Numbers>,
    arena: Res<'w, Arena>,
    round: Option<Res<'w, Round>>,
}

fn advance(
    world: Setting<'_>,
    mut actors: Query<Moving<'_>, With<Alive>>,
    mut moves: Moves<BodyBrick>,
    mut teleports: ResMut<Asked<Teleport>>,
    (mut events, mut counted): (ResMut<BodyEvents>, ResMut<EventStats>),
) {
    events.tick = world.time.tick;
    events.footsteps.clear();
    events.jumps.clear();
    events.landings.clear();
    let Some(numbers) = world.numbers.get() else {
        return;
    };
    let arena = &*world.arena;
    if !arena.ready() {
        return;
    }
    let player = &numbers.player;
    let dt = world.time.delta_seconds;
    for (entity, actor, wish, mut body, mut tally, factor, push) in &mut actors {
        let jumped = body.feet != tally.placed;
        let factor = factor.map_or(1.0, |held| held.0);
        let mut moved = step_once(arena, player, (wish, &body, factor), dt);
        if let Some(mut push) = push
            && push.0 != Vec3::ZERO
        {
            let shove = std::mem::take(&mut push.0);
            moved.body.position = pushed(arena, player, &moved.body, shove);
        }
        moved.body.position = held_in_buy_zone(
            arena,
            world.round.as_deref(),
            actor.slot,
            body.feet,
            moved.body.position,
        );
        let next = moved_body(player, &body, wish, &moved, dt);
        tally.apex = if body.grounded {
            next.feet.y
        } else {
            math::max(tally.apex, next.feet.y)
        };
        tally.sweeps = moved.work.sweeps;
        tally.triangles = moved.work.triangles;
        tally.placed = next.feet;
        steps::record(
            player,
            &steps::Step {
                slot: actor.slot,
                before: &body,
                next: &next,
                moved: &moved,
                wish,
            },
            &mut tally.travel,
            &mut events,
        );
        *body = next;
        let rotation = Quat::from_rotation_y(wish.yaw_radians + std::f32::consts::PI);
        if jumped {
            let scale = moves.local(entity).map_or(Vec3::ONE, |placed| placed.scale);
            teleports.ask(Teleport {
                node: entity,
                to: Transform {
                    translation: next.feet,
                    rotation,
                    scale,
                },
            });
        } else if let Some(mut transform) = moves.get_mut(entity) {
            transform.translation = next.feet;
            transform.rotation = rotation;
        }
    }
    count(&events, &mut counted);
}

fn count(events: &BodyEvents, counted: &mut EventStats) {
    let loud = events.footsteps.iter().filter(|step| step.loud).count();
    let hard = events
        .landings
        .iter()
        .filter(|landing| landing.hard)
        .count();
    counted.footsteps += events.footsteps.len() as u32;
    counted.loud_footsteps += loud as u32;
    counted.jumps += events.jumps.len() as u32;
    counted.landings += events.landings.len() as u32;
    counted.hard_landings += hard as u32;
    if let Some(landing) = events.landings.last() {
        counted.last_fall_speed = landing.fall_speed;
    }
    if !events.footsteps.is_empty() || !events.jumps.is_empty() || !events.landings.is_empty() {
        counted.last_tick = events.tick;
    }
}

fn named<T: Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default()
}

type Shown<'a> = (
    &'a Actor,
    &'a Body,
    &'a Wish,
    &'a Tally,
    Has<Alive>,
    Has<Human>,
);

fn report(
    time: Res<FixedTime>,
    arena: Res<Arena>,
    (counted, watched): (Res<EventStats>, Res<Watched>),
    actors: Query<Shown<'_>>,
    mut stats: ResMut<BodyStats>,
) {
    let mut listed: Vec<_> = actors.iter().collect();
    listed.sort_by_key(|(actor, ..)| actor.slot);
    let human =
        listed
            .iter()
            .find(|(.., human)| *human)
            .map(|&(actor, body, wish, tally, _, _)| HumanStats {
                slot: actor.slot,
                position: body.feet.to_array(),
                eye: body.eye.to_array(),
                velocity: body.velocity.to_array(),
                speed: body.velocity.with_y(0.0).length(),
                grounded: body.grounded,
                crouch: body.crouch,
                motion: named(&body.motion),
                surface: body.ground.as_ref().map(named),
                yaw: wish.yaw_radians,
                pitch: wish.pitch_radians,
                apex: tally.apex,
                sweeps: tally.sweeps,
                triangles: tally.triangles,
            });
    let dropped = arena.dropped();
    let next = BodyStats {
        events: counted.clone(),
        watched: WatchStats {
            slot: watched.slot,
            watch: named(&watched.watch),
        },
        human,
        actors: listed
            .iter()
            .map(|&(actor, body, _, _, alive, _)| ActorStats {
                slot: actor.slot,
                character: actor.character.clone(),
                position: body.feet.to_array(),
                grounded: body.grounded,
                motion: named(&body.motion),
                alive,
            })
            .collect(),
        ticks: time.tick,
        arena: ArenaStats {
            ready: arena.ready(),
            triangles: arena.solids().triangle_count() as u32,
            dropped: dropped.degenerate + dropped.unlabelled + dropped.out_of_range,
        },
    };
    stats.set_if_neq(next);
}
