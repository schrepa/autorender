use engine::prelude::*;
use glam::Vec3;

use super::{ArmsBrick, Book, Setup, drop_weapon, ticks_per_second};
use crate::names::{HEAD, HIP, SPINE};
use crate::play::actors::{Actor, Alive, Body, Equip, Health, SpeedFactor, Wish};
use crate::play::arms::{Arms, Drops, Hand, HandAction, Held};
use crate::play::events::arms::{
    ArmsEvents, DeathEvent, DropEvent, Handling, HandlingEvent, HitEvent, ShotEvent, Target, Weapon,
};
use crate::play::figures::{FigureJoints, LEFT_LEG, RIGHT_LEG};
use crate::play::powers::{EffectHit, Powers};
use crate::play::round::Round;
use crate::play::view::ViewOffsets;
use crate::play::world::Arena;
use crate::rules::{Device, Team};
use crate::solids::{Hit, Kind, Solid};
use crate::weapons::{
    Aim, ArmsNumbers, Blocker, BodyPart, CharacterId, Fired, Footing, GunKind, HandMotion,
    HitJoints, HitShapes, LoadoutInput, LoadoutTick, Pace, Slot, Struck, Vitals, WeaponNumbers,
    forward, gun_damage, hit_shapes, joints_from_body, melee, melee_damage, stance_of, ticks,
    trace, wound,
};

struct Known(Option<f32>);

impl Blocker for Known {
    fn first_hit(&self, _origin: Vec3, _direction: Vec3, max: f32) -> Option<f32> {
        self.0.filter(|distance| *distance <= max)
    }
}

#[derive(Clone, Copy)]
struct Living {
    slot: u32,
    team: Team,
    shapes: Option<HitShapes>,
}

#[derive(Clone, Copy)]
struct Pending {
    shooter: u32,
    victim: u32,
    weapon: Weapon,
    part: BodyPart,
    distance: f32,
    damage: f32,
    tag_slow: f32,
}

type Acting<'a> = (
    Entity,
    &'a Actor,
    &'a Wish,
    &'a Body,
    &'a mut Arms,
    &'a mut Health,
    &'a mut ViewOffsets,
    &'a mut SpeedFactor,
    Has<Alive>,
    Option<&'a FigureJoints>,
);

#[expect(
    clippy::too_many_arguments,
    reason = "the powers door joins the arms step's world facts"
)]
pub(super) fn act(
    time: Res<FixedTime>,
    setup: Res<Setup>,
    arena: Res<Arena>,
    round: Option<Res<Round>>,
    mut powers: Option<ResMut<Powers>>,
    mut events: ResMut<ArmsEvents>,
    mut book: ResMut<Book>,
    mut drops: ResMut<Drops>,
    mut actors: Query<Acting<'_>>,
    mut spawner: Spawner<ArmsBrick>,
) {
    events.tick = time.tick;
    events.shots.clear();
    events.hits.clear();
    events.deaths.clear();
    events.handling.clear();
    events.pickups.clear();
    events.drops.clear();
    let Some(numbers) = setup.numbers.as_ref() else {
        return;
    };
    let Some(tps) = ticks_per_second(&time) else {
        return;
    };
    let mut order: Vec<(u32, Entity)> = actors
        .iter()
        .map(|(entity, actor, ..)| (actor.slot, entity))
        .collect();
    order.sort_unstable();
    let living: Vec<Living> = order
        .iter()
        .filter_map(|(_, entity)| {
            let (_, actor, wish, body, .., alive, posed) = actors.get(*entity).ok()?;
            alive.then(|| Living {
                slot: actor.slot,
                team: actor.team,
                shapes: shapes_of(numbers, actor, body, wish.yaw_radians, posed),
            })
        })
        .collect();
    let mut pending = Vec::new();
    for (_, entity) in &order {
        let Ok((_, actor, wish, body, mut arms, _, mut offsets, mut speed, alive, _)) =
            actors.get_mut(*entity)
        else {
            continue;
        };
        if !alive {
            continue;
        }
        let pace = pace_of(powers.as_deref(), actor.slot);
        let holding = powers
            .as_deref()
            .is_some_and(|powers| powers.held(actor.slot).is_some());
        let role = role_of(round.as_deref(), actor.slot);
        let ticked = Turn {
            slot: actor.slot,
            numbers,
            tps,
            role,
            pace,
            holding,
        }
        .take(wish, body, &mut arms, &mut drops, &mut events);
        let recoil = arms.loadout.recoil();
        offsets.recoil_pitch_radians = recoil.pitch_radians;
        offsets.recoil_yaw_radians = recoil.yaw_radians;
        offsets.zoom = arms.loadout.view_zoom(&numbers.weapons);
        let applied = if arms.busy {
            numbers.combat.device_hand.busy_run_speed_factor
        } else {
            arms.run_factor
        };
        speed.0 = applied * arms.tag_factor();
        handled(&mut events, actor.slot, &arms, &ticked);
        let Some(fired) = ticked.fired else {
            continue;
        };
        let targets: Vec<HitShapes> = living
            .iter()
            .filter(|other| {
                other.slot != actor.slot
                    && (numbers.combat.friendly_fire || other.team != actor.team)
            })
            .filter_map(|other| other.shapes)
            .collect();
        let shooter = Shooter {
            slot: actor.slot,
            numbers,
            arena: &arena,
            targets: &targets,
        };
        let (shot, wounding) = shooter.resolve(fired);
        if let Some(powers) = powers.as_deref_mut()
            && let Some(hit) = struck_effect(powers, &arena, numbers, &shot)
        {
            powers.struck.push(hit);
        }
        events.shots.push(shot);
        pending.extend(wounding);
    }
    book.shots += events.shots.len() as u64;
    if let Some(powers) = powers.as_deref() {
        for wounding in &mut pending {
            wounding.damage *= powers.damage_taken(wounding.victim);
        }
    }
    let wounds = Wounds {
        weapons: &numbers.weapons,
        tps,
        armor_absorb: numbers.economy.armor_absorb,
        tick: time.tick,
        window: u64::from(ticks(numbers.combat.assist_seconds, tps)),
        tag_ticks: ticks(numbers.combat.tag_seconds, tps),
    };
    wounds.apply(
        &pending,
        &mut actors,
        &mut events,
        &mut book,
        &mut drops,
        &mut spawner,
    );
}

struct Turn<'a> {
    slot: u32,
    numbers: &'a ArmsNumbers,
    tps: u32,
    role: Role,
    pace: Pace,
    holding: bool,
}

impl Turn<'_> {
    fn take(
        &self,
        wish: &Wish,
        body: &Body,
        arms: &mut Arms,
        drops: &mut Drops,
        events: &mut ArmsEvents,
    ) -> LoadoutTick {
        let (numbers, tps) = (self.numbers, self.tps);
        let mut redrawn = device_hand(numbers, tps, wish, self.role, arms);
        let in_hand = arms.device_out || arms.busy;
        if wish.drop
            && !in_hand
            && let Some(gun) = arms.loadout.drop_active(&numbers.weapons, tps)
        {
            redrawn = true;
            drops.push(body.feet, &gun);
            events.drops.push(DropEvent {
                slot: self.slot,
                weapon: gun.kind(),
            });
        }
        let blocked = Blocked {
            holding: self.holding || in_hand,
            device: in_hand,
            busy: arms.busy,
        };
        let mut ticked = advance(numbers, tps, (self.pace, blocked), wish, body, arms);
        ticked.switched |= redrawn;
        arms.hand = hand_of(numbers, tps, self.role, arms);
        arms.run_factor = if in_hand {
            numbers.combat.device_hand.run_speed_factor
        } else {
            arms.loadout.speed_factor(&numbers.weapons)
        };
        ticked
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Free,
    Carrying,
    Planting { progress: u32, needed: u32 },
    Defusing { progress: u32, needed: u32 },
}

#[derive(Clone, Copy)]
struct Blocked {
    holding: bool,
    device: bool,
    busy: bool,
}

fn role_of(round: Option<&Round>, slot: u32) -> Role {
    let Some(view) = round.and_then(Round::view) else {
        return Role::Free;
    };
    let slot = slot as usize;
    match *view.device {
        Device::Carried { player, planting } if player == slot => match planting {
            Some(planting) => Role::Planting {
                progress: planting.progress,
                needed: planting.needed,
            },
            None => Role::Carrying,
        },
        Device::Planted { defusing, .. } if defusing.player == Some(slot) => Role::Defusing {
            progress: defusing.progress,
            needed: defusing.needed,
        },
        Device::Carried { .. }
        | Device::Dropped { .. }
        | Device::Planted { .. }
        | Device::Exploded { .. }
        | Device::Defused { .. } => Role::Free,
    }
}

fn device_hand(numbers: &ArmsNumbers, tps: u32, wish: &Wish, role: Role, arms: &mut Arms) -> bool {
    let was_in_hand = arms.device_out || arms.busy;
    let carrying = matches!(role, Role::Carrying | Role::Planting { .. });
    arms.busy = matches!(role, Role::Planting { .. } | Role::Defusing { .. });
    let asked = wish.equip.map(slot_of);
    if !carrying {
        arms.device_out = false;
    } else if asked == Some(None) && !arms.device_out {
        arms.device_out = true;
        arms.device_equip_left = ticks(numbers.combat.device_hand.equip_seconds, tps);
    } else if matches!(asked, Some(Some(_))) && !arms.busy {
        arms.device_out = false;
    } else {
        arms.device_equip_left = arms.device_equip_left.saturating_sub(1);
    }
    if !arms.device_out {
        arms.device_equip_left = 0;
    }
    let in_hand = arms.device_out || arms.busy;
    let switching = asked
        .flatten()
        .is_some_and(|slot| slot != arms.loadout.active);
    if !was_in_hand && in_hand {
        arms.loadout.draw(&numbers.weapons, tps);
    }
    if was_in_hand && !in_hand && !switching {
        arms.loadout.draw(&numbers.weapons, tps);
        return true;
    }
    false
}

fn hand_of(numbers: &ArmsNumbers, tps: u32, role: Role, arms: &Arms) -> Hand {
    let share = |progress: u32, needed: u32| progress as f32 / needed.max(1) as f32;
    let seconds = |needed: u32| needed as f32 / tps.max(1) as f32;
    match role {
        Role::Planting { progress, needed } => {
            return Hand {
                held: Held::Device,
                action: Some(HandAction::Plant),
                progress: share(progress, needed),
                seconds: seconds(needed),
            };
        }
        Role::Defusing { progress, needed } => {
            return Hand {
                held: Held::Device,
                action: Some(HandAction::Defuse),
                progress: share(progress, needed),
                seconds: seconds(needed),
            };
        }
        Role::Free | Role::Carrying => {}
    }
    if arms.device_out {
        let total = ticks(numbers.combat.device_hand.equip_seconds, tps);
        return Hand {
            held: Held::Device,
            action: (arms.device_equip_left > 0).then_some(HandAction::Equip),
            progress: 1.0 - share(arms.device_equip_left, total),
            seconds: seconds(total),
        };
    }
    let held = arms
        .loadout
        .active_gun()
        .map_or(Held::Melee, |gun| Held::Gun(gun.kind()));
    let moving = arms.loadout.hand_moving(&numbers.weapons, tps);
    Hand {
        held,
        action: moving.map(|moving| match moving.motion {
            HandMotion::Equip => HandAction::Equip,
            HandMotion::Swing { heavy } => HandAction::Swing { heavy },
        }),
        progress: moving.map_or(0.0, |moving| moving.progress),
        seconds: moving.map_or(0.0, |moving| moving.seconds),
    }
}

fn advance(
    numbers: &ArmsNumbers,
    tps: u32,
    (pace, blocked): (Pace, Blocked),
    wish: &Wish,
    body: &Body,
    arms: &mut Arms,
) -> LoadoutTick {
    arms.landing_ticks = if body.landed.is_some() {
        ticks(numbers.combat.land_seconds, tps)
    } else {
        arms.landing_ticks.saturating_sub(1)
    };
    arms.tag_ticks = arms.tag_ticks.saturating_sub(1);
    let footing = Footing {
        grounded: body.grounded,
        crouched: body.crouched,
        speed_mps: glam::Vec2::new(body.velocity.x, body.velocity.z).length(),
        landing: arms.landing_ticks > 0,
    };
    arms.stance = stance_of(&numbers.combat, numbers.player.run_speed_mps, footing);
    let holding = blocked.holding;
    let input = LoadoutInput {
        fire_held: !holding && (wish.fire_held || wish.fire_pressed),
        reload_pressed: !blocked.device && wish.reload,
        scope_pressed: !holding && wish.scope,
        switch_to: wish.equip.and_then(slot_of).filter(|_| !blocked.busy),
        pace,
    };
    let aim = crate::weapons::Aim {
        eye: body.eye,
        yaw_radians: wish.yaw_radians,
        pitch_radians: wish.pitch_radians,
    };
    let ticked = arms
        .loadout
        .tick(&numbers.weapons, tps, input, aim, arms.stance);
    arms.kick_pitch_radians = ticked.kick.pitch_radians;
    arms.kick_yaw_radians = ticked.kick.yaw_radians;
    ticked
}

fn handled(events: &mut ArmsEvents, slot: u32, arms: &Arms, ticked: &LoadoutTick) {
    let weapon = arms
        .loadout
        .active_gun()
        .map_or(Weapon::Melee, |gun| Weapon::Gun(gun.kind()));
    let flags = [
        (ticked.switched, Handling::Equip),
        (ticked.active.reload_started, Handling::ReloadStarted),
        (ticked.active.reload_done, Handling::ReloadDone),
        (ticked.active.dry_fire, Handling::DryFire),
    ];
    events.handling.extend(
        flags
            .into_iter()
            .filter(|(raised, _)| *raised)
            .map(|(_, kind)| HandlingEvent { slot, weapon, kind }),
    );
}

fn struck_effect(
    powers: &Powers,
    arena: &Arena,
    numbers: &ArmsNumbers,
    shot: &ShotEvent,
) -> Option<EffectHit> {
    let Weapon::Gun(gun) = shot.weapon else {
        return None;
    };
    if let Target::Actor { .. } = shot.target {
        return None;
    }
    let gun_numbers = numbers.weapons.gun(gun);
    let reach = (shot.end - shot.origin).length();
    let (effect, distance) = match powers
        .effects
        .shootable_ray(shot.origin, shot.direction, reach)
    {
        Some(hit) => (hit.effect, hit.distance),
        None => {
            let hit = arena.ray(shot.origin, shot.direction, gun_numbers.range_m())?;
            let Solid::Block(id) = hit.solid else {
                return None;
            };
            (powers.effects.effect_of_solid(id)?, hit.distance)
        }
    };
    Some(EffectHit {
        effect,
        shooter: shot.shooter,
        damage: gun_damage(gun_numbers, BodyPart::Body, distance),
    })
}

fn pace_of(powers: Option<&Powers>, slot: u32) -> Pace {
    powers.map_or(Pace::SAME, |powers| {
        let modifiers = powers.modifiers(slot);
        Pace {
            fire_rate: modifiers.fire_rate,
            reload: modifiers.reload,
            spread: modifiers.spread,
        }
    })
}

fn slot_of(equip: Equip) -> Option<Slot> {
    match equip {
        Equip::Primary => Some(Slot::Primary),
        Equip::Sidearm => Some(Slot::Sidearm),
        Equip::Melee => Some(Slot::Melee),
        Equip::Device => None,
    }
}

fn shapes_of(
    numbers: &ArmsNumbers,
    actor: &Actor,
    body: &Body,
    yaw_radians: f32,
    posed: Option<&FigureJoints>,
) -> Option<HitShapes> {
    let shapes = &numbers.hit_shapes;
    let head_radius = *shapes.head_radius_m.get(&actor.character)?;
    let height_share = (body.eye.y - body.feet.y) / numbers.player.eye_height_m;
    let (stood, facing) =
        joints_from_body(&shapes.stand_joints_m, body.feet, yaw_radians, height_share);
    let joints = posed.and_then(posed_joints).unwrap_or(stood);
    Some(hit_shapes(
        shapes,
        CharacterId(actor.slot),
        head_radius,
        1.0,
        joints,
        facing,
    ))
}

fn posed_joints(posed: &FigureJoints) -> Option<HitJoints> {
    let at = |name: &str| {
        let index = posed.names.iter().position(|asked| asked == name)?;
        posed
            .joints
            .get(index)
            .filter(|joint| joint.found)
            .map(|joint| joint.position)
    };
    Some(HitJoints {
        head: at(HEAD)?,
        spine: at(SPINE)?,
        hip: at(HIP)?,
        feet: [at(LEFT_LEG[2])?, at(RIGHT_LEG[2])?],
    })
}

struct Shooter<'a> {
    slot: u32,
    numbers: &'a ArmsNumbers,
    arena: &'a Arena,
    targets: &'a [HitShapes],
}

struct Through {
    entry: Vec3,
    kind: Kind,
    kept: f32,
    wall: Option<Hit>,
}

impl Shooter<'_> {
    fn resolve(&self, fired: Fired) -> (ShotEvent, Option<Pending>) {
        match fired {
            Fired::Shot { gun, shot } => self.shoot(gun, shot.origin, shot.direction),
            Fired::Swing { aim, heavy } => self.swing(aim, heavy),
        }
    }

    fn shoot(&self, gun: GunKind, origin: Vec3, direction: Vec3) -> (ShotEvent, Option<Pending>) {
        let numbers = self.numbers.weapons.gun(gun);
        let range = numbers.range_m();
        let weapon = Weapon::Gun(gun);
        let ray = (origin, direction);
        let wall = self.arena.ray(origin, direction, range);
        let shooter = Some(CharacterId(self.slot));
        let blocker = Known(wall.map(|hit| hit.distance));
        let struck = trace(origin, direction, range, shooter, &blocker, self.targets);
        if let Struck::Character {
            character,
            part,
            distance,
        } = struck
        {
            let damage = gun_damage(numbers, part, distance);
            let pending = self.pending(character, weapon, part, distance, damage, numbers.tag_slow);
            let event = shot_event(self.slot, weapon, ray, range, struck, wall);
            return (event, Some(pending));
        }
        let through = wall.and_then(|hit| self.pierce(gun, ray, range, hit));
        let Some(through) = through else {
            let event = shot_event(self.slot, weapon, ray, range, struck, wall);
            return (event, None);
        };
        let blocker = Known(through.wall.map(|hit| hit.distance));
        let beyond = trace(origin, direction, range, shooter, &blocker, self.targets);
        let combat = &self.numbers.combat;
        let pending = match beyond {
            Struck::Character {
                character,
                part,
                distance,
            } => Some(self.pending(
                character,
                weapon,
                part,
                distance,
                gun_damage(numbers, part, distance) * through.kept,
                combat.pierce_tag_slow,
            )),
            Struck::World { .. } | Struck::Nothing => None,
        };
        let mut event = shot_event(self.slot, weapon, ray, range, beyond, through.wall);
        event.pierced = Some(through.entry);
        event.pierced_kind = Some(through.kind);
        (event, pending)
    }

    fn pierce(
        &self,
        gun: GunKind,
        (origin, direction): (Vec3, Vec3),
        range: f32,
        wall: Hit,
    ) -> Option<Through> {
        let table = self
            .numbers
            .combat
            .surface_by(|name| Kind::named(name) == wall.kind)?;
        let reach = table.pierce_m.of(gun);
        let probe = wall.distance + reach;
        if reach <= 0.0 || probe >= range {
            return None;
        }
        let beyond = origin + direction * probe;
        let skin = (probe + 1.0) * 16.0 * f32::EPSILON;
        self.arena.ray(beyond, -direction, reach - skin)?;
        let onward = self
            .arena
            .ray(beyond, direction, range - probe)
            .map(|hit| Hit {
                distance: hit.distance + probe,
                ..hit
            });
        Some(Through {
            entry: origin + direction * wall.distance,
            kind: wall.kind,
            kept: table.damage_kept,
            wall: onward,
        })
    }

    fn swing(&self, aim: Aim, heavy: bool) -> (ShotEvent, Option<Pending>) {
        let melee_numbers = &self.numbers.weapons.melee;
        let direction = forward(aim.yaw_radians, aim.pitch_radians);
        let range = melee_numbers.range_m;
        let wall = self.arena.ray(aim.eye, direction, range);
        let blocker = Known(wall.map(|hit| hit.distance));
        let hit = melee(
            melee_numbers,
            aim,
            CharacterId(self.slot),
            &blocker,
            self.targets,
        );
        let struck = match hit {
            Some(hit) => Struck::Character {
                character: hit.character,
                part: hit.part,
                distance: hit.distance,
            },
            None => trace(aim.eye, direction, range, None, &blocker, &[]),
        };
        let pending = hit.map(|hit| {
            self.pending(
                hit.character,
                Weapon::Melee,
                hit.part,
                hit.distance,
                melee_damage(melee_numbers, &hit, heavy),
                0.0,
            )
        });
        let ray = (aim.eye, direction);
        let event = shot_event(self.slot, Weapon::Melee, ray, range, struck, wall);
        (event, pending)
    }

    fn pending(
        &self,
        victim: CharacterId,
        weapon: Weapon,
        part: BodyPart,
        distance: f32,
        damage: f32,
        tag_slow: f32,
    ) -> Pending {
        Pending {
            shooter: self.slot,
            victim: victim.0,
            weapon,
            part,
            distance,
            damage,
            tag_slow,
        }
    }
}

fn shot_event(
    shooter: u32,
    weapon: Weapon,
    (origin, direction): (Vec3, Vec3),
    range: f32,
    struck: Struck,
    wall: Option<Hit>,
) -> ShotEvent {
    let kind = match (struck, wall) {
        (Struck::Character { .. }, _) | (_, None) => None,
        (Struck::World { .. } | Struck::Nothing, Some(hit)) => Some(hit.kind),
    };
    let (distance, target, normal) = match (struck, wall) {
        (
            Struck::Character {
                character,
                part,
                distance,
            },
            _,
        ) => (
            distance,
            Target::Actor {
                slot: character.0,
                part,
            },
            -direction,
        ),
        (Struck::World { .. } | Struck::Nothing, Some(hit)) => (
            hit.distance,
            Target::World {
                surface: hit.surface,
            },
            hit.normal,
        ),
        (Struck::World { .. } | Struck::Nothing, None) => (range, Target::Nothing, -direction),
    };
    ShotEvent {
        shooter,
        weapon,
        origin,
        direction,
        end: origin + direction * distance,
        normal,
        pierced: None,
        pierced_kind: None,
        kind,
        target,
    }
}

struct Wounds<'a> {
    weapons: &'a WeaponNumbers,
    tps: u32,
    armor_absorb: f32,
    tick: u64,
    window: u64,
    tag_ticks: u32,
}

impl Wounds<'_> {
    fn apply(
        &self,
        pending: &[Pending],
        actors: &mut Query<Acting<'_>>,
        events: &mut ArmsEvents,
        book: &mut Book,
        drops: &mut Drops,
        spawner: &mut Spawner<ArmsBrick>,
    ) {
        let (tick, window) = (self.tick, self.window);
        book.ledger.prune(tick, window);
        let mut fallen: Vec<u32> = Vec::new();
        for wounding in pending {
            if fallen.contains(&wounding.victim) {
                continue;
            }
            let Some((entity, body, mut arms, mut health)) = actors
                .iter_mut()
                .find(|(_, actor, .., alive, _)| actor.slot == wounding.victim && *alive)
                .map(|(entity, _, _, body, arms, health, ..)| (entity, body, arms, health))
            else {
                continue;
            };
            let before = Vitals {
                health: health.health,
                armor: health.armour,
            };
            let hurt = wound(before, wounding.damage, wounding.part, self.armor_absorb);
            health.health = hurt.after.health;
            health.armour = hurt.after.armor;
            if wounding.tag_slow > 0.0 {
                arms.tag_slow = wounding.tag_slow;
                arms.tag_ticks = self.tag_ticks;
                arms.tag_total = self.tag_ticks;
            }
            book.ledger.record(wounding.victim, wounding.shooter, tick);
            book.hits += 1;
            book.heads += u64::from(hurt.headshot);
            book.damage += f64::from(hurt.health_lost);
            events.hits.push(HitEvent {
                shooter: wounding.shooter,
                victim: wounding.victim,
                weapon: wounding.weapon,
                part: wounding.part,
                distance: wounding.distance,
                damage: hurt.damage,
                health_lost: hurt.health_lost,
                armor_lost: hurt.armor_lost,
                health: hurt.after.health,
                armor: hurt.after.armor,
                killed: hurt.killed,
                headshot: hurt.headshot,
            });
            if !hurt.killed {
                continue;
            }
            fallen.push(wounding.victim);
            events.deaths.push(DeathEvent {
                victim: wounding.victim,
                killer: Some(wounding.shooter),
                assistants: book.ledger.assistants(
                    wounding.victim,
                    Some(wounding.shooter),
                    tick,
                    window,
                ),
                headshot: hurt.headshot,
                weapon: Some(wounding.weapon),
                ability: None,
            });
            book.ledger.forget_victim(wounding.victim);
            drop_weapon(&mut arms.loadout, self.weapons, self.tps, body.feet, drops);
            spawner.remove::<Alive>(entity);
        }
    }
}
