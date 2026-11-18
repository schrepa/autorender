use engine::prelude::*;
use glam::Vec3;

use super::{ArmsBrick, Book, Setup, drop_weapon, ticks_per_second};
use crate::names::{HEAD, HIP, SPINE};
use crate::play::actors::{Actor, Alive, Body, Equip, Health, SpeedFactor, Wish};
use crate::play::arms::{Arms, Drops};
use crate::play::events::arms::{
    ArmsEvents, DeathEvent, Handling, HandlingEvent, HitEvent, ShotEvent, Target, Weapon,
};
use crate::play::figures::{FigureJoints, LEFT_LEG, RIGHT_LEG};
use crate::play::powers::{EffectHit, Powers};
use crate::play::view::ViewOffsets;
use crate::play::world::Arena;
use crate::rules::Team;
use crate::solids::{Hit, Solid, Surface};
use crate::weapons::{
    Aim, ArmsNumbers, Blocker, BodyPart, CharacterId, Fired, Footing, GunKind, GunNumbers,
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
        let ticked = advance(numbers, tps, (pace, holding), wish, body, &mut arms);
        let recoil = arms.loadout.recoil();
        offsets.recoil_pitch_radians = recoil.pitch_radians;
        offsets.recoil_yaw_radians = recoil.yaw_radians;
        offsets.zoom = arms.loadout.view_zoom(&numbers.weapons);
        speed.0 = arms.loadout.speed_factor(&numbers.weapons) * arms.tag_factor();
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

fn advance(
    numbers: &ArmsNumbers,
    tps: u32,
    (pace, holding): (Pace, bool),
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
    let input = LoadoutInput {
        fire_held: !holding && (wish.fire_held || wish.fire_pressed),
        reload_pressed: wish.reload,
        scope_pressed: !holding && wish.scope,
        switch_to: wish.equip.and_then(slot_of),
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
        let through = wall.and_then(|hit| self.pierce(numbers, ray, range, hit));
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
                gun_damage(numbers, part, distance) * combat.pierce_damage,
                combat.pierce_tag_slow,
            )),
            Struck::World { .. } | Struck::Nothing => None,
        };
        let mut event = shot_event(self.slot, weapon, ray, range, beyond, through.wall);
        event.pierced = Some(through.entry);
        (event, pending)
    }

    fn pierce(
        &self,
        gun: &GunNumbers,
        (origin, direction): (Vec3, Vec3),
        range: f32,
        wall: Hit,
    ) -> Option<Through> {
        let hardness = *self
            .numbers
            .combat
            .pierce_hardness
            .get(surface_key(wall.surface))?;
        let reach = gun.pierce_m / hardness;
        let probe = wall.distance + reach;
        if reach <= 0.0 || probe >= range {
            return None;
        }
        let beyond = origin + direction * probe;
        let back = self.arena.ray(beyond, -direction, reach)?;
        if back.solid == wall.solid {
            return None;
        }
        let onward = self
            .arena
            .ray(beyond, direction, range - probe)
            .map(|hit| Hit {
                distance: hit.distance + probe,
                ..hit
            });
        Some(Through {
            entry: origin + direction * wall.distance,
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

fn surface_key(surface: Surface) -> &'static str {
    match surface {
        Surface::Stone => "stone",
        Surface::Metal => "metal",
        Surface::Wood => "wood",
        Surface::Dirt => "dirt",
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
