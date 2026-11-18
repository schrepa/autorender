use engine::prelude::*;
use glam::{Mat3, Vec3};

use super::cast::{Asked, rules_slot, slot_key};
use super::hold::Casting;
use super::publish::{self, Sheet};
use super::{Book, CastStat, Parts, PowerSetup, PowerStats, PowersBrick, ShieldHeld};
use crate::abilities::{self, ActorId, Event, World, WorldHit};
use crate::play::actors::{Actor, Alive, Body, Health, Human, SpeedFactor, Wish};
use crate::play::effects::EffectsFeed;
use crate::play::events::arms::{AbilityKill, ArmsEvents, DeathEvent};
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::paths::Hazards;
use crate::play::powers::{Powers, Push};
use crate::play::round::Round;
use crate::play::view::ViewOffsets;
use crate::play::world::Arena;
use crate::rules::{Phase, Team};
use crate::solids::{Block, Surface};
use crate::weapons::{BodyPart, Vitals, forward, wound};

struct ArenaWorld<'a>(&'a mut Arena);

impl World for ArenaWorld<'_> {
    fn ray(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<WorldHit> {
        self.0.ray(origin, direction, max).map(|hit| WorldHit {
            distance: hit.distance,
            normal: hit.normal,
        })
    }

    fn sphere_sweep(&self, centre: Vec3, radius: f32, motion: Vec3) -> Option<WorldHit> {
        self.0
            .solids()
            .sphere_sweep(centre, radius, motion)
            .map(|hit| WorldHit {
                distance: hit.distance,
                normal: hit.normal,
            })
    }

    fn add_box(&mut self, id: u32, centre: Vec3, half_extents: Vec3, axes: Mat3) {
        self.0.solids_mut().add_block(Block {
            id,
            centre,
            half_extents,
            axes,
            surface: Surface::Wood,
        });
    }

    fn remove_box(&mut self, id: u32) {
        self.0.solids_mut().remove_block(id);
    }
}

#[derive(SystemParam)]
pub(super) struct Facts<'w> {
    time: Res<'w, FixedTime>,
    setup: Res<'w, PowerSetup>,
    round: Option<Res<'w, Round>>,
    round_events: Option<Res<'w, RoundEvents>>,
}

#[derive(SystemParam)]
pub(super) struct Out<'w> {
    arena: ResMut<'w, Arena>,
    powers: ResMut<'w, Powers>,
    events: ResMut<'w, PowerEvents>,
    feed: Option<ResMut<'w, EffectsFeed>>,
    hazards: Option<ResMut<'w, Hazards>>,
    stats: ResMut<'w, PowerStats>,
    book: ResMut<'w, Book>,
    arms: Option<ResMut<'w, ArmsEvents>>,
}

pub(super) type Row<'a> = (
    Entity,
    &'a Actor,
    &'a Wish,
    &'a Body,
    &'a mut Health,
    Option<&'a mut SpeedFactor>,
    Option<&'a mut Push>,
    Option<&'a ViewOffsets>,
    Has<Alive>,
    Has<Human>,
);

fn view_of(wish: &Wish, offsets: Option<&ViewOffsets>) -> Vec3 {
    let (yaw, pitch) = offsets.map_or((wish.yaw_radians, wish.pitch_radians), |held| {
        (
            wish.yaw_radians + held.yaw_radians(),
            wish.pitch_radians + held.pitch_radians(),
        )
    });
    forward(yaw, pitch)
}

pub(super) fn power_actor(
    actor: &Actor,
    body: &Body,
    wish: &Wish,
    offsets: Option<&ViewOffsets>,
    parts: &Parts,
) -> abilities::Actor {
    abilities::Actor {
        id: ActorId(actor.slot),
        team: actor.team.index(),
        feet: body.feet,
        eye: body.eye,
        view: view_of(wish, offsets),
        radius: parts.radius_m,
        height: if body.crouched {
            parts.crouch_height_m
        } else {
            parts.height_m
        },
    }
}

fn round_restarted(book: &mut Book, round: Option<&Round>, events: Option<&RoundEvents>) -> bool {
    let matches = round.map_or(0, |round| round.matches);
    let fresh_match = matches != book.matches_seen;
    book.matches_seen = matches;
    let Some(events) = events else {
        return fresh_match;
    };
    if book.round_seen == Some(events.tick) {
        return fresh_match;
    }
    book.round_seen = Some(events.tick);
    fresh_match
        || events.events.iter().any(|event| {
            matches!(
                event,
                crate::rules::Event::Respawn { .. }
                    | crate::rules::Event::Phase {
                        phase: Phase::Buy | Phase::Over,
                        ..
                    }
            )
        })
}

struct Applied {
    deaths: Vec<DeathEvent>,
    human_damage: f32,
}

fn ability_death(
    book: &Book,
    parts: &Parts,
    (victim, killer, ability): (u32, Option<u32>, abilities::AbilityId),
    tick: u64,
) -> DeathEvent {
    DeathEvent {
        victim,
        killer,
        assistants: book
            .ledger
            .assistants(victim, killer, tick, parts.assist_ticks),
        headshot: false,
        weapon: None,
        ability: Some(AbilityKill {
            slot: rules_slot(ability.slot),
            name: book
                .abilities
                .as_ref()
                .map(|abilities| abilities.ability(ability).name.clone())
                .unwrap_or_default(),
        }),
    }
}

fn apply_events(
    events: &[Event],
    rows: &mut Query<Row<'_>>,
    order: &[(u32, Entity)],
    (book, powers, tick): (&mut Book, &Powers, u64),
    spawner: &mut Spawner<PowersBrick>,
) -> Applied {
    let mut applied = Applied {
        deaths: Vec::new(),
        human_damage: 0.0,
    };
    let Some(parts) = book.parts else {
        return applied;
    };
    let entity_of = |slot: u32| {
        order
            .iter()
            .find(|(held, _)| *held == slot)
            .map(|(_, entity)| *entity)
    };
    let mut fallen: Vec<u32> = Vec::new();
    for event in events {
        match event {
            Event::Damage {
                target,
                amount,
                by,
                ability,
            } => {
                if fallen.contains(&target.0) {
                    continue;
                }
                let Some(Ok(mut row)) = entity_of(target.0).map(|entity| rows.get_mut(entity))
                else {
                    continue;
                };
                let taken = powers.damage_taken(target.0);
                let before = Vitals {
                    health: row.4.health,
                    armor: row.4.armour,
                };
                let hurt = wound(before, amount * taken, BodyPart::Body, parts.armor_absorb);
                row.4.health = hurt.after.health;
                row.4.armour = hurt.after.armor;
                if row.9 {
                    applied.human_damage += hurt.damage;
                }
                if by.0 != target.0 {
                    book.ledger.record(target.0, by.0, tick);
                }
                if !hurt.killed {
                    continue;
                }
                fallen.push(target.0);
                let killer = (by.0 != target.0).then_some(by.0);
                applied.deaths.push(ability_death(
                    book,
                    &parts,
                    (target.0, killer, *ability),
                    tick,
                ));
                book.ledger.forget_victim(target.0);
                spawner.remove::<Alive>(row.0);
            }
            Event::Heal { target, amount, .. } => {
                if let Some(Ok(mut row)) = entity_of(target.0).map(|entity| rows.get_mut(entity))
                    && row.8
                    && row.4.health > 0.0
                {
                    let healed = row.4.health + amount;
                    row.4.health = if healed > parts.max_health {
                        parts.max_health
                    } else {
                        healed
                    };
                }
            }
            Event::Shield {
                target,
                amount,
                until,
                ..
            } => {
                if let Some(Ok(mut row)) = entity_of(target.0).map(|entity| rows.get_mut(entity)) {
                    if let Some(until) = until {
                        book.shields.push(ShieldHeld {
                            slot: target.0,
                            until: *until,
                            armour_before: row.4.armour,
                        });
                    }
                    row.4.armour += amount;
                }
            }
            Event::Moved { actor, by } => {
                if let Some(Ok(mut row)) = entity_of(actor.0).map(|entity| rows.get_mut(entity))
                    && let Some(push) = row.6.as_mut()
                {
                    push.0 += *by;
                }
            }
            Event::BuffStarted { .. }
            | Event::BuffEnded { .. }
            | Event::Blinded { .. }
            | Event::RevealStarted { .. }
            | Event::RevealEnded { .. }
            | Event::SolidAdded { .. }
            | Event::SolidRemoved { .. }
            | Event::TrapArmed { .. }
            | Event::TrapTriggered { .. }
            | Event::Destroyed { .. }
            | Event::Transformed { .. }
            | Event::TransformEnded { .. }
            | Event::Ended { .. }
            | Event::Cue { .. } => {}
        }
    }
    applied
}

fn expire_shields(book: &mut Book, rows: &mut Query<Row<'_>>, order: &[(u32, Entity)], tick: u64) {
    let (done, kept): (Vec<ShieldHeld>, Vec<ShieldHeld>) =
        book.shields.iter().partition(|shield| shield.until <= tick);
    book.shields = kept;
    for shield in done {
        let Some((_, entity)) = order.iter().find(|(slot, _)| *slot == shield.slot) else {
            continue;
        };
        if let Ok(mut row) = rows.get_mut(*entity)
            && row.4.armour > shield.armour_before
        {
            row.4.armour = shield.armour_before;
        }
    }
}

fn ordered(rows: &Query<Row<'_>>, spawner: &mut Spawner<PowersBrick>) -> Vec<(u32, Entity)> {
    for (entity, .., push, _, _, _) in rows.iter() {
        if push.is_none() {
            spawner.insert(entity, Push::default());
        }
    }
    let mut order: Vec<(u32, Entity)> = rows
        .iter()
        .map(|(entity, actor, ..)| (actor.slot, entity))
        .collect();
    order.sort_unstable();
    order
}

fn tally(
    (casts_total, last_cast): (&mut u64, &mut Option<CastStat>),
    (refused_total, last_refusal): (&mut u64, &mut Option<CastStat>),
    events: &PowerEvents,
    tick: u64,
) {
    for cast in &events.casts {
        *casts_total += 1;
        *last_cast = Some(CastStat {
            tick,
            slot: cast.slot,
            ability: slot_key(cast.ability).to_owned(),
            reason: None,
        });
    }
    for refused in &events.refused {
        *refused_total += 1;
        *last_refusal = Some(CastStat {
            tick,
            slot: refused.slot,
            ability: slot_key(refused.ability).to_owned(),
            reason: Some(refused.reason.key().to_owned()),
        });
    }
}

fn hidden_from_human(rows: &Query<Row<'_>>, order: &[(u32, Entity)], powers: &Powers) -> Vec<u32> {
    let eyes: Vec<(u32, Vec3, bool)> = order
        .iter()
        .filter_map(|(_, entity)| {
            let row = rows.get(*entity).ok()?;
            Some((row.1.slot, row.3.eye, row.9))
        })
        .collect();
    let Some(&(_, eye, _)) = eyes.iter().find(|held| held.2) else {
        return Vec::new();
    };
    eyes.iter()
        .filter(|held| !held.2 && powers.sight_blocked(eye, held.1))
        .map(|held| held.0)
        .collect()
}

fn faces(rows: &Query<Row<'_>>, order: &[(u32, Entity)]) -> Vec<(u32, Team, String, bool)> {
    order
        .iter()
        .filter_map(|(_, entity)| {
            let row = rows.get(*entity).ok()?;
            Some((row.1.slot, row.1.team, row.1.character.clone(), row.9))
        })
        .collect()
}

pub(super) fn act(
    facts: Facts<'_>,
    mut out: Out<'_>,
    mut rows: Query<Row<'_>>,
    mut spawner: Spawner<PowersBrick>,
) {
    let tick = facts.time.tick;
    let Out {
        arena,
        powers,
        events,
        feed,
        hazards,
        stats,
        book,
        arms,
    } = &mut out;
    let arena: &mut Arena = arena;
    let powers: &mut Powers = powers;
    let events: &mut PowerEvents = events;
    let book: &mut Book = book;
    let stats: &mut PowerStats = stats;
    *events = PowerEvents {
        tick,
        ..PowerEvents::default()
    };
    let order = ordered(&rows, &mut spawner);
    let mut world = ArenaWorld(arena);
    let mut ticked = Vec::new();
    if round_restarted(book, facts.round.as_deref(), facts.round_events.as_deref()) {
        ticked.extend(powers.effects.end_round(&mut world));
        book.shields.clear();
        powers.struck.clear();
        powers.held.clear();
        book.ledger = Default::default();
    }
    for hit in std::mem::take(&mut powers.struck) {
        ticked.extend(powers.effects.damage_effect(
            &mut world,
            hit.effect,
            hit.damage,
            ActorId(hit.shooter),
        ));
    }
    let Some(parts) = book.parts else {
        publish::write_stats(stats, book, &facts.setup, &[], tick);
        return;
    };
    let living: Vec<abilities::Actor> = order
        .iter()
        .filter_map(|(_, entity)| {
            let (_, actor, wish, body, health, _, _, offsets, alive, _) = rows.get(*entity).ok()?;
            (alive && health.health > 0.0).then(|| power_actor(actor, body, wish, offsets, &parts))
        })
        .collect();
    ticked.extend(powers.effects.tick(&mut world, &living, tick));
    book.ledger.prune(tick, parts.assist_ticks);
    let applied = apply_events(
        &ticked,
        &mut rows,
        &order,
        (&mut *book, &*powers, tick),
        &mut spawner,
    );
    expire_shields(book, &mut rows, &order, tick);
    for (_, entity) in &order {
        if let Ok(mut row) = rows.get_mut(*entity)
            && let Some(speed) = row.5.as_mut()
        {
            speed.0 *= powers.modifiers(row.1.slot).speed;
        }
    }
    let asked = Asked {
        book: book.abilities.as_ref(),
        view: facts.round.as_deref().and_then(Round::view),
        free: facts.setup.free,
    };
    let fallen: Vec<u32> = applied.deaths.iter().map(|death| death.victim).collect();
    let casting = Casting {
        relay: facts.setup.cast_as,
        asked: &asked,
        parts: &parts,
        fallen: &fallen,
        free: facts.setup.free,
        tick,
    };
    casting.run(&rows, &order, (&mut *powers, &mut *events, &mut ticked));
    casting.previews(&world, &rows, &order, powers);
    tally(
        (&mut book.casts_total, &mut book.last_cast),
        (&mut book.refused_total, &mut book.last_refusal),
        events,
        tick,
    );
    publish::remember_cues(&mut book.recent_cues, &ticked, tick);
    let faces = faces(&rows, &order);
    let sheet = Sheet {
        book: &*book,
        powers: &*powers,
        asked: &asked,
        tick,
        human: faces
            .iter()
            .find(|face| face.3)
            .map(|face| (face.0, face.2.clone())),
        human_damage: applied.human_damage,
        events: &ticked,
    };
    let actors = sheet.actor_views(&faces);
    if let Some(feed) = feed.as_deref_mut() {
        *feed = sheet.feed();
    }
    if let Some(hazards) = hazards.as_deref_mut() {
        *hazards = sheet.hazards();
    }
    let human_slots = sheet.human_slots();
    let live = sheet.live();
    if let Some(arms) = arms.as_deref_mut() {
        arms.deaths.extend(applied.deaths);
    }
    events.events = ticked;
    powers.actors = actors;
    publish::write_stats(stats, book, &facts.setup, &live, tick);
    publish::write_tick_stats(stats, events, powers, human_slots);
    stats.hidden_from_human = hidden_from_human(&rows, &order, powers);
}
