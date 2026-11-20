engine::brick!(cues);

mod events;
mod numbers;

use std::collections::{BTreeMap, BTreeSet};

use brick_sound::{Anchor, Cue, Ear, Hearing, SoundCues, SoundLoad};
use engine::math::sin_cos;
use engine::prelude::*;
use engine_core::Stats;
use glam::Vec3;

use crate::play::actors::{Actor, Body, Human, Wish};
use crate::play::arms::Drops;
use crate::play::events::arms::ArmsEvents;
use crate::play::events::body::BodyEvents;
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::numbers::Numbers;
use crate::play::powers::Powers;
use crate::play::round::Round;
use crate::play::screen::{ScreenSounds, UiSound};
use crate::play::sound::Volume;
use crate::play::view::Watched;
use crate::play::world::{Arena, Barrier};

use events::Repeaters;
use numbers::CueNumbers;

const BARRIER: u64 = 1 << 32;
const HUM: u64 = 1 << 60;

stats! {
    #[section("cues")]
    pub struct CuesStats {
        pub refusal: Option<String>,
    }
}

#[derive(Resource, Default)]
struct Book {
    revision: Option<u64>,
    numbers: Option<CueNumbers>,
    expected: BTreeSet<String>,
    refusal: Option<String>,
    repeaters: Repeaters,
    drops_seen: Vec<u32>,
    was_ready: bool,
    scopes: BTreeMap<u32, (crate::play::events::arms::Weapon, usize)>,
}

pub struct CuesBrick;

impl Brick for CuesBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<Volume>()
            .resource::<Book>()
            .systems(Schedule::FixedSimulation, cue)
            .systems(Schedule::Presentation, hear)
            .systems(Schedule::Presentation, ui_cues)
            .dev(|dev| {
                dev.stats::<CuesStats>()
                    .systems(Schedule::Presentation, report);
            });
    }
}

fn hear(
    numbers: Res<Numbers>,
    volume: Res<Volume>,
    mut book: ResMut<Book>,
    mut hearing: ResMut<Hearing>,
) {
    if hearing.master != Some(volume.master) {
        hearing.master = Some(volume.master);
    }
    for (group, gain) in [
        ("effects", volume.effects),
        ("interface", volume.interface),
        ("footsteps", volume.footsteps),
    ] {
        match hearing.groups.get_mut(group) {
            Some(held) => *held = gain,
            None => {
                hearing.groups.insert(group.to_owned(), gain);
            }
        }
    }
    let Some(game) = numbers.get() else {
        return;
    };
    if book.revision == Some(numbers.revision()) {
        return;
    }
    book.revision = Some(numbers.revision());
    match numbers::read(game) {
        Ok(read) => {
            book.numbers = Some(read.numbers);
            book.expected = read.expected;
            book.refusal = None;
            hearing.reach = Some(read.reach);
        }
        Err(error) => {
            book.numbers = None;
            book.refusal = Some(error);
        }
    }
}

fn ui_cues(mut screen: Option<ResMut<ScreenSounds>>, mut cues: ResMut<SoundCues>) {
    let Some(screen) = screen.as_deref_mut() else {
        return;
    };
    if screen.queued.is_empty() {
        return;
    }
    for sound in screen.queued.drain(..) {
        let name = match sound {
            UiSound::Hover => "ui.hover",
            UiSound::Click => "ui.click",
            UiSound::Ready => "ui.ready",
        };
        cues.immediate.push(events::cue(name, Anchor::Listener, None));
    }
}

fn report(book: Res<Book>, sound: Res<SoundLoad>, mut stats: Stats<CuesStats>) {
    let missing = (book.expected.iter())
        .find(|event| sound.ready && !sound.events.contains(*event))
        .map(|event| format!("events: no event {event}"));
    stats.edit(|stats| stats.refusal = book.refusal.clone().or(missing));
}

type ActorRow<'a> = (
    &'a Actor,
    &'a Body,
    Has<Human>,
    Option<&'a Wish>,
    Option<&'a crate::play::arms::Arms>,
);

struct Row<'a> {
    slot: u32,
    team: crate::rules::Team,
    character: &'a str,
    body: Body,
    human: bool,
    yaw: f32,
    scope: Option<(crate::play::events::arms::Weapon, usize)>,
}

type Scene<'w> = (
    Option<Res<'w, Barrier>>,
    Option<Res<'w, Drops>>,
    Option<Res<'w, Powers>>,
);

type Feeds<'w> = (
    Option<Res<'w, ArmsEvents>>,
    Option<Res<'w, PowerEvents>>,
    Option<Res<'w, RoundEvents>>,
    Option<Res<'w, BodyEvents>>,
);

fn ear(eye: Vec3, yaw_radians: f32) -> Ear {
    let (sin, cos) = sin_cos(yaw_radians);
    Ear {
        eye,
        right: Vec3::new(cos, 0.0, -sin),
    }
}

fn listener(rows: &[Row<'_>], watched: Option<&Watched>) -> Option<(Option<u32>, Ear)> {
    let heard = |row: &Row<'_>| (Some(row.slot), ear(row.body.eye, row.yaw));
    let watched_row = watched
        .and_then(|watched| watched.slot)
        .and_then(|slot| rows.iter().find(|row| row.slot == slot));
    match (watched_row, watched.and_then(|watched| watched.corpse)) {
        (Some(row), _) => Some(heard(row)),
        (None, Some(corpse)) => Some((None, ear(corpse.eye, corpse.yaw_radians))),
        (None, None) => rows.iter().find(|row| row.human).map(heard),
    }
}

fn listen(
    hearing: &mut Hearing,
    rows: &[Row<'_>],
    heard_by: Option<(Option<u32>, Ear)>,
    barrier: Option<&Barrier>,
    volume: &Volume,
) {
    hearing.ear = heard_by.map(|(_, ear)| ear);
    hearing.own = heard_by.and_then(|(slot, _)| slot).map(u64::from);
    hearing.follows.retain(|key, _| rows.iter().any(|row| u64::from(row.slot) == *key));
    for row in rows {
        hearing.follows.insert(u64::from(row.slot), row.body.eye);
    }
    if let Some(point) = (barrier.filter(|barrier| barrier.up))
        .zip(hearing.ear)
        .and_then(|(barrier, ear)| events::barrier_point(barrier, ear.eye))
    {
        hearing.follows.insert(BARRIER, point);
    }
    for (group, gain) in [
        ("effects", volume.effects),
        ("interface", volume.interface),
        ("footsteps", volume.footsteps),
    ] {
        match hearing.groups.get_mut(group) {
            Some(held) => *held = gain,
            None => {
                hearing.groups.insert(group.to_owned(), gain);
            }
        }
    }
}

#[expect(clippy::too_many_arguments, reason = "a system's parameters")]
fn cue(
    time: Res<FixedTime>,
    arena: Option<Res<Arena>>,
    (arms, powers, round_events, body_events): Feeds<'_>,
    (barrier, drops, power_state): Scene<'_>,
    round: Option<Res<Round>>,
    (watched, volume): (Option<Res<Watched>>, Res<Volume>),
    actors: Query<ActorRow<'_>>,
    mut book: ResMut<Book>,
    (mut cues, mut hearing): (ResMut<SoundCues>, ResMut<Hearing>),
) {
    let tps = time.rate;
    let now = time.tick;
    let rows = rows_of(&actors);
    let human = rows.iter().find(|row| row.human);
    let who = events::Human {
        slot: human.map(|row| row.slot),
        team: human.map(|row| row.team),
    };
    let heard_by = listener(&rows, watched.as_deref());
    let own = heard_by.and_then(|(slot, _)| slot);
    let barrier = barrier.as_deref();
    listen(&mut hearing, &rows, heard_by, barrier, &volume);
    let SoundCues {
        cues: mut fresh,
        mut stops,
        immediate,
        ..
    } = std::mem::take(&mut *cues);
    fresh.clear();
    stops.clear();
    let book = &mut *book;
    if let Some(numbers) = &book.numbers {
        let fed = Fed {
            who,
            rows: &rows,
            round: round.as_deref(),
            numbers,
        };
        gunplay(
            &mut book.scopes,
            arms.filter(|arms| arms.tick == now).as_deref(),
            &fed,
            &mut fresh,
        );
        if let Some(powers) = powers.filter(|powers| powers.tick == now) {
            let heard = events::Heard {
                human: who,
                listener: own,
                casters: &rows,
                now,
                numbers,
                orbs: power_state.as_deref().map_or(&[], Powers::orbs),
            };
            events::powers(&powers, &heard, &mut fresh, &mut stops);
        }
        if let Some(round_events) = round_events {
            events::round(
                &round_events.events,
                who,
                &mut book.repeaters,
                now,
                &mut fresh,
            );
        }
        if let Some(body_events) = body_events.filter(|events| events.tick == now) {
            let slowed = |slot: u32| {
                power_state
                    .as_deref()
                    .is_some_and(|powers| powers.modifiers(slot).speed < 1.0)
            };
            events::bodies(&body_events, own, slowed, numbers, &mut fresh);
        }
        book.repeaters.tick(now, tps, numbers, &mut fresh);
        let clock = Clock {
            numbers,
            tps,
            now,
            human: who.slot,
        };
        timing(
            &clock,
            round.as_deref(),
            barrier,
            &mut book.was_ready,
            &mut fresh,
        );
        if let Some(drops) = drops.as_deref() {
            events::drops(
                drops,
                &mut book.drops_seen,
                arena.as_deref(),
                numbers,
                &mut fresh,
            );
        }
        hum(barrier, numbers, &mut fresh, &mut stops);
    }
    *cues = SoundCues {
        tick: now,
        cues: fresh,
        stops,
        immediate,
    };
}

fn rows_of<'a>(actors: &'a Query<ActorRow<'_>>) -> Vec<Row<'a>> {
    let mut rows: Vec<Row<'a>> = actors
        .iter()
        .map(|(actor, body, human, wish, arms)| Row {
            slot: actor.slot,
            team: actor.team,
            character: &actor.character,
            body: *body,
            human,
            yaw: wish.map_or(actor.yaw_radians, |wish| wish.yaw_radians),
            scope: arms.and_then(|arms| arms.loadout.active_gun()).map(|gun| {
                (
                    crate::play::events::arms::Weapon::Gun(gun.kind()),
                    gun.scope_level(),
                )
            }),
        })
        .collect();
    rows.sort_by_key(|row| row.slot);
    rows
}

struct Fed<'a> {
    who: events::Human,
    rows: &'a [Row<'a>],
    round: Option<&'a Round>,
    numbers: &'a CueNumbers,
}

fn gunplay(
    scopes: &mut BTreeMap<u32, (crate::play::events::arms::Weapon, usize)>,
    arms: Option<&ArmsEvents>,
    fed: &Fed<'_>,
    out: &mut Vec<Cue>,
) {
    let round_kills = fed
        .round
        .and_then(Round::view)
        .zip(fed.who.slot)
        .and_then(|(view, slot)| {
            view.players
                .get(slot as usize)
                .map(|player| player.round_kills)
        });
    if let Some(arms) = arms {
        events::arms(arms, (fed.who, round_kills), fed.numbers, out);
    }
    events::scopes(fed.rows, scopes, arms, out);
}

struct Clock<'a> {
    numbers: &'a CueNumbers,
    tps: u32,
    now: u64,
    human: Option<u32>,
}

fn timing(
    clock: &Clock<'_>,
    round: Option<&Round>,
    barrier: Option<&Barrier>,
    was_ready: &mut bool,
    out: &mut Vec<Cue>,
) {
    let view = round.and_then(Round::view);
    if let Some(view) = &view
        && view.phase == crate::rules::Phase::Buy
    {
        events::countdown(view.ticks_left, clock.tps, clock.numbers, out);
    }
    let human_ready = view
        .zip(clock.human)
        .is_some_and(|(view, slot)| view.ready(slot as usize));
    events::ready(was_ready, human_ready, out);
    if barrier.is_some_and(|barrier| barrier.dropped_tick == Some(clock.now)) {
        out.push(events::cue("barrier.drop", Anchor::Listener, None));
    }
}

fn hum(
    barrier: Option<&Barrier>,
    numbers: &CueNumbers,
    out: &mut Vec<Cue>,
    stops: &mut Vec<u64>,
) {
    if barrier.is_some_and(|barrier| barrier.up && !barrier.faces.is_empty()) {
        out.push(Cue {
            hold_ticks: Some(numbers.barrier_keep_ticks),
            tag: Some(HUM),
            ..events::cue("barrier.hum", Anchor::Follow(BARRIER), None)
        });
    } else {
        stops.push(HUM);
    }
}
