engine::brick!(sound);

mod events;
mod spatial;
mod table;
mod voices;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::Error;
use engine::prelude::*;
use engine_core::{Loaded, Output, Play, SceneCx, SoundId, Sounds, Tune};
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::actors::{Actor, Body, Human, Wish};
use crate::play::events::arms::ArmsEvents;
use crate::play::events::body::BodyEvents;
use crate::play::events::powers::PowerEvents;
use crate::play::events::round::RoundEvents;
use crate::play::round::Round;
use crate::play::sound::{Anchor, Cue, SoundCues, SoundLoad, Volume};
use crate::play::view::Watched;
use crate::play::world::Arena;

use events::Repeaters;
use spatial::Listener;
use table::{Group, Table};
use voices::{Admit, Rng, Voice};

#[derive(Component, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SoundTable {
    table: String,
    #[serde(skip)]
    parsed: Option<Result<Arc<Table>, String>>,
}

impl SoundTable {
    fn files(&self) -> Vec<String> {
        vec![self.table.clone()]
    }

    fn prepared(mut self, cx: &SceneCx<'_>, path: &str) -> Result<Self, Error> {
        let at = format!("{path}.table");
        let Some(bytes) = cx.buffer(&self.table, &at)? else {
            return Ok(self);
        };
        let parsed = Table::parse(&bytes).map(Arc::new);
        self.parsed = Some(parsed.map_err(|error| format!("{}: {error}", self.table)));
        Ok(self)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DoorStats {
    pub output: String,
    pub voices: u32,
    pub ready: u32,
    pub failed: u32,
    pub bytes_decoded: u64,
    pub refused: u64,
    pub dropped: u64,
    pub underruns: Option<u64>,
    pub latency_seconds: Option<f32>,
}

stats! {
    #[section("sound")]
    pub struct SoundStats {
        pub table: String,
        pub cues_tick: u32,
        pub cues: u64,
        pub voices: u32,
        pub started: u64,
        pub stolen: u64,
        pub dropped: u64,
        pub inaudible: u64,
        pub unmapped: u64,
        pub refused: u64,
        pub asked: u32,
        pub loaded: u32,
        pub ready: bool,
        pub waiting: Vec<String>,
        pub door: DoorStats,
    }
}

#[derive(Resource, Default)]
struct Bank {
    table: Option<Arc<Table>>,
    refusal: Option<String>,
    files: BTreeMap<String, SoundId>,
    opened: bool,
    master: Option<f32>,
}

#[derive(Resource, Default)]
struct Mixdown {
    voices: Vec<Voice>,
    rng: Option<Rng>,
    last: BTreeMap<String, usize>,
    repeaters: Repeaters,
    cues_tick: u32,
    cues: u64,
    started: u64,
    stolen: u64,
    dropped: u64,
    inaudible: u64,
    unmapped: u64,
    refused: u64,
}

pub struct SoundBrick;

impl Brick for SoundBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<SoundTable>("sound")
                    .schema(engine_core::schema_of::<SoundTable>)
                    .files(SoundTable::files)
                    .prepare(SoundTable::prepared),
            )
            .resource::<SoundCues>()
            .resource::<SoundLoad>()
            .settings::<Volume>()
            .resource::<Bank>()
            .resource::<Mixdown>()
            .stats::<SoundStats>()
            .systems(Schedule::FixedSimulation, mix)
            .systems(Schedule::Presentation, prepare)
            .systems(Schedule::Presentation, report);
    }
}

fn prepare(
    tables: Query<Ref<SoundTable>>,
    volume: Res<Volume>,
    mut bank: ResMut<Bank>,
    mut sounds: ResMut<Sounds>,
) {
    for table in &tables {
        if !table.is_changed() {
            continue;
        }
        match &table.parsed {
            Some(Ok(parsed)) => adopt(&mut bank, Arc::clone(parsed), &mut sounds),
            Some(Err(error)) => bank.refusal = Some(error.clone()),
            None => {}
        }
    }
    let Some(table) = bank.table.as_ref() else {
        return;
    };
    let master = table.master_gain * volume.master;
    let glide = table.glide_seconds;
    if bank.master != Some(master) {
        sounds.master(master, glide);
        bank.master = Some(master);
    }
}

fn adopt(bank: &mut Bank, table: Arc<Table>, sounds: &mut Sounds) {
    if !bank.opened {
        sounds.open(table.latency_seconds);
        bank.opened = true;
    }
    bank.master = None;
    for sound in table.sounds.values() {
        for file in &sound.files {
            if !bank.files.contains_key(&file.file) {
                let id = sounds.load(&file.file);
                bank.files.insert(file.file.clone(), id);
            }
        }
    }
    bank.table = Some(table);
    bank.refusal = None;
}

fn ticks_per_second(time: &FixedTime) -> Option<u32> {
    (time.delta_seconds > 0.0).then(|| (1.0 / time.delta_seconds).round() as u32)
}

type ActorRow<'a> = (&'a Actor, &'a Body, Has<Human>, Option<&'a Wish>);

struct Row {
    slot: u32,
    team: crate::rules::Team,
    body: Body,
    human: bool,
    yaw: f32,
}

type Feeds<'w> = (
    Option<Res<'w, ArmsEvents>>,
    Option<Res<'w, PowerEvents>>,
    Option<Res<'w, RoundEvents>>,
    Option<Res<'w, BodyEvents>>,
);

fn listener(rows: &[Row], watched: Option<&Watched>) -> Option<(Option<u32>, Listener)> {
    let heard = |row: &Row| (Some(row.slot), Listener::new(row.body.eye, row.yaw));
    let watched_row = watched
        .and_then(|watched| watched.slot)
        .and_then(|slot| rows.iter().find(|row| row.slot == slot));
    match (watched_row, watched.and_then(|watched| watched.corpse)) {
        (Some(row), _) => Some(heard(row)),
        (None, Some(corpse)) => Some((None, Listener::new(corpse.eye, corpse.yaw_radians))),
        (None, None) => rows.iter().find(|row| row.human).map(heard),
    }
}

#[expect(clippy::too_many_arguments, reason = "a system's parameters")]
fn mix(
    time: Res<FixedTime>,
    bank: Res<Bank>,
    arena: Option<Res<Arena>>,
    (arms, powers, round_events, body_events): Feeds<'_>,
    round: Option<Res<Round>>,
    watched: Option<Res<Watched>>,
    volume: Res<Volume>,
    mut cues: ResMut<SoundCues>,
    actors: Query<ActorRow<'_>>,
    mut mixdown: ResMut<Mixdown>,
    mut sounds: ResMut<Sounds>,
) {
    let heard = std::mem::take(&mut cues.cues);
    let (Some(table), Some(tps)) = (bank.table.as_ref(), ticks_per_second(&time)) else {
        return;
    };
    let mixdown = &mut *mixdown;
    let now = time.tick;
    let mut rows: Vec<Row> = actors
        .iter()
        .map(|(actor, body, human, wish)| Row {
            slot: actor.slot,
            team: actor.team,
            body: *body,
            human,
            yaw: wish.map_or(actor.yaw_radians, |wish| wish.yaw_radians),
        })
        .collect();
    rows.sort_by_key(|row| row.slot);
    let human = rows.iter().find(|row| row.human);
    let who = events::Human {
        slot: human.map(|row| row.slot),
        team: human.map(|row| row.team),
    };
    let heard_by = listener(&rows, watched.as_deref());
    let own = heard_by.and_then(|(slot, _)| slot);
    let mut fresh = heard;
    if let Some(arms) = arms.filter(|arms| arms.tick == now) {
        events::arms(&arms, who, table, &mut fresh);
    }
    if let Some(powers) = powers.filter(|powers| powers.tick == now) {
        events::powers(&powers, who, now, &mut fresh);
    }
    if let Some(round_events) = round_events {
        events::round(
            &round_events.events,
            who,
            &mut mixdown.repeaters,
            now,
            &mut fresh,
        );
    }
    if let Some(body_events) = body_events.filter(|events| events.tick == now) {
        events::bodies(&body_events, own, table, &mut fresh);
    }
    mixdown.repeaters.tick(now, tps, table, &mut fresh);
    if let Some(view) = round.as_deref().and_then(Round::view)
        && view.phase == crate::rules::Phase::Buy
    {
        events::countdown(view.ticks_left, tps, table, &mut fresh);
    }
    mixdown.cues_tick = u32::try_from(fresh.len()).unwrap_or(u32::MAX);
    mixdown.cues += fresh.len() as u64;
    let Some((_, listener)) = heard_by else {
        return;
    };
    let eyes: BTreeMap<u32, Vec3> = rows.iter().map(|row| (row.slot, row.body.eye)).collect();
    let place = Place {
        table,
        listener,
        own,
        volume: volume.clone(),
        eyes: &eyes,
        arena: arena.as_deref(),
        now,
        tps,
    };
    follow(mixdown, &place, &mut sounds);
    let rng = mixdown.rng.get_or_insert_with(|| Rng::new(table.seed));
    let mut rng = *rng;
    for cue in fresh {
        start(mixdown, &place, &bank, &mut rng, cue, &mut sounds);
    }
    mixdown.rng = Some(rng);
}

struct Place<'a> {
    table: &'a Table,
    listener: Listener,
    own: Option<u32>,
    volume: Volume,
    eyes: &'a BTreeMap<u32, Vec3>,
    arena: Option<&'a Arena>,
    now: u64,
    tps: u32,
}

impl Place<'_> {
    fn at(&self, anchor: Anchor, last: Vec3) -> Vec3 {
        match anchor {
            Anchor::Listener => self.listener.eye,
            Anchor::Point(point) => point,
            Anchor::Actor(slot) => self.eyes.get(&slot).copied().unwrap_or(last),
        }
    }

    fn group(&self, group: Group) -> f32 {
        match group {
            Group::Effects => self.volume.effects,
            Group::Interface => self.volume.interface,
            Group::Footsteps => self.volume.footsteps,
        }
    }

    fn own(&self, source: Option<u32>) -> bool {
        source.is_some() && source == self.own
    }

    fn occlusion(&self, curve: &table::Curve, at: Vec3) -> f32 {
        self.arena.map_or(1.0, |arena| {
            self.listener
                .occlusion(arena, curve, at, self.table.occlusion_pullback_m)
        })
    }

    fn shape(&self, curve: &table::Curve, own: bool, anchor: Anchor, at: Vec3) -> (f32, f32) {
        if own || anchor == Anchor::Listener || curve.flat {
            return (1.0, 0.0);
        }
        (
            self.listener.attenuation(curve, at),
            self.listener.pan(at, self.table.pan_width),
        )
    }
}

fn follow(mixdown: &mut Mixdown, place: &Place<'_>, sounds: &mut Sounds) {
    let table = place.table;
    let refresh = place
        .now
        .is_multiple_of(u64::from(table.occlusion_every_ticks));
    mixdown.voices.retain_mut(|voice| {
        if voice.ends <= place.now {
            if let (true, Some(door)) = (voice.looping, voice.door) {
                sounds.stop(door, table.hold_fade_seconds);
            }
            return false;
        }
        let Some(curve) = table
            .sounds
            .get(&voice.sound)
            .and_then(|sound| table.curves.get(&sound.curve))
        else {
            return true;
        };
        voice.at = place.at(voice.anchor, voice.at);
        let own = place.own(voice.source);
        if own {
            voice.occlusion = 1.0;
        } else if refresh {
            voice.occlusion = place.occlusion(curve, voice.at);
        }
        let (attenuation, pan) = place.shape(curve, own, voice.anchor, voice.at);
        voice.gain = voice.base * place.group(voice.group) * attenuation * voice.occlusion;
        voice.pan = pan;
        let moved = (voice.gain - voice.sent_gain).abs() > table.tune_step
            || (voice.pan - voice.sent_pan).abs() > table.tune_step;
        if let (true, Some(door)) = (moved, voice.door) {
            sounds.tune(
                door,
                Tune {
                    gain: voice.gain,
                    pan: voice.pan,
                    rate: voice.rate,
                    seconds: table.glide_seconds,
                },
            );
            voice.sent_gain = voice.gain;
            voice.sent_pan = voice.pan;
        }
        true
    });
}

fn start(
    mixdown: &mut Mixdown,
    place: &Place<'_>,
    bank: &Bank,
    rng: &mut Rng,
    cue: Cue,
    sounds: &mut Sounds,
) {
    let table = place.table;
    let Some((name, sound)) = table
        .events
        .get(&cue.event)
        .and_then(|name| Some((name, table.sounds.get(name)?)))
    else {
        mixdown.unmapped += 1;
        return;
    };
    let Some(curve) = table.curves.get(&sound.curve) else {
        mixdown.unmapped += 1;
        return;
    };
    let pick = voices::variation(rng, sound.files.len(), mixdown.last.get(name).copied());
    mixdown.last.insert(name.clone(), pick);
    let Some(file) = sound.files.get(pick) else {
        mixdown.unmapped += 1;
        return;
    };
    let own = place.own(cue.source);
    let at = place.at(cue.anchor, place.listener.eye);
    let occlusion = if own { 1.0 } else { place.occlusion(curve, at) };
    let (attenuation, pan) = place.shape(curve, own, cue.anchor, at);
    let base = sound.gain * cue.gain * if own { sound.own_gain } else { 1.0 };
    let gain = base * place.group(sound.group) * attenuation * occlusion;
    if gain < table.audible_gain {
        mixdown.inaudible += 1;
        return;
    }
    match voices::admit(&mixdown.voices, table.voices, sound.priority, gain) {
        Admit::Free => {}
        Admit::Steal(index) => {
            let stolen = mixdown.voices.swap_remove(index);
            if let Some(door) = stolen.door {
                sounds.stop(door, table.steal_fade_seconds);
            }
            mixdown.stolen += 1;
        }
        Admit::Drop => {
            mixdown.dropped += 1;
            return;
        }
    }
    let rate = 1.0 + rng.signed() * sound.jitter;
    let looping = sound.loops && cue.hold_ticks.is_some();
    let ends = match cue.hold_ticks.filter(|_| looping) {
        Some(hold) => place.now + u64::from(hold),
        None => place.now + (f64::from(file.seconds / rate) * f64::from(place.tps)).ceil() as u64,
    };
    let door = bank.files.get(&file.file).and_then(|id| {
        let played = sounds.play(Play {
            sound: *id,
            gain,
            pan,
            rate,
            looping,
            offset_seconds: 0.0,
        });
        played.ok()
    });
    if door.is_none() {
        mixdown.refused += 1;
    }
    mixdown.started += 1;
    mixdown.voices.push(Voice {
        door,
        sound: name.clone(),
        priority: sound.priority,
        base,
        gain,
        pan,
        rate,
        sent_gain: gain,
        sent_pan: pan,
        anchor: cue.anchor,
        at,
        source: cue.source,
        group: sound.group,
        occlusion,
        looping,
        ends,
    });
}

fn report(
    bank: Res<Bank>,
    mixdown: Res<Mixdown>,
    sounds: Res<Sounds>,
    mut load: ResMut<SoundLoad>,
    mut stats: ResMut<SoundStats>,
) {
    let door = sounds.report();
    let count = |ready: bool| {
        bank.files
            .values()
            .filter(|id| match sounds.loaded(**id) {
                Some(Loaded::Ready(_)) => ready,
                Some(Loaded::Failed(_)) => !ready,
                Some(Loaded::Reading | Loaded::Decoding) | None => false,
            })
            .count()
    };
    let (loaded, failed) = (count(true), count(false));
    let asked = bank.files.len();
    let waiting: Vec<String> = bank
        .files
        .iter()
        .filter_map(|(file, id)| match sounds.loaded(*id) {
            Some(Loaded::Reading) => Some(format!("{file} reading")),
            Some(Loaded::Decoding) => Some(format!("{file} decoding")),
            None => Some(format!("{file} unknown")),
            Some(Loaded::Ready(_) | Loaded::Failed(_)) => None,
        })
        .take(10)
        .collect();
    load.set_if_neq(SoundLoad {
        ready: bank.refusal.is_some() || (bank.table.is_some() && loaded + failed == asked),
        asked: u32::try_from(asked).unwrap_or(u32::MAX),
        loaded: u32::try_from(loaded).unwrap_or(u32::MAX),
        failed: u32::try_from(failed).unwrap_or(u32::MAX),
        refusal: bank.refusal.clone(),
    });
    let table = match (&bank.refusal, &bank.table) {
        (Some(refusal), _) => refusal.clone(),
        (None, Some(_)) => "ready".to_owned(),
        (None, None) => "reading".to_owned(),
    };
    let latency_seconds = match sounds.output() {
        Output::Open {
            latency_seconds, ..
        } => Some(*latency_seconds),
        Output::Closed | Output::Asked | Output::Failed(_) | Output::Absent(_) => None,
    };
    let next = SoundStats {
        table,
        cues_tick: mixdown.cues_tick,
        cues: mixdown.cues,
        voices: u32::try_from(mixdown.voices.len()).unwrap_or(u32::MAX),
        started: mixdown.started,
        stolen: mixdown.stolen,
        dropped: mixdown.dropped,
        inaudible: mixdown.inaudible,
        unmapped: mixdown.unmapped,
        refused: mixdown.refused,
        asked: u32::try_from(asked).unwrap_or(u32::MAX),
        loaded: u32::try_from(loaded).unwrap_or(u32::MAX),
        ready: load.ready,
        waiting,
        door: DoorStats {
            output: door.output.to_owned(),
            voices: door.voices.live,
            ready: door.sounds.ready,
            failed: door.sounds.failed,
            bytes_decoded: door.bytes_decoded,
            refused: door.refused.total(),
            dropped: door.voices.dropped,
            underruns: door.underruns,
            latency_seconds,
        },
    };
    stats.set_if_neq(next);
}
